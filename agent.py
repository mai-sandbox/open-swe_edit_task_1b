"""
Decision-Making Agent Implementation

A LangGraph agent that can use various tools and provide intelligent responses
based on user queries. Features weather, math, and knowledge search capabilities.
"""

import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Define tools
@tool
def get_weather(location: str) -> str:
    """Get current weather information for a location."""
    return f"The weather in {location} is sunny with 72°F temperature."

@tool
def calculate_math(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except:
        return "Unable to calculate that expression."


web_search = TavilySearch(max_results=3)
tools = [get_weather, calculate_math, web_search]
tool_node = ToolNode(tools)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

def call_model(state: MessagesState):
    """
    Call the model with bound tools to generate responses.
    """
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

def create_agent():
    """
    Creates an intelligent agent with tool capabilities.
    """
    # Create the workflow
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", tool_node)
    
    # Add basic edges
    workflow.add_edge(START, "agent")
    workflow.add_edge("tools", "agent")
    workflow.add_edge("agent", END)
    
    # Add memory checkpointer
    checkpointer = InMemorySaver()
    
    # Compile the graph
    app = workflow.compile(checkpointer=checkpointer)
    
    return app

app = create_agent()

def test_agent():
    """Test the agent with various query types."""
    
    agent = app
    
    test_cases = [
        "What's the weather in New York?",
        "Calculate 15 * 24", 
        "What is Python programming language?",
        "Hello, how are you?",
    ]
    
    print("Testing Agent Implementation")
    print("=" * 30)
    
    for i, query in enumerate(test_cases, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 30)
        
        config = {"configurable": {"thread_id": f"test-{i}"}}
        
        try:
            result = agent.invoke(
                {
                    "messages": [HumanMessage(content=query)]
                },
                config
            )
            
            final_message = result["messages"][-1]
            print(f"Response: {final_message.content[:150]}...")
            print("✅ Agent executed successfully")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY environment variable")
        exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Missing TAVILY_API_KEY environment variable")
        exit(1)
    
    test_agent()


