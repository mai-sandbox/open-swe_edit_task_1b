"""
React Agent Implementation

A LangGraph-based React agent for handling user queries with conversation memory.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables
load_dotenv()

def create_agent():
    """
    Creates a React agent for handling user queries.
    
    Returns:
        A configured LangGraph React agent with memory capabilities.
    """
    
    # Initialize the language model
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Available tools for the agent
    tools = []
    
    # Create memory checkpointer for conversation persistence
    checkpointer = InMemorySaver()
    
    # Create the React agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer
    )
    
    return agent

def test_agent():
    """Test the agent with a sample query."""
    
    agent = create_agent()
    
    # Configuration for conversation threading
    config = {"configurable": {"thread_id": "conversation-1"}}
    
    # Test query
    query = "What are the latest developments in AI in 2024?"
    
    print("Testing Agent Implementation")
    print("=" * 30)
    print(f"\nQuery: {query}")
    print("-" * 30)
    
    try:
        # Invoke the agent
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config
        )
        
        # Get the final message
        final_message = result["messages"][-1]
        print(f"Response: {final_message.content[:200]}...")
        print("✅ Agent executed successfully")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return True

if __name__ == "__main__":
    # Verify environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Missing OPENAI_API_KEY environment variable")
        exit(1)
    
    # Test the implementation
    test_agent()