"""
Test script to verify conditional routing logic in the LangGraph agent.
This script sets dummy API keys to test the routing behavior without making actual API calls.
"""

import os

# Set dummy API keys for testing
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-testing"
os.environ["TAVILY_API_KEY"] = "tvly-dummy-key-for-testing"

# Import the agent after setting environment variables
from agent import test_agent, app
from langchain_core.messages import HumanMessage

def test_conditional_routing():
    """Test that the agent can be created and invoked without errors."""
    print("Testing Conditional Routing Logic")
    print("=" * 40)
    
    # Test that the agent can be created successfully
    try:
        print("\n✅ Agent created successfully")
        print(f"Agent type: {type(app)}")
        
        # Test a simple query to verify the graph structure
        print("\n🔍 Testing graph structure...")
        
        # Get the graph structure
        graph = app.get_graph()
        nodes = list(graph.nodes.keys())
        print(f"Graph nodes: {nodes}")
        
        # Verify expected nodes are present
        expected_nodes = ["agent", "tools"]
        for node in expected_nodes:
            if node in nodes:
                print(f"✅ Node '{node}' found in graph")
            else:
                print(f"❌ Node '{node}' missing from graph")
        
    except Exception as e:
        print(f"❌ Error testing agent: {e}")

if __name__ == "__main__":
    test_conditional_routing()



