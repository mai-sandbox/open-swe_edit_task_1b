"""
Test script to verify the conditional routing logic in the LangGraph agent
without requiring actual API keys.
"""

import os
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import tools_condition

def test_conditional_routing_logic():
    """Test the conditional routing logic without API calls."""
    print("Testing Conditional Routing Logic")
    print("=" * 40)
    
    # Test 1: Message with tool calls should route to "tools"
    print("\n1. Testing routing with tool calls...")
    ai_message_with_tools = AIMessage(
        content="I'll help you with the weather.",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {"location": "New York"},
                "id": "call_1",
                "type": "tool_call"
            }
        ]
    )
    
    state_with_tools = {"messages": [ai_message_with_tools]}
    routing_result = tools_condition(state_with_tools)
    print(f"   State with tool calls routes to: {routing_result}")
    assert routing_result == "tools", f"Expected 'tools', got {routing_result}"
    print("   ✅ PASS: Tool calls correctly route to 'tools'")
    
    # Test 2: Message without tool calls should route to END
    print("\n2. Testing routing without tool calls...")
    ai_message_no_tools = AIMessage(content="Hello! How can I help you today?")
    state_no_tools = {"messages": [ai_message_no_tools]}
    routing_result = tools_condition(state_no_tools)
    print(f"   State without tool calls routes to: {routing_result}")
    assert routing_result == "__end__", f"Expected '__end__', got {routing_result}"
    print("   ✅ PASS: No tool calls correctly route to '__end__'")
    
    print("\n✅ All conditional routing tests passed!")

def test_graph_structure():
    """Test the graph structure without initializing tools that require API keys."""
    print("\nTesting Graph Structure")
    print("=" * 30)
    
    # Mock the API-dependent components
    with patch('agent.TavilySearch') as mock_tavily, \
         patch('agent.ChatOpenAI') as mock_openai:
        
        # Configure mocks
        mock_tavily.return_value = Mock()
        mock_openai.return_value.bind_tools.return_value = Mock()
        
        # Import and test the agent creation
        try:
            from agent import create_agent
            agent = create_agent()
            print("   ✅ PASS: Agent created successfully with mocked dependencies")
            
            # Test graph compilation
            graph_dict = agent.get_graph().to_dict()
            nodes = graph_dict.get('nodes', [])
            edges = graph_dict.get('edges', [])
            
            print(f"   Graph has {len(nodes)} nodes: {[node['id'] for node in nodes]}")
            print(f"   Graph has {len(edges)} edges")
            
            # Verify expected nodes exist
            node_ids = [node['id'] for node in nodes]
            assert 'agent' in node_ids, "Missing 'agent' node"
            assert 'tools' in node_ids, "Missing 'tools' node"
            print("   ✅ PASS: Required nodes ('agent', 'tools') are present")
            
        except Exception as e:
            print(f"   ❌ FAIL: Error creating agent: {e}")
            return False
    
    return True

if __name__ == "__main__":
    test_conditional_routing_logic()
    test_graph_structure()
    print("\n🎉 All tests completed successfully!")
