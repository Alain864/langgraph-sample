"""LangGraph agent implementation."""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, ToolMessage
from .tools import tools


class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent():
    """Create and compile the agent graph."""
    
    # Initialize the LLM with tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    def call_model(state: AgentState):
        """Call the model with the current state."""
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def call_tools(state: AgentState):
        """Execute the tools that the model wants to call."""
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_calls = last_message.tool_calls
        
        # Create a mapping of tool names to tool functions
        tools_by_name = {tool.name: tool for tool in tools}
        
        # Execute each tool call
        tool_messages = []
        for tool_call in tool_calls:
            tool = tools_by_name[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            )
        
        return {"messages": tool_messages}
    
    def should_continue(state: AgentState):
        """Determine if we should continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are no tool calls, we're done
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()