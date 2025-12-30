"""Main entry point for the LangGraph agent."""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from .agent import create_agent


def main():
    """Run the agent with example queries."""
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Create the agent
    print("Creating agent...")
    agent = create_agent()
    
    # Example queries
    queries = [
        "What's the weather in San Francisco?",
        "Calculate 15 * 7 + 23",
        "What's the weather in Tokyo and what's 100 divided by 4?"
    ]
    
    for query in queries:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Run the agent
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # Print the final response
        final_message = result["messages"][-1]
        print(f"Response: {final_message.content}")


if __name__ == "__main__":
    main()