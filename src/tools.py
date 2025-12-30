"""Custom tools for the agent."""

from langchain_core.tools import tool


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.
    
    Args:
        location: The city and state, e.g. San Francisco, CA
    """
    # This is a mock implementation
    return f"The weather in {location} is sunny and 72°F"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate, e.g. "2 + 2"
    """
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


# List of all available tools
tools = [get_weather, calculate]