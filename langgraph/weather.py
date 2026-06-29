from mcp.server.fastmcp import FastMCPServer

@mcp.tool():
async def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # Here you would implement the logic to fetch weather data from an API or database.
    # For demonstration purposes, we'll return a mock response.
    return f"The current weather in {location} is sunny with a temperature of 25°C."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")