from mcp.server import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a:int,b:int)->int:
    """summary
    add two  numbers
    """
    return a+b

@mcp.tool()
def subtract(a:int,b:int)->int:
    """summary
    subtract two numbers
    """
    return a-b

@mcp.tool()
def multiply(a:int,b:int)->int:
    """summary
    multiply two numbers
    """
    return a*b  

@mcp.tool()
def divide(a:int,b:int)->float:     
    """summary
    divide two numbers
    """
    if b==0:
        raise ValueError("Division by zero is not allowed.")
    return a/b

@mcp.tool()
def get_weather(location:str)->str:
    """Get weather information for a location"""
    if location.lower() == "new york":
        return "The weather in New York is very hot"
    return f"Weather information for {location} is not available"

#the transport "stdio" atgument tllls the server to :
#use stadnard input and output for communication with the client.

if __name__=="__main__":
    mcp.run(transport="stdio")