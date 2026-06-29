from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()


async def main():
    client = MultiServerMCPClient(
        {
            "main": {
                "command": "python",
                "args": ["langgraph/mathserver.py"],
                "transport": "stdio",
            }
        }
    )

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

    tools = await client.get_tools()
    model = ChatGroq(model="llama-3.3-70b-versatile")

    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)

    print("\n=== Math Request ===")
    response = await model_with_tools.ainvoke([HumanMessage(content="What is the sum of 5 and 3?")])
    print("Response:", response.content if hasattr(response, 'content') else response)

    print("\n=== Multiplication Request ===")
    response = await model_with_tools.ainvoke([HumanMessage(content="What is 7 multiplied by 6?")])
    print("Response:", response.content if hasattr(response, 'content') else response)

    print("\n=== Weather Request ===")
    response = await model_with_tools.ainvoke([HumanMessage(content="What is the weather in New York?")])
    print("Response:", response.content if hasattr(response, 'content') else response)

asyncio.run(main())