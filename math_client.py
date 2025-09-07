# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, ToolMessage

import asyncio

from agent import   # Import the Azure LLM function

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["d:\\python\\mcp_rag\\math_server.py"],
)

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(get_azure_llm(), tools)
            final = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})


    # ----------------------------------------
    # Optional: Inspect Tool Usage Trace
    # ----------------------------------------
    def inspect_tool_usage(trace: dict) -> None:
        """Print tool name, parameters, result, and final answer from trace."""
        if not isinstance(trace, dict) or "messages" not in trace:
            print("Unexpected trace format")
            return

        for msg in trace["messages"]:
            # Detect tool call(s) inside AI message
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"🔧 Tool requested: {call['name']}")
                    print(f"   ↳ parameters: {call['args']}")
            # Print result returned by the tool
            if isinstance(msg, ToolMessage):
                print("Tool result:")
                print(msg.content.strip())
            # Print the final LLM answer
            if isinstance(msg, AIMessage) and msg.content:
                print("Final answer:")
                print(msg.content)

    # Inspect what happened during execution
    inspect_tool_usage(final)

if __name__ == "__main__":
    asyncio.run(main())