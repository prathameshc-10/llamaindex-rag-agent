import asyncio
import os
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv

load_dotenv()

# Define your calculator tool
def multiply(x, y):
    """Useful for multiplying two numbers"""
    return x * y

def add(x, y):
    """Useful for adding two numbers"""
    return x + y

# Use Google SDK class
llm = GoogleGenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
)

# Construct the agent
agent = FunctionAgent(
    name="Calc",
    description="Multiply two numbers",
    tools=[multiply, add],
    llm=llm,
)

# Standard asyncio wrapper to handle the workflow properly
async def main():
    response = await agent.run("What is 1234 + 4567?")
    print(f"Agent response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
