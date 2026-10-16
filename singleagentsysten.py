
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama

# Initialize the local model
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
    num_predict=256,
    timeout=60
)

# Tool 1: Addition
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return their sum."""
    return a + b

# Tool 2: Multiplication
@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two integers and return their product."""
    return a * b

# Tool 3: Greeting
@tool
def get_greeting(name: str) -> str:
    """Generate a greeting for the specified person."""
    return f"Hello, {name}! Welcome to AI agents."

# Create the agent
agent = create_agent(
    model=llm,
    tools=[add_numbers, multiply_numbers, get_greeting],
    system_prompt=(
        "You are a helpful AI agent. "
        "Use add_numbers for addition, multiply_numbers for multiplication, "
        "and get_greeting to greet people. "
        "Call the tools with the correct arguments. "
        "After receiving the tool results, summarize all results clearly. "
        "Do not call a tool again if its task is already completed."
    )
)

# Run the agent
if __name__ == "__main__":
    print("Starting AI agent...")

    try:
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Add 10 and 20, multiply 5 and 6, "
                            "and greet Johnson."
                        )
                    }
                ]
            },
            config={"recursion_limit": 10}
        )

        # Print only the final AI response
        for message in reversed(result["messages"]):
            if message.type == "ai" and message.content:
                print("\nAgent response:")
                print(message.content)
                break

    except Exception as e:
        print(f"\nAgent error: {type(e).__name__}: {e}")
