import os
import uuid
import json
from dotenv import load_dotenv

from typing import List, Annotated
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

# Ensure GOOGLE_API_KEY is set
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("Set GOOGLE_API_KEY in your environment (GOOGLE_API_KEY).")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
)

# Collect info via tool-call
template = """Your job is to collect requirements for a prompt template.

Collect exactly these items:
- objective (what the prompt should accomplish)
- variables (list of variable names that will be provided)
- constraints (what NOT to do)
- requirements (what MUST be done)

If any info is missing, ask concise follow-up questions.
When you have ALL items, call the PromptInstructions tool with the structured data.
"""

def get_messages_info(messages: List):
    return [SystemMessage(content=template)] + messages

class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""
    objective: str = Field(..., description="What the prompt should accomplish")
    variables: List[str] = Field(default_factory=list, description="List of variable names")
    constraints: List[str] = Field(default_factory=list, description="What NOT to do")
    requirements: List[str] = Field(default_factory=list, description="What MUST be done")

llm_with_tool = llm.bind_tools([PromptInstructions])

def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

# Generate the actual prompt template
prompt_system = """You are a world-class prompt engineer.
Given the collected requirements, write a clear, reusable prompt template.

Guidelines:
- Use braces {{var}} for variables.
- Include brief sections: Objective, Inputs (variables), Constraints, Requirements, and Final Task.
- Be concise and actionable.
- Return ONLY the final prompt template text (no extra commentary).

Requirements (JSON):
{reqs}
"""

def get_prompt_messages(messages: List):
    tool_args = None

    # Find the latest AIMessage that contains a tool call with our collected data
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            # Prefer the most recent tool call if multiple
            tool_args = m.tool_calls[-1].get("args")

    if not tool_args:
        raise ValueError("No PromptInstructions tool call found in message history.")
    sys = SystemMessage(content=prompt_system.format(
        reqs=json.dumps(tool_args, ensure_ascii=False, indent=2)
    ))
    user = HumanMessage(content="Generate the final prompt template now.")
    return [sys, user]

def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}


# Routing logic
def get_state(state):
    last = state["messages"][-1]
    # If the model just called the tool, we acknowledge and move to prompt generation
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "add_tool_message"
    # If model asked a follow-up (AIMessage w/o tool call), end this run; next user input continues
    if isinstance(last, AIMessage):
        return END
    # If last is human, continue collecting info
    if isinstance(last, HumanMessage):
        return "info"
    return END

class State(TypedDict):
    messages: Annotated[list, add_messages]


# Graph construction
memory = InMemorySaver()
workflow = StateGraph(State)

workflow.add_node("info", info_chain)
workflow.add_node("prompt", prompt_gen_chain)

@workflow.add_node
def add_tool_message(state: State):
    # Acknowledge the tool call so LangGraph has a ToolMessage bridging the call
    return {
        "messages": [
            ToolMessage(
                content="Requirements captured.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        ]
    }

workflow.add_conditional_edges("info", get_state, ["add_tool_message", "info", END])
workflow.add_edge("add_tool_message", "prompt")
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")

graph = workflow.compile(checkpointer=memory)

# Simple CLI runner
def run_cli():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("PromptSmith CLI (q to quit)")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            break

        if user.lower() == "q":
            print("AI: Byebye")
            break

        output = None
        for update in graph.stream(
            {"messages": [HumanMessage(content=user)]},
            config=config,
            stream_mode="updates",
        ):
            node, payload = next(iter(update.items()))
            msg = payload["messages"][-1]

            role = "AI"
            if isinstance(msg, HumanMessage): role = "You"
            elif isinstance(msg, ToolMessage): role = "Tool"

            print(f"{role}: {getattr(msg, 'content', '')}")
            output = update

        if output and "prompt" in output:
            print("Done!")

if __name__ == "__main__":
    run_cli()

#next step is to export it as function which returns only prompt finally to our main