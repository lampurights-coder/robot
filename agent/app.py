
import json
import os
from typing import Annotated

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from gmail_tool import send_gmail

load_dotenv(override=True)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.gapgpt.app/v1"),
)


@tool
def langsearch(query: str) -> str:
    """Search the web for current information and return source URLs with context."""
    response = requests.post(
        "https://api.langsearch.com/v1/web-search",
        headers={
            "Authorization": f"Bearer {os.environ['LANGSEARCH_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "count": 5,
            "contents": {"text": True},
            "freshness": "noLimit",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    sources = [
        {
            "title": item.get("name"),
            "url": item["url"],
            "context": item.get("text") or item.get("snippet") or "",
        }
        for item in payload["data"]["webPages"]["value"]
    ]
    return json.dumps({"sources": sources}, ensure_ascii=False)


tools = [langsearch, send_gmail]
llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot_node(state: State) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)
builder.add_edge("tools", "chatbot")

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "is ayatollah khamenei alive?"}]}
    )
    print(result["messages"][-1].content)
