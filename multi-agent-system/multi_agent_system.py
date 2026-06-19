"""
Multi-Agent System for Research and Content Creation
This module defines a multi-agent system where agents collaborate to research a topic
and create a blog post based on their findings.
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun

# ChatOllama from langchain-ollama package
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ----- Shared State -----
class AgentState(TypedDict):
    topic: str
    research_data: List[str]  # A list of findings
    blog_post: str            # The final output


def researcher_node(state: AgentState):
    topic = state["topic"]
    print(f"Researcher is looking up: {topic}...")

    search = DuckDuckGoSearchRun()

    try:
        # You can tweak this query as you like
        results = search.run(f"key facts and latest news about {topic}")
    except Exception as e:
        results = f"Could not find data: {e}"

    print("Research complete.")

    # Only return the keys you want to update
    return {"research_data": state.get("research_data", []) + [results]}


def writer_node(state: AgentState):
    print("Writer is drafting the post...")

    topic = state["topic"]
    data = state["research_data"][-1] if state["research_data"] else ""

    llm = ChatOllama(model="llama3.3", temperature=0.8)

    prompt = ChatPromptTemplate.from_template(
        """You are a technical blog writer, a deep researcher, and tech enthusiast.
        Your task is to write an engaging blog post OF 10000 words about the following "{topic}",
        based ONLY on the following research data:

        {data}

        Return the blog post content."""
    )

    chain = prompt | llm
    res = chain.invoke({"topic": topic, "data": data})

    print("Writing complete.")
    return {"blog_post": res.content}


if __name__ == "__main__":
    # ----- Build the LangGraph -----
    print("Starting the Multi-Agent System...\n")
    print("Building LangGraph...\n")

    print("---------------- INITIAL INPUTS ----------------\n")
    # topic = "The future of AI Agents, autonomous agents, and their impact on society. The latest advancements, ethical considerations, and potential use cases."
    topic = "How to develop a robot, from a developers perspective. Programming language considerations, hardware choices, and the future of robotics. Also, good examples of robots in the world and how they work. Full working code examples (repositories) which will help the developers"
    print(f"Topic: {topic}\n")

    inputs: AgentState = {
        "topic": topic,
        "research_data": [],
        "blog_post": "",
    }

    # Define the workflow
    wf = StateGraph(AgentState)

    # Add nodes
    wf.add_node("Researcher", researcher_node)
    wf.add_node("Writer", writer_node)

    # Flow: Start -> Researcher -> Writer -> END
    wf.set_entry_point("Researcher")

    # Add edges
    wf.add_edge("Researcher", "Writer")
    wf.add_edge("Writer", END)

    # Compile the workflow
    app = wf.compile()

    result = app.invoke(inputs)

    print("\n---------------- FINAL OUTPUT ----------------\n")
    print(result["blog_post"])
