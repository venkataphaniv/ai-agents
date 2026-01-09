# Build a Multi-Agent System With LangGraph

The future of AI isn’t about building a smarter chatbot; it’s about building a team. Today, we will build that team using LangGraph. We will create a Multi-Agent System where one AI agent acts as a Researcher (browsing the web), and another acts as a Writer (synthesising that info). They will pass work to each other like colleagues in a newsroom.

## What is LangGraph

We recommend you use LangChain if you want to quickly build agents and autonomous applications. Use LangGraph, a low-level agent orchestration framework and runtime, when you have more advanced needs that require a combination of deterministic and agentic workflows, heavy customization, and carefully controlled latency.

LangChain agents are built on top of LangGraph in order to provide durable execution, streaming, human-in-the-loop, persistence, and more. You do not need to know LangGraph for basic LangChain agent usage.

LangGraph is focused on the underlying capabilities important for agent orchestration: durable execution, streaming, human-in-the-loop, and more.

### Snippet

```python
    # pip install -qU langchain "langchain[anthropic]"

    from langchain.agents import create_agent

    def get_weather(city: str) -> str:
        """Get weather for a given city."""
        return f"It's always sunny in {city}!"

    agent = create_agent(
        model="claude-sonnet-4-5-20250929",
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
    )

    # Run the agent
    agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
    )
```

## Core benefits

LangGraph provides low-level supporting infrastructure for any long-running, stateful workflow or agent. LangGraph does not abstract prompts or architecture, and provides the following central benefits:

- Durable execution: Build agents that persist through failures and can run for extended periods, resuming from where they left off.

- Human-in-the-loop: Incorporate human oversight by inspecting and modifying agent state at any point.

- Comprehensive memory: Create stateful agents with both short-term working memory for ongoing reasoning and long-term memory across sessions.

- Debugging with LangSmith: Gain deep visibility into complex agent behavior with visualization tools that trace execution paths, capture state transitions, and provide detailed runtime metrics.

- Production-ready deployment: Deploy sophisticated agent systems confidently with scalable infrastructure designed to handle the unique challenges of stateful, long-running workflows.

## LangGraph ecosystem

While LangGraph can be used standalone, it also integrates seamlessly with any LangChain product, giving developers a full suite of tools for building agents. To improve your LLM application development, pair LangGraph with LangChain and LangSmith

Imagine a relay race. Runner A has the baton (data). They run their lap (task) and then pass the baton to Runner B. Runner B cannot start until they receive the baton.

LangGraph allows us to code this relay race.

- Nodes: These are the agents or functions (The Runners).
- Edges: These are the rules of who goes next (The Track).
- State: This is the shared memory (The Baton).

Instead of one giant prompt, we break the logic into small, reliable steps.

## The Setup

To keep this accessible and free, we are going to use Ollama to run a local LLM (Llama 3). This means you don’t need an OpenAI API key to follow along, though you will need a decent internet connection for the search tool.

### Prerequisites

- **Ollama Installed:** Download it from ollama.com.
- **Pull the Model:** Open your terminal and run: ollama pull llama3.
- **Run llama3:** In the terminal run: ollama run llama3

### Install the necessary libraries

```bash

    pip install langgraph langchain langchain-community langchain-ollama duckduckgo-search ddgs

```

## Getting Started

We will build this in three parts: The State, The Agents, and The Graph.

### Step 1: Defining the Shared State

Think of the AgentState as a shared clipboard that hangs on the office wall. Every agent can read from it and write to it. This ensures that when the Researcher finds something, the Writer can actually see it.

### Step 2: The Researcher Agent

Our first employee is the Researcher. Their job is simple: take a topic, search DuckDuckGo, and paste the results onto the clipboard (State)

Notice we aren’t using an LLM here yet! We are just using a deterministic tool (Search). This saves cost and reduces hallucinations. We are grounding the workflow in real data first.

### Step 3: The Writer Agent

Now, the Writer steps in. This agent uses Llama 3 (via Ollama). It reads the research_data found by the previous agent and drafts the content
The temperature=0.7 gives the model a bit of creativity. If you wanted a strict report, you might lower this to 0.1.

### Step 4: Wiring the Graph

This is the key part. We define the workflow. It is a linear path:

```flow
    Start -> Researcher -> Writer -> End
```

### Step 5: Run

We run the python script, to see the below typical output:

```bash
$ python ai-agents\multi-agent-system\multi_agent_system.py

Starting the Multi-Agent System...

---------------- INITIAL INPUTS ----------------

Topic: The future of AI Agents

Researcher is looking up: The future of AI Agents...
Research complete.
Writer is drafting the post...
Writing complete.

---------------- FINAL OUTPUT ----------------

**The Future of AI Agents: 5 Trends Redefining Business Value**

As we step into the new decade, the AI revolution is reaching a critical juncture. The latest trends report from Google Cloud, accompanied by the NotebookLM companion, highlights five key shifts that will reshape the landscape of AI agents and drive unprecedented value in the coming year.

**1. Agent-Powered Automation**

Big AI is moving away from static models and toward building agents that can take actions on our behalf. These assistants promise to simplify complex tasks, freeing us up to focus on high-value activities. With the rise of protocol-driven integrations, we'll see a surge in agent-enabled automation across industries.

**2. Cooperative Model Routing**

In 2026, expect to see smaller models delegating tasks to larger ones, creating a network effect that amplifies their collective capabilities. Whoever masters this system-level integration will dominate the market. Gabe Godhart's words echo our sentiment: "Whoever nails that system-level integration will shape the market."

**3. AI Energy Consumption**

As AI agents become more pervasive, energy consumption will become a pressing concern. Expect innovations in AI-powered data centers and cloud infrastructure to address this challenge head-on.

**4. Anthropic's Computer Use**

Anthropic's computer use has opened doors for AI-generated text, but what does the future hold? Will we see increased adoption of AI-generated content, or will concerns around authenticity and credibility lead to a more nuanced approach?

**5. Google Watermarking AI-Generated Text**

Google's watermarking initiative marks a significant shift toward authenticity in AI-generated content. This move could have far-reaching implications for industries that rely on high-quality text generation.

As we look ahead to 2026, it's clear that AI agents will fundamentally reshape business and drive new value. Stay tuned for more insights on these trends as they continue to evolve and transform the landscape of AI.
```
