# Building an Agentic RAG Pipeline

A typical RAG system quickly searches your uploaded tax documents or technical manuals to find a relevant snippet, even for something as simple as a casual greeting. Agentic RAG works differently. Instead of always looking up information, the agent stops to consider whether it really needs to search or if it can answer on its own. Below, we'll see how to build an Agentic RAG pipeline using Python and LangChain.

## Agentic RAG Pipeline: Getting Started

In this example, we’ll build a local, privacy-friendly Agentic RAG pipeline using Python, LangChain, and a lightweight Google model. We’ll go beyond just writing code to create a system that acts a bit more like a person.

We’ll use LangChain to manage the process, ChromaDB for storing vectors, and Google’s Flan-T5 as our local language model. Everything runs on your own computer, so you don’t need any API keys.

You will need a few libraries installed. In your terminal:

    ```python

    pip install langchain langchain-community langchain-chroma transformers sentence-transformers pypdf
    ```

## Step 1: Loading the Knowledge

First, we need to give our AI something to read. We use a function to scan a folder for PDFs. We go through a folder, find PDF files, and load them one page at a time.

## Step 2: Chunking

LLMs can only read a certain amount of text at once, called the context window. Even if they could handle more, giving them a whole 500-page book to answer one question isn’t efficient.

## Step 3: Embeddings & Vector Store

Now we convert text into numbers, called vectors, that the computer can understand. We store these in Chroma, which is a vector database. We’re using all-MiniLM-L6-v2. It’s a small, fast model that’s great for local development. It puts similar concepts close together in space. When we search later, we won’t be matching keywords; we’ll be matching meanings.

## Step 4: The Brain

We need a model to generate the actual answers. We are using google/flan-t5-base. Flan-T5 is a “seq2seq” model. It’s great at following instructions like “Summarize this” or “Answer this,” which makes it ideal for RAG tasks even though it’s smaller.

## Step 5: The Agent

This is the key part. This simple function ***'agent_controller'*** is what makes the pipeline Agentic

Instead of sending everything to the database, this controller analyzes the user’s intent:

    1. Does the user want data from the file? Action: Search
    2. Is the user just chatting or asking for general knowledge? Action: Direct

In a production system, you might use a powerful LLM to make this decision. But for learning, this keyword-based method is a great way to show the Routing Pattern in agentic AI.

## Step 6: The Execution Loop

Finally, we tie it all together:

- In the **first case**, the agent sees the word “PDF” or “summary” and uses the retriever.
- In the **second case**, it knows it doesn’t need your documents to explain a resume format, so it answers using its own pre-trained knowledge.

## Closing Thoughts

When I first started working with AI, I thought bigger was better. I wanted the largest model and the biggest database. But over time, I learned that intelligence is really about efficiency, not just raw power.

By building this Agentic router, you save computing resources and reduce latency. Most importantly, you create a system that respects the user’s context.
