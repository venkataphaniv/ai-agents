


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline


# Load PDFs from a folder
def load_docs(path: str):
    docs = []
    for file in os.listdir(path):
        if file.endswith(".pdf"):
            print(f"Loading PDF: {file}")
            loader = PyPDFLoader(os.path.join(path, file))
            docs.extend(loader.load())
            break
    return docs


# Agent brain
def agent_controller(query):
    q = query.lower()
    if any(word in q for word in ["pdf", "document", "data", "summarize", "information", "find"]):
        return "search"
    return "direct"


# RAG
def rag_answer(llm, retriever, query):
    action = agent_controller(query)

    if action == "search":
        print(f"🕵️ Agent decided to SEARCH document for: '{query}'")
        results = retriever.invoke(query)
        context = "\n".join([r.page_content for r in results])
        final_prompt = f"Use this context:\n{context}\n\nAnswer:\n{query}"
    else:
        print(f"🤖 Agent decided to answer DIRECTLY: '{query}'")
        final_prompt = query

    res = llm(final_prompt)[0]["generated_text"]
    return res



if __name__=="__main__":
    # path to where your PDFs are stored
    pth = 'C:\\Reading-Material\\Robotics, Mechatronics'
    docs = load_docs(pth)
    print(f"PDF Pages Loaded: {len(docs)}")

    # Split PDFs into chunks
    text_splitter = RecursiveCharacterTextSplitter (
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = text_splitter.split_documents(docs)
    print("Chunks Created:", len(chunks))

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Save texts into Chroma vector DB
    texts = [c.page_content for c in chunks]
    db = Chroma(
        collection_name="rag_store",
        embedding_function=embedding_model
    )

    db.add_texts(texts)

    # Retriever
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Local LLM
    llm = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=150
    )

    print(f"\n\n=== Running RAG Agentic Pipeline ===")
    print("-" * 80, "\n")

    print(">>> Test Queries:\n")

    # Test 1: A document-specific question
    q = "Give me a 5-point summary from the PDF"
    print(q)
    print(rag_answer(llm, retriever, q))

    print("-" * 20)

    # Test 2: A general knowledge question
    q = "What is an Ideal Resume Format? Explain in 80 words."
    print(q)
    print(rag_answer(llm, retriever, q))


