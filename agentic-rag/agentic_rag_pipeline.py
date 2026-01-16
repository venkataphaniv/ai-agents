


import os
import argparse as ap
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


# Load PDFs from a folder
def load_docs(file: str | None, path: str | None):
    docs = []
    if file is not None and os.path.isfile(file) and file.endswith(".pdf"):
        print(f"Loading PDF: {file}")
        loader = PyPDFLoader(file)
        docs.extend(loader.load())
    elif path is not None:
        for f in os.listdir(path):
            if f.endswith(".pdf"):
                print(f"Loading PDF: {f}")
                loader = PyPDFLoader(os.path.join(path, f))
                docs.extend(loader.load())
                break
    else:
        print("No valid PDF file or path provided.")
        exit(1)
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
    p = ap.ArgumentParser()

    p.add_argument('-p', '--path', nargs='?', help = 'Provide folder name for reading pdfs')
    p.add_argument('-f', '--file', nargs='?', help = 'Provide file name for reading pdfs')
    args = p.parse_args()

    if args.file is None and args.path is None:
        print("Please provide either folder path using -p or --path, or a file using -f or --file")
        exit(1)
    # path to where your PDFs are stored

    docs = load_docs(args.file, args.path)
    print(f"PDF Pages Loaded: {len(docs)}")

    # Split PDFs into chunks
    text_splitter = RecursiveCharacterTextSplitter (
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = text_splitter.split_documents(docs)
    print("Chunks Created:", len(chunks))

    # Embeddings
    # all-MiniLM-L6-v2
    # Qwen/Qwen3-Embedding-0.6B
    # google/embeddinggemma-300m -- restricted access; must be authenticated to access it
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
    q = "Give me a 10 point summary from the PDF"
    print(q)
    print(rag_answer(llm, retriever, q), '\n')

    print("-" * 20)

    q = "Give me 5 areas of coverage from the PDF, with bullet points and brief explanations."
    print(q)
    print(rag_answer(llm, retriever, q), '\n')

    print("-" * 20)

    # Test 2: A general knowledge question
    q = "What is an Ideal Resume Format? Explain in 80 words."
    print(q)
    print(rag_answer(llm, retriever, q), '\n')


