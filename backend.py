# src/backend.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Global variables (important)
vector_store = None
llm = ChatOllama(model="llama3")


# 1️⃣ Load PDF and create vector store
def load_pdf(pdf_path):
    global vector_store

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS vector store
    vector_store = FAISS.from_documents(split_docs, embeddings)

    print("✅ PDF processed and vector store created.")


# 2️⃣ Get Answer (RAG)
def get_answer(query: str) -> str:
    global vector_store

    if vector_store is None:
        return "⚠ Please upload a PDF first."

    # Retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(query, k=4)
    context_text = "\n\n".join(
        [doc.page_content for doc in retrieved_docs]
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant.
Use ONLY the below context to answer.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""
    )

    formatted_prompt = prompt.format(
        context=context_text,
        question=query
    )

    # LLM response
    response = llm.invoke(formatted_prompt)

    return response.content
