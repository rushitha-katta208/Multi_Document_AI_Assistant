import gradio as gr

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- GLOBAL STATE ---
app_state = {
    "vector_store": None
}

llm_cache = None


# --- PROCESS PDF ---
def process_pdf(file, api_key):
    if not api_key:
        return "❌ Please enter Groq API Key"
    if file is None:
        return "❌ Please upload a PDF"

    try:
        loader = PyPDFLoader(file.name)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        split_docs = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        app_state["vector_store"] = FAISS.from_documents(split_docs, embeddings)

        return "✅ PDF indexed successfully! You can now chat."

    except Exception as e:
        return f"⚠ Error: {str(e)}"


# --- CHAT ---
def predict(message, history, api_key):
    global llm_cache

    if not api_key:
        return "❌ API Key missing"

    if app_state["vector_store"] is None:
        return "⚠ Please upload and process a PDF first."

    # create LLM once
    if llm_cache is None:
        llm_cache = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=api_key
        )

    # retrieve context
    docs = app_state["vector_store"].similarity_search(message, k=4)
    context = "\n\n".join([d.page_content for d in docs])

    # format history
    history_text = ""
    for user_msg, bot_msg in history:
        history_text += f"User: {user_msg}\nAssistant: {bot_msg}\n"

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer ONLY using the context.

Context:
{context}

Chat History:
{history}

Question:
{question}
""")

    chain = prompt | llm_cache

    response = chain.invoke({
        "context": context,
        "history": history_text,
        "question": message
    })

    return response.content


# --- UI ---
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 PDF RAG Assistant (Groq + LangChain)")

    with gr.Row():
        with gr.Column(scale=1):
            api_input = gr.Textbox(label="Groq API Key", type="password")
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Index PDF")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            chat = gr.ChatInterface(
                fn=predict,
                additional_inputs=[api_input],
                examples=[
                    ["Summarize this document", ""],
                    ["What are the key points?", ""],
                    ["Explain in simple terms", ""]
                ]
            )

    upload_btn.click(
        process_pdf,
        inputs=[file_input, api_input],
        outputs=[status]
    )

demo.launch(theme=gr.themes.Soft())
