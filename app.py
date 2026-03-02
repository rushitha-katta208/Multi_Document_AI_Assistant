import gradio as gr
from src.backend import get_answer, load_pdf

# Store chat history
chat_history = []

# Function to handle question answering
def chat_with_pdf(question):
    if not question:
        return chat_history

    # Append user question
    chat_history.append({"role": "user", "content": question})

    # Get answer from your backend
    answer = get_answer(question)

    # Append assistant answer
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history

# Function to upload PDF
def upload_pdf(file):
    if file is None:
        return "Please upload a PDF."
    load_pdf(file.name)
    return "✅ PDF uploaded and processed successfully!"

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 📄 Smart PDF QA Chatbot
        ### Ask Questions, Get Instant Answers 😊
        Upload a PDF and ask questions intelligently.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(label="📂 Upload PDF", file_types=[".pdf"])
            upload_btn = gr.Button("Process PDF")
            upload_status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                placeholder="Ask something about your PDF...",
                lines=2
            )
            send_btn = gr.Button("Send")

    # Upload action
    upload_btn.click(
        upload_pdf,
        inputs=pdf_file,
        outputs=upload_status
    )

    # Chat action
    send_btn.click(
        chat_with_pdf,
        inputs=msg,
        outputs=chatbot
    )

if __name__ == "__main__":
    demo.launch(share=True)
