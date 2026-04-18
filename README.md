
# 🤖 Multi-Document AI Assistant

A Multi-Document AI Assistant powered by **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**. Upload multiple PDFs and ask questions — the bot retrieves relevant content and generates accurate, context-aware answers in real time.

---

## 🌟 Key Features

- 📄 Upload single or multiple PDF files and ask questions interactively
- 🔍 Uses **RAG (Retrieval-Augmented Generation)** to fetch the most relevant content from PDFs
- 🧠 Powered by **LLaMA3 via LangChain** for natural language understanding and answer generation
- 📦 **FAISS Vector Store** for fast and efficient semantic search over document embeddings
- 🖥️ Clean and user-friendly web interface built with **Gradio**

---

## 🏗️ Architecture

```
PDF Input
   ↓
Text Extraction & Cleaning
   ↓
Text Chunking
   ↓
Embedding Generation (HuggingFace)
   ↓
FAISS Vector Store
   ↓
User Query → Semantic Search → Relevant Chunks
   ↓
LLaMA3 (via LangChain) → Context-Aware Answer
   ↓
Gradio Web Interface
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core programming language |
| LangChain | QA workflow and LLM orchestration |
| LLaMA3 | Large Language Model for answer generation |
| HuggingFace | Embedding model for semantic search |
| FAISS | Vector store for storing and retrieving embeddings |
| Gradio | Web-based user interface |
| PyPDF2 | PDF text extraction |

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/rushitha-katta208/Smart-PDF-QA-Bot.git
cd Smart-PDF-QA-Bot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:7860
```

---

## 📸 Screenshots

> Upload a PDF and start asking questions instantly!

![Multi-Document AI Assistant Interface](screenshots/demo.png)

---

## 🔮 Future Improvements

- [ ] Support for more file formats (DOCX, TXT, CSV)
- [ ] Multi-language support for non-English PDFs
- [ ] Chat history and conversation memory
- [ ] Cloud deployment (Hugging Face Spaces / AWS)
- [ ] Support for larger documents with better chunking strategies

---

## 👩‍💻 Author

**Rushitha Katta**
- 📧 krushitha7@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/rushitha-katta)
- 💻 [GitHub](https://github.com/rushitha-katta208)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
