# 🦙 multimodal-rag-system

A **Retrieval-Augmented Generation (RAG) system** for answering questions grounded in your PDF documents, powered by [LangChain](https://python.langchain.com/), [ChromaDB](https://www.trychroma.com/), and [Gradio](https://gradio.app/).

---

## 🚀 Features

- **PDF Document Loading:** Load and split PDF files into manageable chunks for efficient retrieval.
- **Embeddings & Vector Store:** Generate embeddings for document chunks and store them in a persistent ChromaDB vector store.
- **RAG Retriever:** Retrieve the most relevant document chunks for a user query using semantic search.
- **Agentic Answering:** An LLM agent synthesizes answers, citing or summarizing retrieved content.
- **Interactive Gradio UI:** User-friendly web interface for asking questions and viewing answers.
- **Sample PDFs Included:** Try out the system with included tutorial PDFs.

---

## 🖥️ Demo

![Gradio UI Screenshot](https://raw.githubusercontent.com/manju07/multimodal-rag-system/main/assets/demo_screenshot.png)

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/manju07/multimodal-rag-system.git
   cd multimodal-rag-system
   uv venv
   source .venv/bin/activate
   ```

2. **Install dependencies (with [uv](https://github.com/astral-sh/uv) for faster installs):**
   ```bash
   uv pip install -r requirements.txt or uv add -r requirments.txt
   ```

3. **Download or add your PDF files:**
   - Place your PDFs in the `data/pdf_files/` directory.
   - Sample PDFs are already included.

4. **Set up environment variables:**
   - Copy `.env.example` to `.env` and fill in your OpenAI API key and any other required variables.

---

## 🏃‍♂️ Usage

Start the Gradio app:
