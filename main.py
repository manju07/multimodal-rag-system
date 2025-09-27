import os
import asyncio
from dotenv import load_dotenv

from model.pdf_document_loader import PDFDocumentsLoader
from model.embedding_manager import EmbeddingManager
from model.vector_store import VectorStore
from model.rag_retriever import RAGRetriever
from agents import Agent, Runner, function_tool

import gradio as gr

class RAGSystem:

    def __init__(self):
        print("Hello from main.py!")
        load_dotenv(override=True)
        print("Loaded environment variables")

        self.rag_retriever = None
        self.rag_agent = None

    def initialize_rag_retriever(self):
        print("Hello from multimodal-rag-system!")
        pdf_documents_loader = PDFDocumentsLoader()
        pdf_documents = pdf_documents_loader.load_pdf_documents()
        pdf_documents_chunks = pdf_documents_loader.split_documents(pdf_documents)
        texts = [doc.page_content for doc in pdf_documents_chunks]
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store = VectorStore()
        vector_store.add_documents(pdf_documents_chunks, embeddings)
        self.rag_retriever = RAGRetriever(vector_store, embedding_manager)
        return self.rag_retriever

    def load_rag_agent(self):
        instructions = (
            "You are a Retrieval-Augmented Generation (RAG) agent. "
            "When a user asks a question, use the 'retrieve_with_rag_retriever' tool to search for the most relevant information "
            "from the document collection. Synthesize a clear, accurate answer using the retrieved data. "
            "If the information is insufficient, state that explicitly."
        )
        # To avoid Pydantic schema generation errors with RAGRetriever, we wrap the tool in a closure
        def retrieve_with_rag_retriever(query: str):
            """Retrieve relevant documents using rag_retriever."""
            return self.rag_retriever.retrieve(query)
        # Decorate the closure with function_tool
        tool = function_tool(retrieve_with_rag_retriever)
        self.rag_agent = Agent(
            name="RAG-Agent",
            tools=[tool],
            instructions=instructions,
            model="gpt-4o-mini"
        )
        return self.rag_agent

    async def run_rag_agent(self, query):
        if self.rag_agent is None:
            return "RAG agent is not initialized."
        return await Runner.run(self.rag_agent, query)

    def launch_gradio(self):
        with gr.Blocks() as iface:
            gr.Markdown("# 🦙 RAG Agent Demo")
            gr.Markdown("Ask a question and get an answer using the RAG agent. Try something like: **What is agentic AI?**")
            with gr.Row():
                with gr.Column():
                    query = gr.Textbox(lines=3, label="Ask a question", placeholder="Type your question here...")
                    submit_btn = gr.Button("Submit", variant="primary")
                with gr.Column():
                    output = gr.Textbox(label="RAG Agent Response", lines=8, interactive=False)
            submit_btn.click(fn=self.run_rag_agent, inputs=query, outputs=output)
        iface.launch()

if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_retriever = rag_system.initialize_rag_retriever()
    rag_system.load_rag_agent()
    rag_system.launch_gradio()
