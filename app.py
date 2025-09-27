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
            "from the document collection. For each answer, clearly cite or summarize the most relevant retrieved content, "
            "and synthesize a concise, well-structured response that directly addresses the user's question. "
            "If the retrieved information is insufficient or inconclusive, state this clearly and suggest what additional information might be needed. "
            "Always strive for clarity, completeness, and transparency in your answers."
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
        with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as iface:
            # Header with logo and title
            with gr.Row():
                with gr.Column(scale=1, min_width=80):
                    gr.HTML(
                        '<div style="display: flex; align-items: center; justify-content: center; height: 100%;">'
                        '<span style="font-size:3em;">🦙</span>'
                        '</div>'
                    )
                with gr.Column(scale=8):
                    gr.Markdown(
                        """
                        <div style="font-size:2.2em; font-weight: bold; letter-spacing: -1px; color: #1a237e;">
                            RAG Agent: Retrieval-Augmented Generation
                        </div>
                        <div style="font-size:1.1em; color: #444; margin-top: 0.2em;">
                            <b>Ask questions and get answers grounded in your PDF documents.</b>
                        </div>
                        """,
                        elem_id="header"
                    )

            # Subheader with instructions and sample docs
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        """
                        <div style="font-size:1.05em; color: #333;">
                            <b>How to use:</b> Type your question below and click <b>Ask</b>.<br>
                            <i>Example:</i> <span style="color: #1976d2;"><b>What is agentic AI?</b></span>
                            <br><br>
                            <b>Sample PDF documents available for RAG:</b>
                            <ul style="margin-bottom: 0.5em;">
                                <li>
                                    <a href="https://github.com/manju07/multimodal-rag-system/tree/main/data/pdf_files/agentic_ai_tutorial.pdf" target="_blank">agentic_ai_tutorial.pdf</a>
                                </li>
                                <li>
                                    <a href="https://github.com/manju07/multimodal-rag-system/tree/main/data/pdf_files/huggingface_tutorial.pdf" target="_blank">huggingface_tutorial.pdf</a>
                                </li>
                                <li>
                                    <a href="https://github.com/manju07/multimodal-rag-system/tree/main/data/pdf_files/langgraph_tutorial.pdf" target="_blank">langgraph_tutorial.pdf</a>
                                </li>
                                <li>
                                    <a href="https://github.com/manju07/multimodal-rag-system/tree/main/data/pdf_files/openai_api_tutorial.pdf" target="_blank">openai_api_tutorial.pdf</a>
                                </li>
                                <li>
                                    <a href="https://github.com/manju07/multimodal-rag-system/tree/main/data/pdf_files/langchain_learning.pdf" target="_blank">langchain_learning.pdf</a>
                                </li>
                            </ul>
                            <div style="font-size: 0.97em; color: #555;">
                                <b>Browse or download the documents from the <a href="https://github.com/manju07/multimodal-rag-system/tree/main/data/pdf_files" target="_blank">project's data repository</a>.</b>
                            </div>
                        </div>
                        """,
                        elem_id="subheader"
                    )

            # Main interaction area
            with gr.Row():
                with gr.Column(scale=5, min_width=350):
                    query = gr.Textbox(
                        lines=4,
                        label="Your Question",
                        placeholder="Type your question here...",
                        elem_id="query-box",
                        autofocus=True,
                        show_copy_button=True,
                        container=True,
                        max_lines=8
                    )
                    with gr.Row():
                        submit_btn = gr.Button("🔍 Ask", variant="primary", size="lg", elem_id="ask-btn")
                        clear_btn = gr.Button("🧹 Clear", variant="secondary", size="sm", elem_id="clear-btn")
                    gr.Examples(
                        examples=[
                            "What is agentic AI?",
                            "Summarize the main points from the huggingface tutorial.",
                            "How do I use the OpenAI API?",
                            "What is LangGraph?",
                            "Explain the key concepts in langchain_learning.pdf."
                        ],
                        inputs=query,
                        label="Example Questions"
                    )
                with gr.Column(scale=7, min_width=400):
                    output = gr.Textbox(
                        label="RAG Agent Response",
                        lines=16,
                        interactive=False,
                        elem_id="response-box",
                        show_copy_button=True,
                        container=True,
                        max_lines=32
                    )
                    status = gr.State("")

            # Add a footer
            gr.Markdown(
                """
                <div style="text-align: center; color: #888; font-size: 0.98em; margin-top: 2em;">
                    Built with <a href="https://github.com/manju07/multimodal-rag-system" target="_blank">multimodal-rag-system</a> | Powered by Gradio
                </div>
                """,
                elem_id="footer"
            )

            # Async handler for status updates
            async def _run_and_status(query, status):
                status_msg = "⏳ Generating answer..."
                # Show status message
                yield gr.update(value=""), gr.update(value=status_msg)
                result = await self.run_rag_agent(query)
                # If result is a RunResult object, try to extract its output
                if hasattr(result, "output"):
                    output_text = result.output
                else:
                    output_text = str(result)
                # Clear status message after completion
                yield gr.update(value=output_text), gr.update(value="")

            # Use gradio's event streaming for async status updates
            submit_btn.click(
                fn=_run_and_status,
                inputs=[query, status],
                outputs=[output, status],
                api_name="ask",
                queue=True,
                show_progress="full"
            )
            query.submit(
                fn=_run_and_status,
                inputs=[query, status],
                outputs=[output, status],
                api_name="ask",
                queue=True,
                show_progress="full"
            )
            clear_btn.click(
                fn=lambda: ("", ""),
                inputs=None,
                outputs=[query, output]
            )

        iface.launch(show_api=False, inbrowser=True)

if __name__ == "__main__":
    rag_system = RAGSystem()
    rag_retriever = rag_system.initialize_rag_retriever()
    rag_system.load_rag_agent()
    rag_system.launch_gradio()
