from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class PDFDocumentsLoader:
    """Class for loading and splitting PDF documents."""

    def __init__(self, directory="./data/pdf_files/", glob_pattern="*.pdf", show_progress=False):
        self.directory = directory
        self.glob_pattern = glob_pattern
        self.show_progress = show_progress

    def load_pdf_documents(self):
        """Load PDF documents from a directory using PyMuPDFLoader"""
        loader = DirectoryLoader(
            self.directory,
            loader_cls=PyMuPDFLoader,
            show_progress=self.show_progress,
            glob=self.glob_pattern
        )
        documents = loader.load()
        print(documents)
        return documents

    @staticmethod
    def split_documents(documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into smaller chunks for better RAG performance"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        
        # Show example of a chunk
        if split_docs:
            print(f"\nExample chunk:")
            print(f"Content: {split_docs[0].page_content[:200]}...")
            print(f"Metadata: {split_docs[0].metadata}")
        
        return split_docs