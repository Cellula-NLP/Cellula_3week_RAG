from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List
from langchain_core.documents import Document


class DataProcessor:
    """Handles document loading, splitting, and vector store creation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def load_and_split(self, directory_path: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[Document]:
        """Loads all .txt, .pdf, and .md files from a directory and splits them into chunks."""

        documents = []

        # load .txt Files
        txt_loader = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        documents.extend(txt_loader.load())

        # Load .pdf Files
        pdf_loader = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents.extend(pdf_loader.load())

        # Load .md Files
        md_loader = DirectoryLoader(
            directory_path,
            glob="**/*.md",
            loader_cls=TextLoader,
            show_progress=True
        )
        documents.extend(md_loader.load())

        # Split all loaded documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def create_vector_store(self, chunks: List[Document], persist_directory: str = "./chroma_db"):
        """Embeds chunks and stores them in a Chroma vector database."""
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        return vectorstore