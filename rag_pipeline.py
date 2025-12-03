"""
RAG Pipeline Module
Handles document loading, processing, chunking, and vector store management.
"""
import os
from typing import List, Optional, Union
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    WebBaseLoader,
    WikipediaLoader,
    ArxivLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document

from config import Config


class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document from file path."""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
        }
        
        if file_extension not in loaders:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        loader = loaders[file_extension](str(file_path))
        return loader.load()
    
    def load_directory(self, directory_path: str, glob_pattern: str = "**/*") -> List[Document]:
        """Load all documents from a directory."""
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=PyPDFLoader if glob_pattern.endswith('.pdf') else TextLoader,
        )
        return loader.load()
    
    def load_web_url(self, url: str) -> List[Document]:
        """Load document from a web URL."""
        loader = WebBaseLoader(url)
        return loader.load()
    
    def load_wikipedia(self, query: str, load_max_docs: int = 1) -> List[Document]:
        """Load documents from Wikipedia."""
        loader = WikipediaLoader(query=query, load_max_docs=load_max_docs)
        return loader.load()
    
    def load_arxiv(self, query: str, load_max_docs: int = 1) -> List[Document]:
        """Load documents from arXiv."""
        loader = ArxivLoader(query=query, load_max_docs=load_max_docs)
        return loader.load()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        return self.text_splitter.split_documents(documents)
    
    def process_documents(
        self, 
        source: Union[str, List[str]], 
        source_type: str = "file",
        add_metadata: bool = True
    ) -> List[Document]:
        """
        Process documents from various sources.
        
        Args:
            source: File path, directory path, URL, or list of sources
            source_type: Type of source - 'file', 'directory', 'url', 'wikipedia', 'arxiv'
            add_metadata: Whether to add upload timestamp and source info to metadata
        """
        import time
        from pathlib import Path
        
        documents = []
        upload_timestamp = time.time()
        
        if source_type == "file":
            if isinstance(source, list):
                for file_path in source:
                    loaded_docs = self.load_document(file_path)
                    # Add metadata to track source file
                    if add_metadata:
                        file_name = Path(file_path).name
                        for doc in loaded_docs:
                            if doc.metadata is None:
                                doc.metadata = {}
                            doc.metadata['source_file'] = file_name
                            doc.metadata['upload_timestamp'] = upload_timestamp
                            doc.metadata['source_path'] = str(file_path)
                    documents.extend(loaded_docs)
            else:
                loaded_docs = self.load_document(source)
                # Add metadata to track source file
                if add_metadata:
                    file_name = Path(source).name
                    for doc in loaded_docs:
                        if doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata['source_file'] = file_name
                        doc.metadata['upload_timestamp'] = upload_timestamp
                        doc.metadata['source_path'] = str(source)
                documents = loaded_docs
        
        elif source_type == "directory":
            documents = self.load_directory(source)
            if add_metadata:
                for doc in documents:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['upload_timestamp'] = upload_timestamp
        
        elif source_type == "url":
            if isinstance(source, list):
                for url in source:
                    loaded_docs = self.load_web_url(url)
                    if add_metadata:
                        for doc in loaded_docs:
                            if doc.metadata is None:
                                doc.metadata = {}
                            doc.metadata['source_url'] = url
                            doc.metadata['upload_timestamp'] = upload_timestamp
                    documents.extend(loaded_docs)
            else:
                documents = self.load_web_url(source)
                if add_metadata:
                    for doc in documents:
                        if doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata['source_url'] = source
                        doc.metadata['upload_timestamp'] = upload_timestamp
        
        elif source_type == "wikipedia":
            documents = self.load_wikipedia(source)
            if add_metadata:
                for doc in documents:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['source_type'] = 'wikipedia'
                    doc.metadata['upload_timestamp'] = upload_timestamp
        
        elif source_type == "arxiv":
            documents = self.load_arxiv(source)
            if add_metadata:
                for doc in documents:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata['source_type'] = 'arxiv'
                    doc.metadata['upload_timestamp'] = upload_timestamp
        
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Split documents into chunks
        # Note: LangChain's split_documents automatically preserves metadata
        split_docs = self.split_documents(documents)
        
        return split_docs


class VectorStoreManager:
    """Manages vector store creation and operations."""
    
    def __init__(self, embedding_model: Optional[str] = None):
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.embeddings = None
        self.vector_store = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model."""
        try:
            if Config.OPENAI_API_KEY:
                self.embeddings = OpenAIEmbeddings(
                    model=self.embedding_model,
                    openai_api_key=Config.OPENAI_API_KEY
                )
            else:
                # Fallback to sentence transformers for local embeddings
                print("Using local sentence transformers for embeddings...")
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                except ImportError:
                    # Try alternative import path
                    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(
                    model_name='sentence-transformers/all-MiniLM-L6-v2'
                )
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            # Fallback to local embeddings
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2'
            )
    
    def create_vector_store(
        self, 
        documents: List[Document], 
        store_type: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """Create a vector store from documents."""
        store_type = store_type or Config.VECTOR_STORE_TYPE
        persist_directory = persist_directory or Config.PERSIST_DIRECTORY
        
        # Ensure persist directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
        if store_type == "chromadb":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
            self.vector_store.persist()
        
        elif store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Save FAISS index
            self.vector_store.save_local(persist_directory)
        
        elif store_type == "objectbox":
            try:
                from langchain_objectbox import ObjectBoxVectorStore
                self.vector_store = ObjectBoxVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            except ImportError:
                print("ObjectBox not available, falling back to ChromaDB")
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=persist_directory
                )
        
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        return self.vector_store
    
    def add_documents_to_store(
        self,
        documents: List[Document],
        store_type: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """Add documents to an existing vector store, or create one if it doesn't exist."""
        store_type = store_type or Config.VECTOR_STORE_TYPE
        persist_directory = persist_directory or Config.PERSIST_DIRECTORY
        
        # If no vector store exists, create one
        if self.vector_store is None:
            return self.create_vector_store(documents, store_type, persist_directory)
        
        # Add documents to existing store
        if store_type == "chromadb":
            # Add documents to existing ChromaDB store
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
        
        elif store_type == "faiss":
            # For FAISS, we need to merge the new documents
            # Create a temporary FAISS store and merge
            temp_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            self.vector_store.merge_from(temp_store)
            self.vector_store.save_local(persist_directory)
        
        else:
            # For other types, try to add documents
            try:
                self.vector_store.add_documents(documents)
            except AttributeError:
                # If add_documents doesn't exist, create a new store with all documents
                print("Warning: Cannot add to existing store, creating new one with all documents")
                # Get existing documents if possible
                try:
                    existing_docs = self.vector_store.similarity_search("", k=10000)
                    all_documents = existing_docs + documents
                except:
                    all_documents = documents
                return self.create_vector_store(all_documents, store_type, persist_directory)
        
        return self.vector_store
    
    def load_vector_store(
        self, 
        store_type: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """Load an existing vector store."""
        store_type = store_type or Config.VECTOR_STORE_TYPE
        persist_directory = persist_directory or Config.PERSIST_DIRECTORY
        
        if not os.path.exists(persist_directory):
            return None
        
        if store_type == "chromadb":
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        
        elif store_type == "faiss":
            self.vector_store = FAISS.load_local(
                persist_directory,
                self.embeddings
            )
        
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        return self.vector_store
    
    def get_retriever(self, k: int = 4, search_type: str = "similarity"):
        """Get a retriever from the vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load a vector store first.")
        
        if search_type == "similarity":
            return self.vector_store.as_retriever(search_kwargs={"k": k})
        elif search_type == "mmr":
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k, "fetch_k": k * 2}
            )
        else:
            raise ValueError(f"Unsupported search type: {search_type}")


class RAGPipeline:
    """Main RAG pipeline that combines document processing and vector store."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.vector_store = None
    
    def ingest_documents(
        self,
        source: Union[str, List[str]],
        source_type: str = "file",
        store_type: Optional[str] = None,
        persist_directory: Optional[str] = None,
        add_to_existing: bool = True
    ):
        """
        Ingest documents into the RAG pipeline.
        
        Args:
            source: Document source(s)
            source_type: Type of source
            store_type: Vector store type
            persist_directory: Directory to persist store
            add_to_existing: If True, add to existing store; if False, create new store
        """
        # Process documents
        documents = self.processor.process_documents(source, source_type)
        print(f"Processed {len(documents)} document chunks")
        
        # Check if we should add to existing store or create new one
        if add_to_existing and self.vector_store_manager.vector_store is not None:
            # Add documents to existing vector store
            self.vector_store_manager.add_documents_to_store(
                documents,
                store_type,
                persist_directory
            )
            # Update our reference to the vector store
            self.vector_store = self.vector_store_manager.vector_store
            print(f"Added {len(documents)} document chunks to existing vector store")
        else:
            # Create new vector store or add to existing one (which will create if needed)
            self.vector_store_manager.add_documents_to_store(
                documents,
                store_type,
                persist_directory
            )
            # Update our reference to the vector store
            self.vector_store = self.vector_store_manager.vector_store
            print(f"Created/updated vector store with {len(documents)} document chunks")
        
        return self.vector_store
    
    def load_existing_store(
        self,
        store_type: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        """Load an existing vector store."""
        self.vector_store = self.vector_store_manager.load_vector_store(
            store_type,
            persist_directory
        )
        return self.vector_store
    
    def get_retriever(self, k: int = 4, search_type: str = "similarity"):
        """Get a retriever for querying."""
        # Ensure we're using the current vector store from the manager
        if self.vector_store_manager.vector_store is not None:
            self.vector_store = self.vector_store_manager.vector_store
        return self.vector_store_manager.get_retriever(k, search_type)
    
    def clear_vector_store(self, persist_directory: Optional[str] = None):
        """Clear the vector store and remove persisted data."""
        persist_directory = persist_directory or Config.PERSIST_DIRECTORY
        
        # Clear in-memory store
        self.vector_store = None
        self.vector_store_manager.vector_store = None
        
        # Remove persisted directory if it exists
        if os.path.exists(persist_directory):
            import shutil
            try:
                shutil.rmtree(persist_directory)
                print(f"Cleared vector store at {persist_directory}")
            except Exception as e:
                print(f"Warning: Could not remove vector store directory: {e}")
        
        return True

