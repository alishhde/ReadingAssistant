from src.core.brain import Engine
from src.core.processor import Processor
from src.core.agent_tools import PDFRetrieverTool
from smolagents import CodeAgent, DuckDuckGoSearchTool

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

import logging
from typing import List, Optional, Any
import os
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Muscles():
    def __init__(self, 
                 engine: Engine,
                 processor: Processor,
                 embedding_model: str,
                 persist_directory: str
                 ) -> None:
        """
        Initialize the Muscles class with required components.
        
        Args:
            engine: The language model engine instance
            processor: The document processor instance
            embedding_model: Name of the embedding model to use
        """
        self.engine = engine
        self.processor = processor
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory

    def doc_loader(self, 
                  file: Any,
                  type: str
                  ) -> List[Any]:
        """
        Load a document and return it.
        
        Args:
            file: The document file to load (can be single file or list of files)
            type: Type of the document (e.g., 'pdf')
            
        Returns:
            The loaded document content
        """
        # Handle multiple files
        if isinstance(file, list):
            all_documents = []
            for single_file in file:
                documents = self.processor.loader(single_file, type)
                all_documents.extend(documents)
            return all_documents
        else:
            # Handle single file
            return list(self.processor.loader(file, type))


    def doc_splitter(self, 
                    document: List[Any],
                    method: str = "recursive",
                    chunk_size: int = 1000,  # Increased for better context
                    chunk_overlap: int = 200  # Increased overlap to maintain context
                    ) -> Optional[List[Any]]:
        """
        Split the document into chunks and return a list of chunks.
        
        Args:
            document: The document to split
            method: Splitting method (currently only supports "recursive")
            chunk_size: Size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of document chunks or None if method is invalid
        """
        if method == "recursive":
            logger.info("Initializing recursive text splitter...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""],  # Re-enabled separators
                length_function=len
            )
        else:
            logger.error("Invalid method!")
            return None
        
        logger.info("Finished. Splitting document into chunks...")
        chunks = splitter.split_documents(document)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    

    def _get_file_hash(self, file_path: str) -> str:
        """Compute a hash for the file to uniquely identify its content."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()


    def _load_vectorized_files(self) -> set:
        """Load the set of already vectorized file hashes."""
        meta_path = os.path.join(self.persist_directory, "vectorized_files.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                return set(json.load(f))
        return set()


    def _save_vectorized_files(self, file_hashes: set) -> None:
        """Save the set of vectorized file hashes."""
        meta_path = os.path.join(self.persist_directory, "vectorized_files.json")
        with open(meta_path, "w") as f:
            json.dump(list(file_hashes), f)
        

    def save_vector_store(self, vector_store: Chroma) -> None:
        """
        Save the vector store to disk for persistence.
        
        Args:
            vector_store: The Chroma vector store instance to save.
        """
        logger.info(f"Persisting vector store to {self.persist_directory} ...")
        vector_store.persist()
        logger.info("Vector store persisted successfully!")
        

    def doc_vector_store(self, 
                        chunks: List[Any],
                        embedding_type: str = "huggingface",
                        vector_database_type: str = "chroma",
                        file_path: str = None
                        ) -> Optional[Chroma]:
        """
        Create a vector database from the chunks and return the vector database.
        
        Args:
            chunks: List of document chunks
            embedding_type: Type of embedding model to use
            vector_database_type: Type of vector database to use
            file_path: Path to the file to vectorize
        Returns:
            Vector database instance or None if type is invalid
        """
        # Step 1: Check if file is already vectorized
        file_hash = self._get_file_hash(file_path)
        vectorized_files = self._load_vectorized_files()
        if file_hash in vectorized_files:
            logger.info(f"File '{file_path}' is already vectorized. Skipping vectorization.\n\n\n")
            # Load the existing vector store
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=HuggingFaceEmbeddings(model_name=self.embedding_model)
            )
            return vector_store

        # Step 2: Defining the embedding model
        if embedding_type == "huggingface": 
            logger.info("Initializing embedding model...")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        else:
            logger.error("Invalid embedding type!")
            return None
            
        # Step 3: Creating the vector database
        if vector_database_type == "chroma":
            logger.info("Creating vector database...")
            # Load or create the vector store
            if os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
                vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=embeddings
                )
                vector_store.add_documents(chunks)  # Add new chunks to existing store
            else:
                vector_store = Chroma.from_documents(
                    chunks, 
                    embeddings,
                    persist_directory=self.persist_directory
                    )
            logger.info("Vector database created successfully!")
            self.save_vector_store(vector_store)
            vectorized_files.add(file_hash)
            self._save_vectorized_files(vectorized_files)
            return vector_store
        else:
            logger.error("Invalid vector database type!")
            return None


    def doc_retriever_instance(self, 
                             vector_store: Chroma,
                             ) -> Any:
        """
        Create a retriever instance from the vector database.
        
        Args:
            vector_store: The vector store to create the retriever from
            
        Returns:
            Retriever instance
        """
        logger.info("Creating retriever instance...")
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        return retriever


    def doc_answer_retriever(self, 
                           retriever_instance: Any,
                           query: str,
                           ) -> str:
        """
        Answer a question using the retriever.
        
        Args:
            retriever_instance: The retriever instance to use
            query: The question to answer
            
        Returns:
            The model's answer
        """
        logger.info("=== Starting Question Processing ===")
        logger.info(f"Input Question: {query}")

        # Create the tools' instances for agent
        pdf_tool = PDFRetrieverTool(retriever_instance)
        web_search_tool = DuckDuckGoSearchTool()

        # Create the agent
        agent = CodeAgent(tools=[pdf_tool, web_search_tool], model=self.engine.model, max_steps=3)

        logger.info("Agent Created Successfully!")

        response = agent.run(query)
        return response
