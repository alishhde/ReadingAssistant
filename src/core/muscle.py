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


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Muscles():
    def __init__(self, 
                 engine: Engine,
                 processor: Processor,
                 embedding_model: str
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


    def doc_loader(self, 
                  file: Any,
                  type: str
                  ) -> List[Any]:
        """
        Load a document and return it.
        
        Args:
            file: The document file to load
            type: Type of the document (e.g., 'pdf')
            
        Returns:
            The loaded document content
        """
        return self.processor.loader(file, type)


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
    

    def doc_vector_store(self, 
                        chunks: List[Any],
                        embedding_type: str = "huggingface",
                        vector_database_type: str = "chroma",
                        ) -> Optional[Chroma]:
        """
        Create a vector database from the chunks and return the vector database.
        
        Args:
            chunks: List of document chunks
            embedding_type: Type of embedding model to use
            vector_database_type: Type of vector database to use
            
        Returns:
            Vector database instance or None if type is invalid
        """
        # Step 1: Defining the embedding model
        if embedding_type == "huggingface": 
            logger.info("Initializing embedding model...")
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        else:
            logger.error("Invalid embedding type!")
            return None
        
        # Step 2: Creating the vector database
        if vector_database_type == "chroma":
            # Create a new vector store with the unique collection name
            logger.info("Creating vector database...")
            vector_store = Chroma.from_documents(chunks, embeddings)
            logger.info("Vector database created successfully!")
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
