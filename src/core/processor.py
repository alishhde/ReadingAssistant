from typing import List, Any, Optional
import logging
from langchain_community.document_loaders import PyPDFLoader, ArxivLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Processor:
    def __init__(self) -> None:
        """
        Initialize the Processor class.
        """
        self.current_document: Optional[Any] = None
        self.document_content: Optional[Any] = None

    def loader(self, 
              input_source: Any,
              doc_type: str
              ) -> List[Any]:
        """
        Process the input source and store its content.
        
        Args:
            input_source: Either a file object (for PDF) or arxiv_id (for arXiv)
            doc_type: Type of document ("pdf" or "arxiv")
            
        Returns:
            List of document pages or error message string
            
        Raises:
            Exception: If document processing fails
        """
        try:
            if doc_type.lower() == "pdf":
                logger.info(f"Loading PDF from: {input_source.name}")
                return self.pdf_loader(input_source)
            elif doc_type.lower() == "arxiv":
                return self.arxiv_loader(input_source)
            else:
                logger.warning(f"Unsupported document type: {doc_type}")
                return "Unsupported document type"
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return f"Error processing document: {str(e)}"

    def pdf_loader(self, 
                  file_obj: Any
                  ) -> List[Any]:
        """
        Load a PDF file using the langchain PyPDFLoader.
        
        Args:
            file_obj: File object containing the PDF
            
        Returns:
            List of document pages
            
        Raises:
            Exception: If PDF loading fails
        """
        logger.info(f"Initializing PDF loader for: {file_obj.name}")
        document_loader_instance = PyPDFLoader(file_obj.name)
        loaded_document = document_loader_instance.load()
        logger.info(f"Successfully loaded {len(loaded_document)} pages")
        return loaded_document

    def arxiv_loader(self, 
                    arxiv_id: str
                    ) -> List[Any]:
        """
        Load an arXiv paper using the langchain ArxivLoader.
        
        Args:
            arxiv_id: arXiv paper ID
            
        Returns:
            List of document pages
            
        Raises:
            Exception: If arXiv loading fails
        """
        logger.info(f"Loading arXiv paper: {arxiv_id}")
        document_loader_instance = ArxivLoader(query=arxiv_id, load_max_docs=1)
        loaded_document = document_loader_instance.load()
        logger.info(f"Successfully loaded arXiv paper")
        return loaded_document
