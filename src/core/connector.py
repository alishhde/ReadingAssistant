from src.state.state_manager import ProcessingStatus, StateManager
from src.core.muscle import Muscles
from src.core.brain import Engine
from src.core.processor import Processor

from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Connector:
    def __init__(self, 
                 variables: Dict[str, Any],
                 state_manager: StateManager
                 ) -> None:
        """
        Initialize the Connector class with required components.
        
        Args:
            variables: Dictionary containing environment variables
            state_manager: Instance of StateManager for state handling
        """
        self.state_manager = state_manager
        self.engine = Engine(
            openai_api_key=variables['OPENAI_API_KEY'],
            model_type=variables['MODEL_TYPE'],
            agent_model_loader=variables['AGENT_MODEL_LOADER'],
            config=variables['CONFIG']
        )
        self.processor = Processor()
        self.muscles = Muscles(
            engine=self.engine,
            processor=self.processor,
            embedding_model=variables['EMBEDDING_MODEL'],
            persist_directory=variables['CHROMA_DATABASE_PATH']
        )
        
        # Register as observer
        self.state_manager.add_observer(self)

        self.retriever_instance: Optional[Any] = None
        self._last_processed_question: Optional[str] = None
    

    def update(self, state: Any) -> None:
        """
        Handle state changes in backend.
        
        Args:
            state: Current state object containing processing status and question information
        """
        logger.info("=== Backend Handler Update ===")
        logger.info(f"Current State - Processing: {state.processing_status}, Question Status: {state.processing_question_status}")
        logger.info(f"Current Question: {state.last_question}")
        logger.info(f"Last Processed Question: {self._last_processed_question}")
        
        # Handle file processing state
        if state.processing_status == ProcessingStatus.PROCESSING:
            try:
                logger.info("=== Starting File Processing ===")
                # Step 1: Process the file and get its content
                self.file_processor()

                # Step 2: Split the file content into chunks
                chunks = self.muscles.doc_splitter(self.state_manager.get_state().file_content)
                logger.info(f"Created {len(chunks)} chunks")

                # Step 3: Defining the vector database
                file_path = self.state_manager.get_state().current_file
                vectordb = self.muscles.doc_vector_store(chunks, file_path=file_path)
                logger.info("Vector database created successfully!")

                # Step 4: Defining a retriever instance
                self.retriever_instance = self.muscles.doc_retriever_instance(vector_store=vectordb)
                logger.info("Retriever instance created successfully!")
            except Exception as e:
                logger.error(f"Error during file processing: {str(e)}")
                self.state_manager.set_error(str(e))
        
        elif state.processing_status == ProcessingStatus.COMPLETED:
            if self.state_manager.get_state().file_content:
                logger.info("\nThe following is the first 500 characters of the document:")
                logger.info(self.state_manager.get_state().file_content[0].page_content[:500])
            else:
                logger.warning("No documents content found!")
        elif state.processing_status == ProcessingStatus.ERROR:
            logger.error("An error occurred while processing the file!")

        # Handle question processing state
        if state.processing_question_status == ProcessingStatus.PROCESSING:
            logger.info("=== Question Processing State ===")
            logger.info(f"Current question: {state.last_question}")
            logger.info(f"Last processed question: {self._last_processed_question}")
            
            # Only process if this is a new question
            if state.last_question != self._last_processed_question:
                try:
                    logger.info("Processing new question...")
                    self._last_processed_question = state.last_question
                    self.question_processor(state)
                except Exception as e:
                    logger.error(f"Error processing question: {str(e)}")
                    self.state_manager.set_error(str(e))
            else:
                logger.info("Question already processed, skipping...")

        elif state.processing_question_status == ProcessingStatus.ERROR:
            logger.error("An error occurred while processing the question!")


    def file_processor(self) -> None:
        """
        Process the file and update state with file content.
        
        Raises:
            Exception: If file processing fails
        """
        try:
            doc_type = self.state_manager.get_state().current_file.split('.')[-1]

            # Loading the file
            processed_content = self.muscles.doc_loader(
                self.state_manager.get_state().current_file, 
                type=doc_type
            )

            self.state_manager.update_file_content(processed_content)
        except Exception as e:
            logger.error(f"Error in file processing: {str(e)}")
            self.state_manager.set_error(str(e))
            

    def question_processor(self, state: Any) -> None:
        """
        Process the question and generate answer.
        
        Args:
            state: Current state object containing the question
            
        Raises:
            ValueError: If no retriever instance is available
            Exception: If question processing fails
        """
        try:
            if not self.retriever_instance:
                raise ValueError("No retriever instance available. Please process a document first.")

            logger.info("Processing question: %s", state.last_question)
            
            # Step 5: Answer the question
            answer = self.muscles.doc_answer_retriever(
                retriever_instance=self.retriever_instance,
                query=state.last_question
            )
            self.state_manager.update_answer(answer)
        except Exception as e:
            logger.error(f"Error in question processing: {str(e)}")
            self.state_manager.set_error(str(e))
