from typing import Optional, Any, List, Protocol
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Enumeration of possible processing states."""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    WORKING = "working"
    ERROR = "error"


class Observer(Protocol):
    """Protocol defining the interface for state observers."""
    def update(self, state: 'ApplicationState') -> None:
        """Update the observer with new state."""
        ...


@dataclass
class ApplicationState:
    """
    Data class representing the application's state.
    
    Attributes:
        current_files: List of paths to the currently processed files
        file_contents: List of contents of the processed files
        processing_status: Current status of file processing
        processing_question_status: Current status of question processing
        error_message: Any error message from processing
        last_question: The most recent question asked
        last_answer: The most recent answer provided
    """
    current_files: list = field(default_factory=list)
    file_contents: list = field(default_factory=list)

    processing_status: ProcessingStatus = ProcessingStatus.IDLE
    processing_question_status: ProcessingStatus = ProcessingStatus.IDLE
    error_message: Optional[str] = None

    last_question: Optional[str] = None
    last_answer: Optional[str] = None

    @property
    def current_file(self):
        return self.current_files[-1] if self.current_files else None

    @property
    def file_content(self):
        return self.file_contents[-1] if self.file_contents else None


class StateManager:
    """
    Singleton class managing application state and observer notifications.
    
    This class implements the Observer pattern to notify components of state changes.
    It maintains a single instance of ApplicationState and manages its updates.
    """
    _instance = None
    
    def __new__(cls) -> 'StateManager':
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance.state = ApplicationState()
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the state manager."""
        self.state = ApplicationState()
        self._observers: List[Observer] = []
    
    # ----------------------- Observer Section -----------------------
    def add_observer(self, observer: Observer) -> None:
        """
        Add an observer to be notified of state changes.
        
        Args:
            observer: Object implementing the Observer protocol
        """
        if observer not in self._observers:
            self._observers.append(observer)
            logger.debug(f"Added observer: {observer.__class__.__name__}")
    
    def remove_observer(self, observer: Observer) -> None:
        """
        Remove an observer from notifications.
        
        Args:
            observer: Object implementing the Observer protocol
        """
        if observer in self._observers:
            self._observers.remove(observer)
            logger.debug(f"Removed observer: {observer.__class__.__name__}")
    
    def notify_observers(self) -> None:
        """Notify all observers of state changes."""
        logger.debug(f"Notifying {len(self._observers)} observers of state change")
        for observer in self._observers:
            observer.update(self.state)
    
    # ----------------------- File State Section -----------------------
    def update_file(self, file_path: str) -> None:
        """
        Update the current file in state.
        
        Args:
            file_path: Path to the file being processed
        """
        logger.info(f"Updating current file to: {file_path}")
        self.state.current_files.append(file_path)
        if len(self.state.current_files) == 1:
            self.state.processing_status = ProcessingStatus.PROCESSING
        else:
            self.state.processing_status = ProcessingStatus.WORKING
        self.notify_observers()
    
    def update_file_content(self, content: Any) -> None:
        """
        Update the file content in state.
        
        Args:
            content: Content of the processed file
        """
        logger.info("Updating file content")
        self.state.file_contents.append(content)
        self.state.processing_status = ProcessingStatus.COMPLETED
        self.notify_observers()
    
    # ----------------------- Question Section -----------------------
    def update_question(self, question: str) -> None:
        """
        Update the question in state.
        
        Args:
            question: The question being processed
        """
        logger.info(f"Updating question: {question}")
        self.state.last_question = question
        self.state.processing_question_status = ProcessingStatus.PROCESSING
        self.notify_observers()

    def update_answer(self, answer: str) -> None:
        """
        Update the answer in state.
        
        Args:
            answer: The answer to the question
        """
        logger.info("Updating answer")
        self.state.last_answer = answer
        self.state.processing_question_status = ProcessingStatus.COMPLETED
        self.notify_observers()

    # ----------------------- Current States Section -----------------------
    def get_state(self) -> ApplicationState:
        """
        Get current state.
        
        Returns:
            Current application state
        """
        return self.state 

    def set_error(self, error_message: str) -> None:
        """
        Set error state.
        
        Args:
            error_message: Description of the error
        """
        logger.error(f"Setting error state: {error_message}")
        self.state.error_message = error_message
        self.state.processing_status = ProcessingStatus.ERROR
        self.notify_observers()