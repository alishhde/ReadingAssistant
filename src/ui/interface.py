import gradio as gr
from typing import Optional, Any
import logging
from src.state.state_manager import ProcessingStatus, StateManager, ApplicationState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradioApp:
    """
    Gradio-based user interface for the Reading Assistant.
    
    This class implements a web-based interface using Gradio, providing:
    - Document upload functionality
    - Question input and answer display
    - Real-time character counting
    - Status updates and error handling
    """
    def __init__(self, 
                 state_manager: StateManager
                 ) -> None:
        """
        Initialize the Gradio interface.
        
        Args:
            state_manager: Instance of StateManager for state handling
        """
        self.demo = gr.Blocks(title="Reading Assistant", theme=gr.themes.Soft())
        
        # Store state manager and question processor
        self.state_manager = state_manager
        
        # Register as observer
        self.state_manager.add_observer(self)

        with self.demo:
            gr.Markdown("# Reading Assistant")
            gr.Markdown("Upload your document and ask questions about it")
            
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("### Document Upload")
                    self.create_file_upload_section()
                    gr.Markdown("### Ask Questions")
                with gr.Column(scale=1):
                    self.create_text_section()

    def launch(self, 
              server_name: str = "127.0.0.1", 
              server_port: int = 7860
              ) -> None:
        """
        Launch the Gradio interface.
        
        Args:
            server_name: Host address for the server
            server_port: Port number for the server
        """
        logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
        self.demo.launch(server_name=server_name, server_port=server_port)

    def update(self, state: ApplicationState) -> str:
        """
        Update UI based on state changes.
        
        Args:
            state: Current application state
            
        Returns:
            Status message to display
        """
        if state.processing_status == ProcessingStatus.PROCESSING:
            return "Processing document..."
        elif state.processing_status == ProcessingStatus.COMPLETED:
            return f"✅ Successfully loaded: {state.current_file.split('/')[-1]}"
        elif state.processing_status == ProcessingStatus.WORKING:
            return f"🟢 Ready for more files. You can ask questions about all uploaded documents. Last: {state.current_file.split('/')[-1]}"
        elif state.processing_status == ProcessingStatus.ERROR:
            return f"❌ Error: {state.error_message}"

        if state.processing_question_status == ProcessingStatus.PROCESSING:
            return "Processing question..."
        elif state.processing_question_status == ProcessingStatus.COMPLETED:
            return f"✅ Successfully answered: {state.last_question}"
        elif state.processing_question_status == ProcessingStatus.ERROR:
            return f"❌ Error: {state.error_message}"

        return ""

    def process_file(self, files: Optional[Any]) -> str:
        """
        Process an uploaded file.
        
        Args:
            files: The uploaded file objects
            
        Returns:
            Status message about the processing
        """
        if not files:
            return "No file uploaded"
        if not isinstance(files, list):
            files = [files]
        try:
            for file in files:
                logger.info(f"Processing file: {file}")
                self.state_manager.update_file(file)
            return self.update(self.state_manager.get_state())
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            self.state_manager.set_error(str(e))
            return self.update(self.state_manager.get_state())

    def create_file_upload_section(self) -> None:
        """Create the file upload section of the interface."""
        file_input = gr.File(
            label="Upload your document",
            file_count="multiple",
            file_types=[".txt", ".pdf", ".doc", ".docx"],
            type="filepath"
        )
        self.file_status = gr.Textbox(
            label="Upload Status",
            interactive=False
        )
        file_input.change(
            fn=self.process_file,
            inputs=file_input,
            outputs=self.file_status
        )

    def process_question(self, question: str) -> str:
        """
        Process a user question.
        
        Args:
            question: The user's question
            
        Returns:
            Answer to the question or status message
        """
        if not question:
            return "Please enter a question"
        if not self.state_manager.get_state().current_file:
            return "Please upload a document first"
        try:
            logger.info(f"Processing question: {question}")
            self.state_manager.update_question(question)
            return self.state_manager.get_state().last_answer or "Processing your question..."
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            self.state_manager.set_error(str(e))
            return f"Error: {str(e)}"

    def count_characters(self, text: Optional[str]) -> str:
        """
        Count characters in the text.
        
        Args:
            text: Text to count characters in
            
        Returns:
            String containing character count
        """
        if not text:
            return "0 characters"
        return f"{len(text)} characters"

    def create_text_section(self) -> None:
        """Create the text input/output section of the interface."""
        # Question input with character count
        text_input = gr.Textbox(
            label="Your Question",
            placeholder="Type your question about the document here...",
            lines=3
        )
        input_char_count = gr.Markdown("0 characters")
        
        # Answer output with character count
        text_output = gr.Textbox(
            label="Answer",
            lines=5,
            interactive=False
        )
        output_char_count = gr.Markdown("0 characters")

        # Submit button
        submit_btn = gr.Button("Submit Question")

        # Update character counts in real-time
        text_input.change(
            fn=self.count_characters,
            inputs=[text_input],
            outputs=[input_char_count],
            show_progress=False
        )
        
        text_output.change(
            fn=self.count_characters,
            inputs=[text_output],
            outputs=[output_char_count],
            show_progress=False
        )

        # Process question and update answer
        submit_btn.click(
            fn=self.process_question,
            inputs=[text_input],
            outputs=[text_output]
        ).then(  # Chain the character count update after the answer is displayed
            fn=self.count_characters,
            inputs=[text_output],
            outputs=[output_char_count]
        )
