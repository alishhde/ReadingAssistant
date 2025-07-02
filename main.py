from dotenv import load_dotenv
import os

from src.ui.interface import GradioApp
from src.core.connector import Connector 
from src.state.state_manager import StateManager

# Load environment variables from .env file
load_dotenv()


def main():
    # Load environment variables
    env_vars = {
        'HF_TOKEN': os.getenv('HF_TOKEN'),
        'LOCAL_MODEL_NAME': os.getenv('LOCAL_MODEL_NAME'),
        'REMOTE_MODEL_NAME': os.getenv('REMOTE_MODEL_NAME'),
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'MODEL_TYPE': os.getenv('MODEL_TYPE'),
        'SERVER_NAME': os.getenv('SERVER_NAME'),
        'SERVER_PORT': int(os.getenv('SERVER_PORT')),
        'AGENT_MODEL_LOADER': bool(os.getenv('AGENT_MODEL_LOADER').lower() == 'true')
    }
    
    # Create the State Manager instance
    state_manager = StateManager()
    
    # Initialize the Connector instance with environment variables and state manager
    backend = Connector(env_vars, state_manager)
    
    # Create and launch the Gradio interface with the same state manager
    app = GradioApp(state_manager)
    app.launch(server_name=env_vars['SERVER_NAME'], server_port=env_vars['SERVER_PORT'])

if __name__ == "__main__":
    main()
