from src.ui.interface import GradioApp
from src.core.connector import Connector 
from src.state.state_manager import StateManager

from dotenv import load_dotenv
import ast
import os
import yaml


load_dotenv(override=True)


def main():
    # 1. Load environment variables
    variables = {
        'EMBEDDING_MODEL': os.getenv('EMBEDDING_MODEL'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'CONFIG_PATH': os.getenv('CONFIG_PATH'),
        'MODEL_TYPE': os.getenv('MODEL_TYPE'),
        'SERVER_NAME': os.getenv('SERVER_NAME'),
        'SERVER_PORT': int(os.getenv('SERVER_PORT')),
        'AGENT_MODEL_LOADER': bool(os.getenv('AGENT_MODEL_LOADER').lower() == 'true'),
        'CHROMA_DATABASE_PATH': os.getenv('CHROMA_PATH')
    }

    # 2. Load the Configuration files
    with open(variables['CONFIG_PATH'], 'r') as file:
        config = yaml.safe_load(file)
    
    # 3. Updating the environment variables with the configuration file
    variables['CONFIG'] = config
    
    # 4. Create the State Manager instance
    state_manager = StateManager()
    
    # 5. Initialize the Connector instance with environment variables and state manager
    backend = Connector(variables, state_manager)
    
    # 6. Create and launch the Gradio interface with the same state manager
    app = GradioApp(state_manager)
    app.launch(server_name=variables['SERVER_NAME'], server_port=variables['SERVER_PORT'])


if __name__ == "__main__":
    main()
