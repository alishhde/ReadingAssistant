from langchain_openai import ChatOpenAI
from smolagents import OpenAIServerModel, LiteLLMModel

from typing import Optional, Tuple
import logging
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Engine:
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 model_type: Optional[str] = None,
                 agent_model_loader: Optional[str] = None,
                 config: Optional[dict] = None
                ) -> None:
        """
        Initialize the Engine with specified model configuration.
        
        Args:
            openai_api_key: OpenAI API key
            model_type: Type of model to use ("openai")
            agent_model_loader: Whether to load the model for Smolagent
            openai_config: Configuration for the OpenAI model
        """
        if model_type not in ["openai", "ollama"]:
            raise ValueError(f"\nUnsupported model type: {model_type}")
            
        self.model_type = model_type
        self.openai_api_key = openai_api_key
        self.config = config
        
        try:
            if model_type == "openai":
                if agent_model_loader:
                    logger.info("\nInitializing OpenAI model for Smolagent...")
                    if not self.openai_api_key:
                        raise ValueError("OpenAI API key not provided. Set it via openai_api_key parameter.")
                    self.model = self.load_openai_model_smolagent()
                    logger.info("OpenAI model for Smolagent loaded successfully!")
                else:
                    logger.info("\nInitializing OpenAI model...")
                    if not self.openai_api_key:
                        raise ValueError("OpenAI API key not provided. Set it via openai_api_key parameter.")
                    self.model = self.load_openai_model()
                    logger.info("OpenAI model loaded successfully!")
                    logger.info("Testing the model...")
                    print(self.sample_remote_generation())
                    logger.info("Model tested successfully!")
            elif model_type == 'ollama':
                if agent_model_loader:
                    logger.info("\nInitializing Ollama model for Smolagent...")
                    self.model = self.load_ollama_model_smolagent()
                    logger.info("Ollama model for Smolagent loaded successfully!")
                else:
                    raise ValueError("Ollama simple model loader is not provided.")
            else:
                raise ValueError(f"\nUnsupported model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise


    def load_ollama_model_smolagent(self) -> LiteLLMModel:
        """
        Load and configure Ollama model for Smolagent.
        
        Returns:
            LiteLLMModel: Configured Ollama model instance for Smolagent input
        """
        try:
            config = self.config['Ollama_CONFIG']
            model = LiteLLMModel(
                model_id=config["model_name"],
                api_base=config["url"]
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load Ollama model for Smolagent: {str(e)}")
            raise


    def load_openai_model_smolagent(self) -> ChatOpenAI:
        """
        Load and configure OpenAI model for Smolagent.
        
        Returns:
            ChatOpenAI: Configured OpenAI model instance for Smolagent input
        """
        try:
            config = self.config['OPENAI_CONFIG']
            model = OpenAIServerModel(
                model_id=config["model_name"],
                api_key=self.openai_api_key
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load OpenAI model for Smolagent: {str(e)}")
            raise


    def load_openai_model(self) -> ChatOpenAI:
        """
        Load and configure OpenAI model.
        
        Returns:
            ChatOpenAI: Configured OpenAI model instance
        """
        try:
            config = self.config['OPENAI_CONFIG']
            model = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model_name=config["model_name"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load OpenAI model: {str(e)}")
            raise


    def sample_remote_generation(self, 
                               prompt: str = "What is next to this sentence? ",
                               max_tokens: int = 15,
                               temperature: float = 0.5
                               ) -> str:
        """
        Generate text using remote model (OpenAI or HuggingFace endpoint).
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
        """
        try:
            if self.model_type == "openai":
                response = self.model.invoke(prompt)
                return response.content
            else:
                response = self.model.invoke(prompt)
                return response
        except Exception as e:
            logger.error(f"Failed to generate text with remote model: {str(e)}")
            raise


    def __del__(self):
        """Cleanup resources when the object is destroyed."""
        if hasattr(self, 'model') and hasattr(self.model, 'close'):
            self.model.close()
