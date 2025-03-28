from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import os
import logging
from typing import Optional, Tuple

import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Engine:
    # Default configurations for different model types
    DEFAULT_CONFIG = {
        "openai": {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 512
        },
        "huggingface": {
            "temperature": 0.7,
            "max_new_tokens": 512,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
    }

    def __init__(self, 
                 local_model_name: Optional[str] = None, 
                 remote_model_name: Optional[str] = None,
                 HF_TOKEN: Optional[str] = None,
                 running_locally: bool = False,
                 openai_api_key: Optional[str] = None,
                 model_type: Optional[str] = None  
                ) -> None:
        """
        Initialize the Engine with specified model configuration.
        
        Args:
            local_model_name: Name of the local HuggingFace model
            remote_model_name: Name of the remote HuggingFace model
            HF_TOKEN: HuggingFace API token
            running_locally: Whether to run models locally
            openai_api_key: OpenAI API key
            model_type: Type of model to use ("openai", "huggingface", or "local")
        """
        if model_type not in ["openai", "huggingface", "local"]:
            raise ValueError(f"\nUnsupported model type: {model_type}")
            
        self.model_type = model_type
        self.local_model_name = local_model_name
        self.remote_model_name = remote_model_name
        self.HF_TOKEN = HF_TOKEN
        self.openai_api_key = openai_api_key
        
        try:
            if model_type == "openai":
                logger.info("\nInitializing OpenAI model...")
                if not self.openai_api_key:
                    raise ValueError("OpenAI API key not provided. Set it via openai_api_key parameter.")
                self.model = self.load_openai_model()
                logger.info("OpenAI model loaded successfully!")
                logger.info("Testing the model...")
                print(self.sample_remote_generation())
                logger.info("Model tested successfully!")
                
            elif model_type == "huggingface":
                logger.info("\nInitializing HuggingFace endpoint model...")
                if not self.HF_TOKEN:
                    raise ValueError("HuggingFace token not provided. Set it via HF_TOKEN environment variable or constructor parameter.")
                login(token=self.HF_TOKEN)
                self.model = self.load_endpoint_model()
                logger.info(f"Model {self.remote_model_name} loaded successfully!")
                
            else:  # local model
                logger.info("\nInitializing local HuggingFace model...")
                if not self.HF_TOKEN:
                    raise ValueError("HuggingFace token not provided. Set it via HF_TOKEN environment variable or constructor parameter.")
                login(token=self.HF_TOKEN)
                self.model, self.tokenizer = self.load_local_model()
                logger.info(f"Model {self.local_model_name} loaded successfully!")
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def load_openai_model(self) -> ChatOpenAI:
        """
        Load and configure OpenAI model.
        
        Returns:
            ChatOpenAI: Configured OpenAI model instance
        """
        try:
            config = self.DEFAULT_CONFIG["openai"]
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

    def load_endpoint_model(self) -> HuggingFaceEndpoint:
        """
        Load and configure HuggingFace endpoint model.
        
        Returns:
            HuggingFaceEndpoint: Configured HuggingFace endpoint model
        """
        try:
            config = self.DEFAULT_CONFIG["huggingface"]
            model = HuggingFaceEndpoint(
                repo_id=self.remote_model_name,
                temperature=config["temperature"],
                huggingfacehub_api_token=self.HF_TOKEN,
                task="text-generation",
                do_sample=True,
                max_new_tokens=config["max_new_tokens"],
                top_p=config["top_p"],
                repetition_penalty=config["repetition_penalty"],
                return_full_text=False
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load HuggingFace endpoint model: {str(e)}")
            raise

    def load_local_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load and configure local HuggingFace model.
        
        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: Model and tokenizer
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.local_model_name, token=self.HF_TOKEN)            
            model = AutoModelForCausalLM.from_pretrained(
                self.local_model_name,
                token=self.HF_TOKEN,
                torch_dtype=torch.float16,
                device_map="auto"
            ).eval()
            
            return model, tokenizer
        except Exception as e:
            logger.error(f"Failed to load local model: {str(e)}")
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

    def sample_local_generation(self, 
                              prompt: str = "What is next to this sentence? ",
                              max_tokens: int = 15,
                              temperature: float = 0.5
                              ) -> str:
        """
        Generate text using local model.
        
        Args:
            prompt: Input prompt for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated text
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_tokens,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            logger.error(f"Failed to generate text with local model: {str(e)}")
            raise

    def __del__(self):
        """Cleanup resources when the object is destroyed."""
        if hasattr(self, 'model') and hasattr(self.model, 'close'):
            self.model.close()
