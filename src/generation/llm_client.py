"""LLM client for calls to hosted/local LLM endpoints."""

from typing import Optional, Dict, Any, List
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.core.logger import Logger
from src.core.config import Config
from src.core.security import SecurityManager


class LLMClient:
    """Client for making calls to various LLM providers."""
    
    def __init__(
        self,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            model_provider: Provider name (groq, openai, google)
            model_name: Model name
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            logger: Optional logger instance
        """
        config = Config.load()
        security = SecurityManager()
        
        self.model_provider = model_provider or config.DEFAULT_MODEL_PROVIDER
        self.model_name = model_name or config.DEFAULT_MODEL
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.logger = logger or Logger()
        
        # Validate API key
        if not security.validate_api_key(self.model_provider):
            self.logger.error(f"API key not found for provider: {self.model_provider}")
            raise ValueError(f"API key required for {self.model_provider}")
        
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM model."""
        try:
            self.model = init_chat_model(
                self.model_name,
                model_provider=self.model_provider
            )
            self.logger.info(f"Initialized {self.model_provider} model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error initializing model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response text
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.model.invoke(messages)
            self.logger.info("Generated response successfully")
            return response.content
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            raise
    
    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Generate response from a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters
        
        Returns:
            Generated response text
        """
        langchain_messages = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                langchain_messages.append(HumanMessage(content=content))
        
        try:
            response = self.model.invoke(langchain_messages)
            self.logger.info("Generated response from messages successfully")
            return response.content
        except Exception as e:
            self.logger.error(f"Error generating response from messages: {e}")
            raise
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ):
        """
        Stream response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
        
        Yields:
            Chunks of the response
        """
        messages = []
        
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        
        messages.append(HumanMessage(content=prompt))
        
        try:
            for chunk in self.model.stream(messages):
                yield chunk.content
        except Exception as e:
            self.logger.error(f"Error streaming response: {e}")
            raise
    
    def set_parameters(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update generation parameters.
        
        Args:
            temperature: New temperature
            max_tokens: New max tokens
        """
        if temperature is not None:
            self.temperature = temperature
            self.logger.info(f"Updated temperature to {temperature}")
        
        if max_tokens is not None:
            self.max_tokens = max_tokens
            self.logger.info(f"Updated max_tokens to {max_tokens}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.model_provider,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
