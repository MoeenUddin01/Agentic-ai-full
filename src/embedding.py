"""Embedding wrapper for easy text to vector conversion."""

from typing import List, Optional
from src.retrieval.embedder import Embedder
from src.core.logger import Logger
from src.core.config import Config


class Embedding:
    """Convenience class for text embedding operations."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fallback: bool = False,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the embedding wrapper.
        
        Args:
            model_name: Name of the embedding model
            use_fallback: Whether to use fallback simple embeddings
            logger: Optional logger instance
        """
        config = Config.load()
        self.logger = logger or Logger()
        self.embedder = Embedder(
            model_name=model_name or config.EMBEDDING_MODEL,
            use_fallback=use_fallback,
            logger=logger
        )
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        """
        return self.embedder.embed_texts(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector
        """
        return self.embedder.embed_query(query)
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.embedder.get_embedding_dimension()
