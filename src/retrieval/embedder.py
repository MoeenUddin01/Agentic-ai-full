"""Text to vector conversion for embeddings."""

import re
import numpy as np
from typing import List, Optional
from langchain_community.embeddings import OpenAIEmbeddings
from src.core.logger import Logger
from src.core.config import Config


class Embedder:
    """Generate embeddings for text using various models."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_fallback: bool = False,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the embedding model
            use_fallback: Whether to use fallback simple embeddings
            logger: Optional logger instance
        """
        config = Config.load()
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_fallback = use_fallback
        self.logger = logger or Logger()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            self.model = OpenAIEmbeddings(model=self.model_name)
            self.logger.info(f"Loaded OpenAI embeddings: {self.model_name}")
        except Exception as e:
            self.logger.warning(f"Failed to load OpenAI embeddings: {e}")
            self.logger.info("Using fallback simple embeddings")
            self.use_fallback = True
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert list of texts into embeddings.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        """
        if self.use_fallback or self.model is None:
            return self._simple_embed(texts)
        
        try:
            embeddings = self.model.embed_documents(texts)
            self.logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            return self._simple_embed(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Convert a single query text into embedding.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector
        """
        if self.use_fallback or self.model is None:
            return self._simple_embed([query])[0]
        
        try:
            embedding = self.model.embed_query(query)
            self.logger.info("Generated embedding for query")
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            return self._simple_embed([query])[0]
    
    def _simple_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate simple local embeddings when OpenAI is unavailable.
        
        Args:
            texts: List of text strings
        
        Returns:
            List of embedding vectors
        """
        self.logger.info("Using fallback simple embeddings")
        
        token_counts = []
        vocabulary = set()
        
        for text in texts:
            tokens = re.findall(r"\w+", text.lower())
            counts = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            token_counts.append(counts)
            vocabulary.update(counts.keys())
        
        vocabulary = sorted(vocabulary)[:256]
        embedded_texts = []
        
        for counts in token_counts:
            vector = [float(counts.get(token, 0)) for token in vocabulary]
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = [value / norm for value in vector]
            embedded_texts.append(vector)
        
        return embedded_texts
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        if self.use_fallback:
            return 256  # Fallback dimension
        
        config = Config.load()
        return config.EMBEDDING_DIMENSION
