"""Chunking wrapper for easy document splitting."""

from typing import List, Optional
from langchain_core.documents import Document
from src.ingestion.chunkers import TextChunker
from src.core.logger import Logger
from src.core.config import Config


class Chunking:
    """Convenience class for document chunking operations."""
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the chunking wrapper.
        
        Args:
            chunk_size: Maximum chunk size
            chunk_overlap: Chunk overlap
            logger: Optional logger instance
        """
        config = Config.load()
        self.logger = logger or Logger()
        self.chunker = TextChunker(
            chunk_size=chunk_size or config.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
            logger=logger
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of documents to chunk
        
        Returns:
            List of chunked documents
        """
        return self.chunker.split_documents(documents)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk a single text.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        return self.chunker.split_text(text)
    
    def chunk_by_structure(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents by structure (paragraphs, sections).
        
        Args:
            documents: List of documents to chunk
        
        Returns:
            List of structure-aware chunked documents
        """
        return self.chunker.split_by_structure(documents)
    
    def get_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about chunks.
        
        Args:
            chunks: List of chunked documents
        
        Returns:
            Statistics dictionary
        """
        return self.chunker.get_chunk_stats(chunks)
