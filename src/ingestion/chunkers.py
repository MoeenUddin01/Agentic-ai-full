"""Text chunkers for structure-aware document splitting."""

from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.core.logger import Logger


class TextChunker:
    """Split documents into smaller chunks for embedding and retrieval."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: callable = len,
        separators: Optional[List[str]] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            length_function: Function to measure chunk length
            separators: List of separators for splitting
            logger: Optional logger instance
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger or Logger()
        
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=separators
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
        
        Returns:
            List of chunked documents
        """
        self.logger.info(f"Splitting {len(documents)} documents into chunks")
        
        try:
            split_docs = self.text_splitter.split_documents(documents)
            self.logger.info(f"Created {len(split_docs)} chunks from {len(documents)} documents")
            return split_docs
        except Exception as e:
            self.logger.error(f"Error splitting documents: {e}")
            return []
    
    def split_text(self, text: str) -> List[str]:
        """
        Split a single text into chunks.
        
        Args:
            text: Text to split
        
        Returns:
            List of text chunks
        """
        self.logger.info("Splitting text into chunks")
        
        try:
            chunks = self.text_splitter.split_text(text)
            self.logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Error splitting text: {e}")
            return []
    
    def split_by_structure(self, documents: List[Document]) -> List[Document]:
        """
        Split documents by structure (paragraphs, sections).
        
        Args:
            documents: List of documents to split
        
        Returns:
            List of structure-aware chunked documents
        """
        self.logger.info("Performing structure-aware splitting")
        
        chunks = []
        for doc in documents:
            # Split by paragraphs first
            paragraphs = doc.page_content.split("\n\n")
            
            for para in paragraphs:
                if len(para.strip()) > 50:  # Filter out very short paragraphs
                    chunk = Document(
                        page_content=para.strip(),
                        metadata=doc.metadata.copy()
                    )
                    chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} structure-aware chunks")
        return chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> dict:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunked documents
        
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "min_length": min(chunk_lengths),
            "max_length": max(chunk_lengths),
            "avg_length": sum(chunk_lengths) / len(chunk_lengths),
            "total_characters": sum(chunk_lengths)
        }
