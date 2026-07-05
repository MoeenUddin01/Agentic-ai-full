"""Data loader for loading documents from various sources."""

from typing import List, Optional
from langchain_core.documents import Document
from src.ingestion.loaders import DocumentLoader
from src.ingestion.chunkers import TextChunker
from src.core.logger import Logger
from src.core.config import Config


class DataLoader:
    """Convenience class for loading and processing documents."""
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the data loader.
        
        Args:
            logger: Optional logger instance
        """
        config = Config.load()
        self.logger = logger or Logger()
        self.loader = DocumentLoader(logger=logger)
        self.chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            logger=logger
        )
    
    def load_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        chunk: bool = True
    ) -> List[Document]:
        """
        Load documents from a directory and optionally chunk them.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file matching
            chunk: Whether to chunk the documents
        
        Returns:
            List of documents (chunked if chunk=True)
        """
        documents = self.loader.load_from_directory(directory_path, glob_pattern)
        documents = self.loader.add_source_metadata(documents)
        
        if chunk:
            documents = self.chunker.split_documents(documents)
        
        return documents
    
    def load_file(self, file_path: str, chunk: bool = True) -> Optional[List[Document]]:
        """
        Load a single file and optionally chunk it.
        
        Args:
            file_path: Path to the file
            chunk: Whether to chunk the document
        
        Returns:
            List of documents (chunked if chunk=True) or None
        """
        document = self.loader.load_from_file(file_path)
        if not document:
            return None
        
        if chunk:
            return self.chunker.split_documents([document])
        
        return [document]
