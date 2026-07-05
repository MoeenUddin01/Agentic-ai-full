"""Document loaders for extracting text from various file formats."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain_core.documents import Document
from src.core.logger import Logger


class DocumentLoader:
    """Load documents from various file formats."""
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the document loader.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or Logger()
        self.supported_formats = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".doc": UnstructuredWordDocumentLoader
        }
    
    def load_from_directory(
        self,
        directory_path: str,
        glob_pattern: str = "**/*.pdf",
        show_progress: bool = True
    ) -> List[Document]:
        """
        Load all documents from a directory matching the glob pattern.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file matching
            show_progress: Whether to show progress
        
        Returns:
            List of loaded documents
        """
        self.logger.info(f"Loading documents from {directory_path}")
        
        if not os.path.exists(directory_path):
            self.logger.error(f"Directory not found: {directory_path}")
            return []
        
        # Determine loader class based on file extension
        loader_cls = self._get_loader_class(glob_pattern)
        
        if loader_cls:
            loader = DirectoryLoader(
                path=directory_path,
                glob=glob_pattern,
                loader_cls=loader_cls,
                show_progress=show_progress
            )
        else:
            # Use default loader
            loader = DirectoryLoader(
                path=directory_path,
                glob=glob_pattern,
                show_progress=show_progress
            )
        
        try:
            documents = loader.load()
            self.logger.info(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading documents: {e}")
            return []
    
    def load_from_file(self, file_path: str) -> Optional[Document]:
        """
        Load a single document from a file.
        
        Args:
            file_path: Path to the file
        
        Returns:
            Loaded document or None if failed
        """
        self.logger.info(f"Loading document from {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            return None
        
        file_ext = Path(file_path).suffix.lower()
        loader_cls = self.supported_formats.get(file_ext)
        
        if not loader_cls:
            self.logger.warning(f"Unsupported file format: {file_ext}")
            return None
        
        try:
            loader = loader_cls(file_path)
            documents = loader.load()
            if documents:
                self.logger.info(f"Successfully loaded {file_path}")
                return documents[0]
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
        
        return None
    
    def add_source_metadata(self, documents: List[Document]) -> List[Document]:
        """
        Add source filename to document metadata.
        
        Args:
            documents: List of documents
        
        Returns:
            Documents with added metadata
        """
        for doc in documents:
            source = doc.metadata.get("source", "")
            if source:
                doc.metadata["filename"] = os.path.basename(source)
        
        self.logger.info("Added source metadata to documents")
        return documents
    
    def _get_loader_class(self, glob_pattern: str):
        """
        Get the appropriate loader class based on glob pattern.
        
        Args:
            glob_pattern: Glob pattern for files
        
        Returns:
            Loader class or None
        """
        if ".pdf" in glob_pattern:
            return PyPDFLoader
        elif ".txt" in glob_pattern:
            return TextLoader
        elif ".md" in glob_pattern:
            return UnstructuredMarkdownLoader
        elif ".doc" in glob_pattern:
            return UnstructuredWordDocumentLoader
        return None
