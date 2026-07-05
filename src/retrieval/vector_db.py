"""Vector database clients for storing and retrieving embeddings."""

import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.core.logger import Logger
from src.core.config import Config


class VectorDatabase:
    """Handle storage and retrieval of embeddings using ChromaDB."""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            logger: Optional logger instance
        """
        config = Config.load()
        self.collection_name = collection_name or config.VECTOR_DB_COLLECTION
        self.persist_directory = persist_directory or config.VECTOR_DB_PATH
        self.logger = logger or Logger()
        
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            self.logger.info(f"Vector store initialized: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None
    ):
        """
        Add documents with embeddings to vector database.
        
        Args:
            documents: List of document dictionaries with content and metadata
            embeddings: Optional pre-computed embeddings
        """
        if not documents:
            self.logger.warning("No documents to add")
            return
        
        if len(documents) != len(embeddings) if embeddings else False:
            raise ValueError("Documents and embeddings must have same length")
        
        self.logger.info(f"Adding {len(documents)} documents to vector store")
        
        ids = []
        metadatas = []
        contents = []
        embeddings_list = []
        
        for i, doc in enumerate(documents):
            # Unique ID
            doc_id = doc.get("id", f"doc_{uuid.uuid4().hex[:8]}_{i}")
            ids.append(doc_id)
            
            # Metadata
            metadata = dict(doc.get("metadata", {}))
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.get("content", ""))
            metadatas.append(metadata)
            
            # Content
            contents.append(doc.get("content", ""))
            
            # Embedding
            if embeddings and i < len(embeddings):
                emb = embeddings[i]
                if hasattr(emb, "tolist"):
                    embeddings_list.append(emb.tolist())
                else:
                    embeddings_list.append(list(emb))
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list if embeddings_list else None,
                metadatas=metadatas,
                documents=contents
            )
            self.logger.info(f"Successfully added {len(documents)} documents")
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        include: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector database for similar documents.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            include: What to include in results (metadatas, documents, distances)
        
        Returns:
            Query results from ChromaDB
        """
        if include is None:
            include = ["metadatas", "documents", "distances"]
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=include
            )
            self.logger.info(f"Retrieved {n_results} results")
            return results
        except Exception as e:
            self.logger.error(f"Error querying vector store: {e}")
            return {}
    
    def delete(self, ids: List[str]):
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"Deleted {len(ids)} documents")
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            self.logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            raise
