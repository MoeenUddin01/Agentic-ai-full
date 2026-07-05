"""Vector store wrapper for easy vector database operations."""

from typing import List, Dict, Any, Optional
from src.retrieval.vector_db import VectorDatabase
from src.retrieval.embedder import Embedder
from src.core.logger import Logger
from src.core.config import Config


class VectorStore:
    """Convenience class for vector database operations."""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            logger: Optional logger instance
        """
        self.logger = logger or Logger()
        self.vector_db = VectorDatabase(collection_name, persist_directory, logger)
        self.embedder = Embedder(logger=logger)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents with automatic embedding generation.
        
        Args:
            documents: List of document dictionaries with content
        
        Returns:
            Number of documents added
        """
        # Extract texts for embedding
        texts = [doc.get("content", "") for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedder.embed_texts(texts)
        
        # Add to vector database
        self.vector_db.add_documents(documents, embeddings)
        
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum similarity score
        
        Returns:
            List of search results
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Query vector database
        results = self.vector_db.query(query_embedding, n_results=top_k)
        
        # Format results
        formatted_results = []
        if results and results.get("ids") and results["ids"][0]:
            for i, (doc_id, doc_content, doc_metadata, doc_distance) in enumerate(
                zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )
            ):
                similarity_score = 1 - doc_distance
                
                if similarity_score >= score_threshold:
                    formatted_results.append({
                        "id": doc_id,
                        "content": doc_content,
                        "metadata": doc_metadata,
                        "score": similarity_score,
                        "rank": i + 1
                    })
        
        return formatted_results
    
    def delete(self, ids: List[str]):
        """
        Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
        """
        self.vector_db.delete(ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Statistics dictionary
        """
        return self.vector_db.get_collection_stats()
    
    def clear(self):
        """Clear all documents from the vector store."""
        self.vector_db.clear_collection()
