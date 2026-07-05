"""Search wrapper for easy retrieval operations."""

from typing import List, Optional, Dict, Any
from src.retrieval.embedder import Embedder
from src.retrieval.vector_db import VectorDatabase
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.rerank import Reranker
from src.core.logger import Logger
from src.core.config import Config


class Search:
    """Convenience class for search and retrieval operations."""
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        logger: Optional[Logger] = None
    ):
        """
        Initialize the search wrapper.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            logger: Optional logger instance
        """
        self.logger = logger or Logger()
        self.embedder = Embedder(logger=logger)
        self.vector_db = VectorDatabase(collection_name, persist_directory, logger)
        self.hybrid_search = HybridSearch(self.embedder, self.vector_db, logger)
        self.reranker = Reranker(logger=logger)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
        rerank: bool = True,
        use_mmr: bool = True,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            method: Search method ('hybrid', 'dense', 'sparse')
            rerank: Whether to rerank results
            use_mmr: Whether to use MMR for diversity
            score_threshold: Minimum similarity score
        
        Returns:
            List of search results
        """
        if method == "hybrid":
            results = self.hybrid_search.search(
                query=query,
                top_k=top_k,
                use_mmr=use_mmr
            )
        elif method == "dense":
            results = self._dense_search(query, top_k)
        elif method == "sparse":
            results = self._sparse_search(query, top_k)
        else:
            self.logger.error(f"Unknown search method: {method}")
            return []
        
        # Apply score threshold
        if score_threshold > 0:
            results = [r for r in results if r.get("score", 0) >= score_threshold]
        
        # Rerank if requested
        if rerank:
            results = self.reranker.rerank(query, results, top_k)
        
        return results
    
    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform dense vector search.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of search results
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.vector_db.query(query_embedding, n_results=top_k)
        
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
                formatted_results.append({
                    "id": doc_id,
                    "content": doc_content,
                    "metadata": doc_metadata,
                    "score": similarity_score,
                    "search_type": "dense"
                })
        
        return formatted_results
    
    def _sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform sparse keyword search.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of search results
        """
        query_terms = set(query.lower().split())
        
        # Get documents from vector DB
        all_results = self.vector_db.query(
            self.embedder.embed_query(query),
            n_results=100
        )
        
        formatted_results = []
        if all_results and all_results.get("documents") and all_results["documents"][0]:
            for i, (doc_id, doc_content, doc_metadata) in enumerate(
                zip(
                    all_results["ids"][0],
                    all_results["documents"][0],
                    all_results["metadatas"][0]
                )
            ):
                content_lower = doc_content.lower()
                matches = sum(1 for term in query_terms if term in content_lower)
                score = matches / len(query_terms) if query_terms else 0
                
                if score > 0:
                    formatted_results.append({
                        "id": doc_id,
                        "content": doc_content,
                        "metadata": doc_metadata,
                        "score": score,
                        "search_type": "sparse"
                    })
        
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        return formatted_results[:top_k]
    
    def get_search_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about search results.
        
        Args:
            results: List of search results
        
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        scores = [r.get("score", 0) for r in results]
        
        return {
            "total_results": len(results),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "search_types": {r.get("search_type", "unknown") for r in results}
        }
