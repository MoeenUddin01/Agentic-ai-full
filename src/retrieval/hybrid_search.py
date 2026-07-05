"""Hybrid search combining dense and sparse retrieval with MMR."""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.retrieval.embedder import Embedder
from src.retrieval.vector_db import VectorDatabase
from src.core.logger import Logger


class HybridSearch:
    """Combine dense (vector) and sparse (keyword) search with MMR for diverse results."""
    
    def __init__(
        self,
        embedder: Embedder,
        vector_db: VectorDatabase,
        logger: Optional[Logger] = None
    ):
        """
        Initialize hybrid search.
        
        Args:
            embedder: Embedder instance for dense search
            vector_db: Vector database for dense search
            logger: Optional logger instance
        """
        self.embedder = embedder
        self.vector_db = vector_db
        self.logger = logger or Logger()
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            dense_weight: Weight for dense (vector) search
            sparse_weight: Weight for sparse (keyword) search
            use_mmr: Whether to use Maximal Marginal Relevance
            mmr_lambda: Lambda parameter for MMR (diversity vs relevance)
        
        Returns:
            List of ranked documents with scores
        """
        self.logger.info(f"Performing hybrid search for: {query}")
        
        # Dense search (vector similarity)
        dense_results = self._dense_search(query, top_k * 2)
        
        # Sparse search (keyword matching)
        sparse_results = self._sparse_search(query, top_k * 2)
        
        # Combine results
        combined = self._combine_results(
            dense_results,
            sparse_results,
            dense_weight,
            sparse_weight
        )
        
        # Apply MMR for diversity if requested
        if use_mmr:
            combined = self._apply_mmr(combined, top_k, mmr_lambda)
        else:
            combined = combined[:top_k]
        
        self.logger.info(f"Returning {len(combined)} results")
        return combined
    
    def _dense_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Perform dense vector search.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of dense search results
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
            List of sparse search results
        """
        # Simple keyword matching implementation
        # In production, use BM25 or similar
        query_terms = set(query.lower().split())
        
        # Get all documents from vector DB
        all_results = self.vector_db.query(
            self.embedder.embed_query(query),
            n_results=100  # Get more for sparse filtering
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
                # Calculate keyword match score
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
        
        # Sort by score and return top_k
        formatted_results.sort(key=lambda x: x["score"], reverse=True)
        return formatted_results[:top_k]
    
    def _combine_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine dense and sparse search results.
        
        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results
            dense_weight: Weight for dense results
            sparse_weight: Weight for sparse results
        
        Returns:
            Combined and re-ranked results
        """
        combined = {}
        
        # Add dense results
        for result in dense_results:
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = result
                combined[doc_id]["combined_score"] = result["score"] * dense_weight
            else:
                combined[doc_id]["combined_score"] += result["score"] * dense_weight
        
        # Add sparse results
        for result in sparse_results:
            doc_id = result["id"]
            if doc_id not in combined:
                combined[doc_id] = result
                combined[doc_id]["combined_score"] = result["score"] * sparse_weight
            else:
                combined[doc_id]["combined_score"] += result["score"] * sparse_weight
        
        # Sort by combined score
        results = list(combined.values())
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return results
    
    def _apply_mmr(
        self,
        results: List[Dict[str, Any]],
        top_k: int,
        lambda_param: float
    ) -> List[Dict[str, Any]]:
        """
        Apply Maximal Marginal Relevance for diverse results.
        
        Args:
            results: List of search results
            top_k: Number of results to return
            lambda_param: Balance between relevance and diversity
        
        Returns:
            Diversified results
        """
        if not results:
            return []
        
        selected = []
        remaining = results.copy()
        
        # Select first result (highest score)
        if remaining:
            selected.append(remaining.pop(0))
        
        # Select remaining results using MMR
        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = -float('inf')
            
            for i, candidate in enumerate(remaining):
                # Relevance component
                relevance = candidate["combined_score"]
                
                # Diversity component (max similarity to selected)
                max_similarity = 0
                for sel in selected:
                    similarity = self._compute_similarity(candidate, sel)
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _compute_similarity(self, doc1: Dict[str, Any], doc2: Dict[str, Any]) -> float:
        """
        Compute similarity between two documents.
        
        Args:
            doc1: First document
            doc2: Second document
        
        Returns:
            Similarity score
        """
        # Simple content overlap similarity
        words1 = set(doc1["content"].lower().split())
        words2 = set(doc2["content"].lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
