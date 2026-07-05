"""Cross-encoder re-rankers for improving retrieval quality."""

import numpy as np
from typing import List, Dict, Any, Optional
from src.core.logger import Logger


class Reranker:
    """Re-rank search results using cross-encoder models for better relevance."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        logger: Optional[Logger] = None
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model
            logger: Optional logger instance
        """
        self.model_name = model_name
        self.logger = logger or Logger()
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
            self.logger.info(f"Loaded cross-encoder model: {self.model_name}")
        except ImportError:
            self.logger.warning("sentence-transformers not installed, using fallback reranking")
        except Exception as e:
            self.logger.warning(f"Failed to load cross-encoder: {e}")
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results based on query-document relevance.
        
        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Number of top results to return
        
        Returns:
            Re-ranked results
        """
        if not results:
            return []
        
        self.logger.info(f"Reranking {len(results)} results")
        
        if self.model is None:
            # Fallback: simple keyword-based reranking
            return self._fallback_rerank(query, results, top_k)
        
        try:
            # Prepare query-document pairs
            pairs = [(query, result["content"]) for result in results]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Add scores to results
            for i, result in enumerate(results):
                result["rerank_score"] = float(scores[i])
                result["original_score"] = result.get("score", 0.0)
            
            # Sort by rerank score
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
            # Return top_k if specified
            if top_k:
                results = results[:top_k]
            
            self.logger.info(f"Reranked to {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during reranking: {e}")
            return self._fallback_rerank(query, results, top_k)
    
    def _fallback_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fallback reranking using keyword matching.
        
        Args:
            query: Search query
            results: List of search results
            top_k: Number of top results to return
        
        Returns:
            Re-ranked results
        """
        query_terms = set(query.lower().split())
        
        for result in results:
            content_lower = result["content"].lower()
            
            # Calculate term overlap score
            matches = sum(1 for term in query_terms if term in content_lower)
            overlap_score = matches / len(query_terms) if query_terms else 0
            
            # Combine with original score
            original_score = result.get("score", 0.0)
            result["rerank_score"] = 0.7 * original_score + 0.3 * overlap_score
            result["original_score"] = original_score
        
        # Sort by rerank score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    def rerank_with_diversity(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        diversity_lambda: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Re-rank with diversity consideration using MMR.
        
        Args:
            query: Search query
            results: List of search results
            top_k: Number of results to return
            diversity_lambda: Balance between relevance and diversity
        
        Returns:
            Diversified re-ranked results
        """
        # First rerank by relevance
        reranked = self.rerank(query, results)
        
        # Apply MMR for diversity
        selected = []
        remaining = reranked.copy()
        
        if remaining:
            selected.append(remaining.pop(0))
        
        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = -float('inf')
            
            for i, candidate in enumerate(remaining):
                relevance = candidate["rerank_score"]
                
                # Calculate max similarity to selected
                max_similarity = 0
                for sel in selected:
                    similarity = self._compute_similarity(candidate, sel)
                    max_similarity = max(max_similarity, similarity)
                
                mmr_score = diversity_lambda * relevance - (1 - diversity_lambda) * max_similarity
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        self.logger.info(f"Reranked with diversity to {len(selected)} results")
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
        words1 = set(doc1["content"].lower().split())
        words2 = set(doc2["content"].lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_rerank_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about reranked results.
        
        Args:
            results: Reranked results
        
        Returns:
            Statistics dictionary
        """
        if not results:
            return {}
        
        rerank_scores = [r.get("rerank_score", 0) for r in results]
        original_scores = [r.get("original_score", 0) for r in results]
        
        return {
            "total_results": len(results),
            "avg_rerank_score": np.mean(rerank_scores),
            "avg_original_score": np.mean(original_scores),
            "score_improvement": np.mean(rerank_scores) - np.mean(original_scores),
            "top_rerank_score": max(rerank_scores),
            "top_original_score": max(original_scores)
        }
