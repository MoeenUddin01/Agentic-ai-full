"""Evaluation module for retrieval quality and citation metrics."""

from typing import List, Dict, Any, Optional, Set
import numpy as np
from src.core.logger import Logger


class Metrics:
    """Calculate evaluation metrics for retrieval quality and citation accuracy."""
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or Logger()
    
    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Precision@K score
        """
        if k > len(retrieved):
            k = len(retrieved)
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = len(retrieved_at_k & relevant)
        
        precision = relevant_retrieved / k if k > 0 else 0.0
        self.logger.debug(f"Precision@{k}: {precision:.4f}")
        return precision
    
    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0
        
        if k > len(retrieved):
            k = len(retrieved)
        
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = len(retrieved_at_k & relevant)
        
        recall = relevant_retrieved / len(relevant)
        self.logger.debug(f"Recall@{k}: {recall:.4f}")
        return recall
    
    def f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        
        Args:
            precision: Precision score
            recall: Recall score
        
        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        self.logger.debug(f"F1 Score: {f1:.4f}")
        return f1
    
    def mean_reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
        
        Returns:
            MRR score
        """
        if not relevant:
            return 0.0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                mrr = 1.0 / i
                self.logger.debug(f"MRR: {mrr:.4f}")
                return mrr
        
        return 0.0
    
    def average_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
        
        Returns:
            Average Precision score
        """
        if not relevant:
            return 0.0
        
        relevant_retrieved = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                relevant_retrieved += 1
                precision_sum += relevant_retrieved / i
        
        ap = precision_sum / len(relevant)
        self.logger.debug(f"Average Precision: {ap:.4f}")
        return ap
    
    def ndcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider
        
        Returns:
            NDCG@K score
        """
        if k > len(retrieved):
            k = len(retrieved)
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculate ideal DCG
        idcg = 0.0
        for i in range(1, min(k, len(relevant)) + 1):
            idcg += 1.0 / np.log2(i + 1)
        
        ndcg = dcg / idcg if idcg > 0 else 0.0
        self.logger.debug(f"NDCG@{k}: {ndcg:.4f}")
        return ndcg
    
    def citation_accuracy(
        self,
        response: str,
        retrieved_docs: List[Dict[str, Any]],
        ground_truth: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate citation accuracy metrics.
        
        Args:
            response: Generated response
            retrieved_docs: List of retrieved documents
            ground_truth: Optional set of ground truth document IDs
        
        Returns:
            Dictionary with citation metrics
        """
        # Extract cited document IDs from response (simplified)
        cited_docs = set()
        for doc in retrieved_docs:
            if doc.get("id") in response:
                cited_docs.add(doc["id"])
        
        retrieved_ids = {doc["id"] for doc in retrieved_docs}
        
        metrics = {
            "citations_count": len(cited_docs),
            "retrieved_count": len(retrieved_ids),
            "citation_rate": len(cited_docs) / len(retrieved_ids) if retrieved_ids else 0.0
        }
        
        if ground_truth:
            correct_citations = len(cited_docs & ground_truth)
            metrics["citation_precision"] = correct_citations / len(cited_docs) if cited_docs else 0.0
            metrics["citation_recall"] = correct_citations / len(ground_truth) if ground_truth else 0.0
        
        self.logger.info(f"Citation metrics: {metrics}")
        return metrics
    
    def retrieval_coverage(
        self,
        queries: List[str],
        retriever,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Calculate retrieval coverage across multiple queries.
        
        Args:
            queries: List of query strings
            retriever: Retriever instance with retrieve method
            k: Number of results per query
        
        Returns:
            Dictionary with coverage metrics
        """
        all_retrieved = set()
        unique_docs_per_query = []
        
        for query in queries:
            results = retriever.retrieve(query, top_k=k)
            retrieved_ids = {doc["id"] for doc in results}
            all_retrieved.update(retrieved_ids)
            unique_docs_per_query.append(len(retrieved_ids))
        
        coverage = {
            "total_queries": len(queries),
            "total_unique_docs": len(all_retrieved),
            "avg_unique_per_query": np.mean(unique_docs_per_query),
            "coverage_ratio": len(all_retrieved) / (len(queries) * k) if queries else 0.0
        }
        
        self.logger.info(f"Retrieval coverage: {coverage}")
        return coverage
    
    def calculate_all_metrics(
        self,
        retrieved: List[str],
        relevant: Set[str],
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Calculate all standard retrieval metrics.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k_values: List of k values for P@K and R@K
        
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "mrr": self.mean_reciprocal_rank(retrieved, relevant),
            "map": self.average_precision(retrieved, relevant),
            "precision_at_k": {},
            "recall_at_k": {},
            "f1_at_k": {},
            "ndcg_at_k": {}
        }
        
        for k in k_values:
            precision = self.precision_at_k(retrieved, relevant, k)
            recall = self.recall_at_k(retrieved, relevant, k)
            f1 = self.f1_score(precision, recall)
            ndcg = self.ndcg_at_k(retrieved, relevant, k)
            
            metrics["precision_at_k"][f"p@{k}"] = precision
            metrics["recall_at_k"][f"r@{k}"] = recall
            metrics["f1_at_k"][f"f1@{k}"] = f1
            metrics["ndcg_at_k"][f"ndcg@{k}"] = ndcg
        
        self.logger.info("Calculated all retrieval metrics")
        return metrics
