"""Document indexers for metadata handling and vector uploads."""

import uuid
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.core.logger import Logger


class DocumentIndexer:
    """Handle metadata processing and preparation for vector database upload."""
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the document indexer.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or Logger()
    
    def prepare_for_upload(
        self,
        chunks: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare documents for vector database upload.
        
        Args:
            chunks: List of document chunks
            embeddings: Optional pre-computed embeddings
        
        Returns:
            List of dictionaries ready for upload
        """
        self.logger.info(f"Preparing {len(chunks)} chunks for upload")
        
        prepared_docs = []
        
        for i, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid4())
            
            prepared_doc = {
                "id": doc_id,
                "content": chunk.page_content,
                "metadata": self._enhance_metadata(chunk.metadata, i),
                "chunk_index": i
            }
            
            if embeddings and i < len(embeddings):
                prepared_doc["embedding"] = embeddings[i]
            
            prepared_docs.append(prepared_doc)
        
        self.logger.info(f"Prepared {len(prepared_docs)} documents for upload")
        return prepared_docs
    
    def _enhance_metadata(self, metadata: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Enhance document metadata with additional information.
        
        Args:
            metadata: Original metadata
            index: Chunk index
        
        Returns:
            Enhanced metadata
        """
        enhanced = metadata.copy()
        enhanced["chunk_index"] = index
        enhanced["content_length"] = len(metadata.get("content", ""))
        
        # Add timestamp if not present
        if "timestamp" not in enhanced:
            import time
            enhanced["timestamp"] = time.time()
        
        return enhanced
    
    def add_custom_metadata(
        self,
        documents: List[Dict[str, Any]],
        custom_fields: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Add custom metadata fields to documents.
        
        Args:
            documents: List of prepared documents
            custom_fields: Dictionary of custom fields to add
        
        Returns:
            Documents with added custom metadata
        """
        self.logger.info(f"Adding custom metadata to {len(documents)} documents")
        
        for doc in documents:
            doc["metadata"].update(custom_fields)
        
        return documents
    
    def filter_by_metadata(
        self,
        documents: List[Dict[str, Any]],
        filter_criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter documents based on metadata criteria.
        
        Args:
            documents: List of prepared documents
            filter_criteria: Dictionary of field:value pairs to match
        
        Returns:
            Filtered documents
        """
        filtered = []
        
        for doc in documents:
            match = True
            for key, value in filter_criteria.items():
                if doc["metadata"].get(key) != value:
                    match = False
                    break
            
            if match:
                filtered.append(doc)
        
        self.logger.info(f"Filtered {len(documents)} documents to {len(filtered)}")
        return filtered
    
    def deduplicate_by_content(
        self,
        documents: List[Dict[str, Any]],
        similarity_threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate documents based on content similarity.
        
        Args:
            documents: List of prepared documents
            similarity_threshold: Threshold for considering content similar
        
        Returns:
            Deduplicated documents
        """
        self.logger.info(f"Deduplicating {len(documents)} documents")
        
        seen_contents = set()
        unique_docs = []
        
        for doc in documents:
            content = doc["content"]
            # Simple hash-based deduplication
            content_hash = hash(content)
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
        
        self.logger.info(f"Removed {len(documents) - len(unique_docs)} duplicates")
        return unique_docs
    
    def get_upload_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about documents ready for upload.
        
        Args:
            documents: List of prepared documents
        
        Returns:
            Statistics dictionary
        """
        if not documents:
            return {}
        
        content_lengths = [len(doc["content"]) for doc in documents]
        
        return {
            "total_documents": len(documents),
            "min_content_length": min(content_lengths),
            "max_content_length": max(content_lengths),
            "avg_content_length": sum(content_lengths) / len(content_lengths),
            "has_embeddings": any("embedding" in doc for doc in documents),
            "metadata_fields": list(documents[0]["metadata"].keys()) if documents else []
        }
