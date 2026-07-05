"""Response constraints and hallucination checks for LLM outputs."""

import re
from typing import List, Dict, Any, Optional, Set
from src.core.logger import Logger


class Guardrails:
    """Apply constraints and checks to LLM responses for safety and accuracy."""
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the guardrails.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or Logger()
        self.forbidden_patterns: Set[str] = set()
        self.required_keywords: Set[str] = set()
        self.max_length: Optional[int] = None
        self.min_length: Optional[int] = None
        self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Load default forbidden patterns and constraints."""
        self.forbidden_patterns = {
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript protocol
            r'data:text/html',  # Data URI
        }
    
    def add_forbidden_pattern(self, pattern: str):
        """
        Add a forbidden regex pattern.
        
        Args:
            pattern: Regex pattern to forbid
        """
        self.forbidden_patterns.add(pattern)
        self.logger.info(f"Added forbidden pattern: {pattern}")
    
    def add_required_keyword(self, keyword: str):
        """
        Add a required keyword that must be present in response.
        
        Args:
            keyword: Required keyword
        """
        self.required_keywords.add(keyword.lower())
        self.logger.info(f"Added required keyword: {keyword}")
    
    def set_length_constraints(self, min_length: Optional[int] = None, max_length: Optional[int] = None):
        """
        Set length constraints for responses.
        
        Args:
            min_length: Minimum character length
            max_length: Maximum character length
        """
        self.min_length = min_length
        self.max_length = max_length
        self.logger.info(f"Set length constraints: min={min_length}, max={max_length}")
    
    def check_response(self, response: str) -> Dict[str, Any]:
        """
        Check a response against all guardrails.
        
        Args:
            response: LLM response to check
        
        Returns:
            Dictionary with check results
        """
        results = {
            "passed": True,
            "violations": [],
            "checks": {}
        }
        
        # Check forbidden patterns
        pattern_check = self._check_forbidden_patterns(response)
        results["checks"]["forbidden_patterns"] = pattern_check
        if not pattern_check["passed"]:
            results["passed"] = False
            results["violations"].extend(pattern_check["violations"])
        
        # Check required keywords
        keyword_check = self._check_required_keywords(response)
        results["checks"]["required_keywords"] = keyword_check
        if not keyword_check["passed"]:
            results["passed"] = False
            results["violations"].extend(keyword_check["violations"])
        
        # Check length constraints
        length_check = self._check_length(response)
        results["checks"]["length"] = length_check
        if not length_check["passed"]:
            results["passed"] = False
            results["violations"].extend(length_check["violations"])
        
        # Check for hallucination indicators
        hallucination_check = self._check_hallucination_indicators(response)
        results["checks"]["hallucination"] = hallucination_check
        if not hallucination_check["passed"]:
            results["passed"] = False
            results["violations"].extend(hallucination_check["violations"])
        
        if results["passed"]:
            self.logger.info("Response passed all guardrails")
        else:
            self.logger.warning(f"Response failed guardrails: {results['violations']}")
        
        return results
    
    def _check_forbidden_patterns(self, response: str) -> Dict[str, Any]:
        """
        Check for forbidden patterns in response.
        
        Args:
            response: Response to check
        
        Returns:
            Check result dictionary
        """
        violations = []
        for pattern in self.forbidden_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                violations.append(f"Found forbidden pattern: {pattern}")
        
        return {
            "passed": len(violations) == 0,
            "violations": violations
        }
    
    def _check_required_keywords(self, response: str) -> Dict[str, Any]:
        """
        Check for required keywords in response.
        
        Args:
            response: Response to check
        
        Returns:
            Check result dictionary
        """
        violations = []
        response_lower = response.lower()
        
        for keyword in self.required_keywords:
            if keyword not in response_lower:
                violations.append(f"Missing required keyword: {keyword}")
        
        return {
            "passed": len(violations) == 0,
            "violations": violations
        }
    
    def _check_length(self, response: str) -> Dict[str, Any]:
        """
        Check length constraints.
        
        Args:
            response: Response to check
        
        Returns:
            Check result dictionary
        """
        violations = []
        length = len(response)
        
        if self.min_length and length < self.min_length:
            violations.append(f"Response too short: {length} < {self.min_length}")
        
        if self.max_length and length > self.max_length:
            violations.append(f"Response too long: {length} > {self.max_length}")
        
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "length": length
        }
    
    def _check_hallucination_indicators(self, response: str) -> Dict[str, Any]:
        """
        Check for potential hallucination indicators.
        
        Args:
            response: Response to check
        
        Returns:
            Check result dictionary
        """
        violations = []
        
        # Check for uncertainty phrases
        uncertainty_phrases = [
            "i'm not sure",
            "i don't know",
            "i cannot confirm",
            "it might be",
            "possibly",
            "perhaps"
        ]
        
        response_lower = response.lower()
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                violations.append(f"Uncertainty indicator: '{phrase}'")
        
        # Check for excessive speculation
        speculation_count = response_lower.count("might") + response_lower.count("could")
        if speculation_count > 3:
            violations.append(f"Excessive speculation: {speculation_count} instances")
        
        return {
            "passed": len(violations) == 0,
            "violations": violations
        }
    
    def sanitize_response(self, response: str) -> str:
        """
        Sanitize response by removing forbidden patterns.
        
        Args:
            response: Response to sanitize
        
        Returns:
            Sanitized response
        """
        sanitized = response
        for pattern in self.forbidden_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        self.logger.info("Response sanitized")
        return sanitized
    
    def filter_pii(self, response: str) -> str:
        """
        Filter potential PII (Personally Identifiable Information).
        
        Args:
            response: Response to filter
        
        Returns:
            Response with PII masked
        """
        # Simple PII patterns (in production, use more sophisticated detection)
        pii_patterns = {
            r'\b\d{3}-\d{2}-\d{4}\b': '[SSN]',  # SSN pattern
            r'\b\d{3}-\d{3}-\d{4}\b': '[PHONE]',  # Phone pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b': '[EMAIL]',  # Email
        }
        
        filtered = response
        for pattern, replacement in pii_patterns.items():
            filtered = re.sub(pattern, replacement, filtered)
        
        self.logger.info("PII filtered from response")
        return filtered
