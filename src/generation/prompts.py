"""Prompt templates and context budgeting for LLM generation."""

from typing import Dict, Any, List, Optional
from jinja2 import Template
from src.core.logger import Logger


class PromptManager:
    """Manage prompt templates and context budgeting for LLM generation."""
    
    def __init__(self, logger: Optional[Logger] = None):
        """
        Initialize the prompt manager.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or Logger()
        self.templates: Dict[str, Template] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        self.templates = {
            "rag": Template(
                "Use the following context to answer the question.\n\n"
                "Context:\n{{ context }}\n\n"
                "Question: {{ question }}\n\n"
                "Answer:"
            ),
            "summarization": Template(
                "Summarize the following text in {{ max_sentences }} sentence(s).\n\n"
                "Text:\n{{ text }}\n\n"
                "Summary:"
            ),
            "extraction": Template(
                "Extract the following information from the text: {{ fields }}\n\n"
                "Text:\n{{ text }}\n\n"
                "Extracted information:"
            ),
            "qa": Template(
                "Answer the following question based on your knowledge.\n\n"
                "Question: {{ question }}\n\n"
                "Answer:"
            ),
            "chat": Template(
                "You are a helpful AI assistant. Respond to the user's message.\n\n"
                "User: {{ message }}\n\n"
                "Assistant:"
            )
        }
    
    def add_template(self, name: str, template: str):
        """
        Add a custom prompt template.
        
        Args:
            name: Name of the template
            template: Template string (Jinja2 format)
        """
        self.templates[name] = Template(template)
        self.logger.info(f"Added template: {name}")
    
    def render(
        self,
        template_name: str,
        **kwargs
    ) -> str:
        """
        Render a prompt template with given variables.
        
        Args:
            template_name: Name of the template to render
            **kwargs: Variables to substitute in the template
        
        Returns:
            Rendered prompt string
        """
        if template_name not in self.templates:
            self.logger.error(f"Template not found: {template_name}")
            return ""
        
        try:
            template = self.templates[template_name]
            rendered = template.render(**kwargs)
            self.logger.debug(f"Rendered template: {template_name}")
            return rendered
        except Exception as e:
            self.logger.error(f"Error rendering template {template_name}: {e}")
            return ""
    
    def budget_context(
        self,
        context: str,
        max_tokens: int = 4000,
        tokens_per_char: float = 0.25
    ) -> str:
        """
        Budget context to fit within token limits.
        
        Args:
            context: Context string to budget
            max_tokens: Maximum allowed tokens
            tokens_per_char: Approximate tokens per character
        
        Returns:
            Truncated context fitting within budget
        """
        max_chars = int(max_tokens / tokens_per_char)
        
        if len(context) <= max_chars:
            return context
        
        # Truncate from the end, keeping beginning
        truncated = context[:max_chars]
        self.logger.info(f"Context truncated from {len(context)} to {len(truncated)} chars")
        return truncated
    
    def format_context(
        self,
        documents: List[Dict[str, Any]],
        max_docs: int = 5,
        max_chars_per_doc: int = 500
    ) -> str:
        """
        Format retrieved documents into context string.
        
        Args:
            documents: List of retrieved documents
            max_docs: Maximum number of documents to include
            max_chars_per_doc: Maximum characters per document
        
        Returns:
            Formatted context string
        """
        selected_docs = documents[:max_docs]
        context_parts = []
        
        for i, doc in enumerate(selected_docs, 1):
            content = doc.get("content", "")
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc] + "..."
            
            context_parts.append(f"Document {i}:\n{content}")
        
        context = "\n\n".join(context_parts)
        self.logger.info(f"Formatted context from {len(selected_docs)} documents")
        return context
    
    def get_template_names(self) -> List[str]:
        """
        Get list of available template names.
        
        Returns:
            List of template names
        """
        return list(self.templates.keys())
