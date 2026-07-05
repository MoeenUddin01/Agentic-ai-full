"""Security management for AgenticAI."""

import os
from typing import Optional, List, Dict, Any
from enum import Enum


class Role(Enum):
    """User roles for access control."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class SecurityManager:
    """Manages security and access control for the application."""
    
    def __init__(self):
        """Initialize the security manager."""
        self.api_keys: Dict[str, str] = {}
        self.user_roles: Dict[str, Role] = {}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment variables."""
        self.api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "groq": os.getenv("GROQ_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY"),
            "tavily": os.getenv("TAVILY_API_KEY"),
        }
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider.
        
        Args:
            provider: The provider name (e.g., 'openai', 'groq')
        
        Returns:
            The API key if found, None otherwise
        """
        return self.api_keys.get(provider.lower())
    
    def set_api_key(self, provider: str, api_key: str):
        """
        Set API key for a specific provider.
        
        Args:
            provider: The provider name
            api_key: The API key
        """
        self.api_keys[provider.lower()] = api_key
    
    def validate_api_key(self, provider: str) -> bool:
        """
        Validate if an API key exists for the provider.
        
        Args:
            provider: The provider name
        
        Returns:
            True if API key exists, False otherwise
        """
        return bool(self.get_api_key(provider))
    
    def add_user(self, user_id: str, role: Role = Role.USER):
        """
        Add a user with a specific role.
        
        Args:
            user_id: The user identifier
            role: The user's role
        """
        self.user_roles[user_id] = role
    
    def remove_user(self, user_id: str):
        """
        Remove a user.
        
        Args:
            user_id: The user identifier
        """
        if user_id in self.user_roles:
            del self.user_roles[user_id]
    
    def get_user_role(self, user_id: str) -> Optional[Role]:
        """
        Get the role of a user.
        
        Args:
            user_id: The user identifier
        
        Returns:
            The user's role if found, None otherwise
        """
        return self.user_roles.get(user_id)
    
    def check_permission(self, user_id: str, required_role: Role) -> bool:
        """
        Check if a user has the required permission level.
        
        Args:
            user_id: The user identifier
            required_role: The required role
        
        Returns:
            True if user has required or higher role, False otherwise
        """
        user_role = self.get_user_role(user_id)
        if not user_role:
            return False
        
        role_hierarchy = {
            Role.GUEST: 0,
            Role.USER: 1,
            Role.ADMIN: 2
        }
        
        return role_hierarchy.get(user_role, 0) >= role_hierarchy.get(required_role, 0)
    
    def mask_api_key(self, api_key: str, visible_chars: int = 4) -> str:
        """
        Mask an API key for display purposes.
        
        Args:
            api_key: The API key to mask
            visible_chars: Number of characters to show at the end
        
        Returns:
            The masked API key
        """
        if not api_key or len(api_key) <= visible_chars:
            return "***"
        return "*" * (len(api_key) - visible_chars) + api_key[-visible_chars:]
