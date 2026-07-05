"""Configuration management for AgenticAI."""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseSettings, Field


class Config(BaseSettings):
    """Global configuration settings for the application."""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    GROQ_API_KEY: Optional[str] = Field(None, env="GROQ_API_KEY")
    GOOGLE_API_KEY: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    TAVILY_API_KEY: Optional[str] = Field(None, env="TAVILY_API_KEY")
    
    # Model Configuration
    DEFAULT_MODEL_PROVIDER: str = Field("groq", env="DEFAULT_MODEL_PROVIDER")
    DEFAULT_MODEL: str = Field("llama-3.3-70b-versatile", env="DEFAULT_MODEL")
    
    # Vector Database Configuration
    VECTOR_DB_PATH: str = Field("./vector_store", env="VECTOR_DB_PATH")
    VECTOR_DB_COLLECTION: str = Field("pdf_documents", env="VECTOR_DB_COLLECTION")
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(1536, env="EMBEDDING_DIMENSION")
    
    # Chunking Configuration
    CHUNK_SIZE: int = Field(1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(200, env="CHUNK_OVERLAP")
    
    # Retrieval Configuration
    DEFAULT_TOP_K: int = Field(5, env="DEFAULT_TOP_K")
    DEFAULT_SCORE_THRESHOLD: float = Field(0.0, env="DEFAULT_SCORE_THRESHOLD")
    
    # LLM Configuration
    LLM_TEMPERATURE: float = Field(0.7, env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(500, env="LLM_MAX_TOKENS")
    
    # Logging Configuration
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field("agenticai.log", env="LOG_FILE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment variables."""
        load_dotenv()
        return cls()
