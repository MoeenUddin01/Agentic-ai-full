"""Logging configuration for AgenticAI."""

import logging
import sys
from typing import Optional
from pathlib import Path


class Logger:
    """Custom logger for AgenticAI applications."""
    
    _instances = {}
    
    def __new__(cls, name: str = "agenticai", log_file: Optional[str] = None, log_level: str = "INFO"):
        """Create or return existing logger instance."""
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
            instance._initialize(name, log_file, log_level)
        return cls._instances[name]
    
    def _initialize(self, name: str, log_file: Optional[str], log_level: str):
        """Initialize the logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
