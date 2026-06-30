"""
Logging setup module.

Provides a configured logger that outputs INFO logs to the console
and DEBUG logs to a log file.
"""

import logging
import os
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    Creates and configures a standard logger.
    
    Args:
        name: Name of the logger (usually __name__).
        
    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # File Handler (DEBUG level)
    file_handler = logging.FileHandler("logs/app.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console Handler (INFO level)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
