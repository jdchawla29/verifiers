"""Centralized logging configuration for the verifiers package."""

import logging
import logging.config
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def get_default_logging_config() -> Dict[str, Any]:
    """Get the default logging configuration dictionary."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            },
            "structured": {
                "format": "%(asctime)s|%(name)s|%(levelname)s|%(message)s|%(pathname)s:%(lineno)d|%(funcName)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "detailed",
                "stream": "ext://sys.stderr"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "structured",
                "filename": str(log_dir / "verifiers.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8"
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(log_dir / "verifiers_errors.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3,
                "encoding": "utf-8"
            }
        },
        "loggers": {
            "verifiers": {
                "level": "INFO",
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "verifiers.trainers": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "verifiers.envs": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "verifiers.parsers": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "verifiers.rubrics": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "verifiers.examples": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }


def setup_centralized_logging(
    config_dict: Optional[Dict[str, Any]] = None,
    log_level: Optional[str] = None,
    log_to_file: bool = True,
    log_format: Optional[str] = None
) -> None:
    """
    Set up centralized logging configuration for the verifiers package.
    
    Args:
        config_dict: Optional custom logging configuration dictionary
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to enable file logging (default: True)
        log_format: Override log format ('detailed', 'simple', 'structured')
    """
    # Use provided config or default
    config = config_dict or get_default_logging_config()
    
    # Apply environment variable overrides
    env_log_level = os.environ.get("VERIFIERS_LOG_LEVEL", log_level)
    env_log_format = os.environ.get("VERIFIERS_LOG_FORMAT", log_format)
    env_log_to_file = os.environ.get("VERIFIERS_LOG_TO_FILE", str(log_to_file)).lower() == "true"
    
    # Update log level if specified
    if env_log_level:
        for logger_config in config["loggers"].values():
            logger_config["level"] = env_log_level.upper()
        config["root"]["level"] = env_log_level.upper()
    
    # Update formatter if specified
    if env_log_format and env_log_format in config["formatters"]:
        for handler in config["handlers"].values():
            handler["formatter"] = env_log_format
    
    # Disable file handlers if not logging to file
    if not env_log_to_file:
        # Remove file handlers from all loggers
        for logger_config in config["loggers"].values():
            logger_config["handlers"] = [h for h in logger_config["handlers"] 
                                       if h not in ["file", "error_file"]]
        # Remove file handler definitions
        config["handlers"] = {k: v for k, v in config["handlers"].items() 
                            if k not in ["file", "error_file"]}
    
    # Apply the configuration
    logging.config.dictConfig(config)
    
    # Log the initialization
    logger = logging.getLogger("verifiers")
    logger.info(f"Logging initialized with level: {env_log_level or 'INFO'}, "
               f"format: {env_log_format or 'detailed'}, "
               f"file logging: {env_log_to_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the appropriate name.
    
    Args:
        name: Logger name (e.g., "verifiers.envs.MyEnvironment")
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def get_module_logger(module_name: str, class_name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with standardized naming for verifiers modules.
    
    Args:
        module_name: The module name (e.g., "envs", "parsers", "rubrics")
        class_name: Optional class name to append
        
    Returns:
        Logger instance with standardized name
    """
    if class_name:
        logger_name = f"verifiers.{module_name}.{class_name}"
    else:
        logger_name = f"verifiers.{module_name}"
    return logging.getLogger(logger_name)


# Convenience function for backward compatibility
def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: str = "%Y-%m-%d %H:%M:%S"
) -> None:
    """
    Set up logging for the verifiers package (backward compatible).
    
    Args:
        level: The logging level (default: INFO)
        log_format: The log format string (uses default if None)
        date_format: The date format string
    """
    # Create a simple config that matches the old behavior
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    simple_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": log_format,
                "datefmt": date_format
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "default",
                "stream": "ext://sys.stderr"
            }
        },
        "loggers": {
            "verifiers": {
                "level": level,
                "handlers": ["console"],
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(simple_config)