"""
Enhanced logging utilities for the verifiers package.
Provides consistent logging with structured output and rich formatting.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class StructuredFormatter(logging.Formatter):
    """Formatter that adds structured data to log records."""
    
    def format(self, record):
        # Start with standard format
        msg = super().format(record)
        
        # Add structured data if present
        if hasattr(record, 'extra_data') and record.extra_data:
            # Append as JSON for machine parsing while keeping human readability
            msg += f" | {json.dumps(record.extra_data, separators=(',', ':'))}"
        
        return msg


class RichConsoleHandler(logging.Handler):
    """Handler that uses Rich for pretty console output."""
    
    def __init__(self, console: Optional[Console] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console = console or Console(stderr=True)
        
    def emit(self, record):
        try:
            msg = self.format(record)
            
            # Color based on level
            style = {
                logging.DEBUG: "dim",
                logging.INFO: "default",
                logging.WARNING: "yellow",
                logging.ERROR: "red bold",
                logging.CRITICAL: "red bold reverse",
            }.get(record.levelno, "default")
            
            self.console.print(msg, style=style)
        except Exception:
            self.handleError(record)


def setup_logging(
    level: str = "INFO",
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
    use_rich: bool = True,
    log_file: Optional[Path] = None,
) -> None:
    """
    Setup logging configuration for the verifiers package.
    
    Args:
        level: The logging level to use. Defaults to "INFO".
        log_format: Custom log format string. If None, uses default format.
        date_format: Custom date format string. If None, uses default format.
        use_rich: Whether to use rich console output. Defaults to True.
        log_file: Optional file path to write logs to.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Get the root logger for the verifiers package
    logger = logging.getLogger("verifiers")
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False
    
    # Console handler
    if use_rich:
        console_handler = RichConsoleHandler()
        console_handler.setFormatter(logging.Formatter(fmt="%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            StructuredFormatter(fmt=log_format, datefmt=date_format)
        )
    
    console_handler.setLevel(level.upper())
    logger.addHandler(console_handler)
    
    # File handler if requested
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            StructuredFormatter(fmt=log_format, datefmt=date_format)
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with consistent configuration.
    
    Args:
        name: The name of the logger (usually __name__)
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Add method to log with structured data
    def log_structured(level, msg, **kwargs):
        extra = {'extra_data': kwargs} if kwargs else {}
        logger.log(level, msg, extra=extra)
    
    logger.debug_structured = lambda msg, **kw: log_structured(logging.DEBUG, msg, **kw)
    logger.info_structured = lambda msg, **kw: log_structured(logging.INFO, msg, **kw)
    logger.warning_structured = lambda msg, **kw: log_structured(logging.WARNING, msg, **kw)
    logger.error_structured = lambda msg, **kw: log_structured(logging.ERROR, msg, **kw)
    
    return logger


def print_prompt_completions_sample(
    prompts: List[Union[str, List[Dict[str, Any]]]],
    completions: List[Union[str, Dict[str, Any], List[Dict[str, Any]]]],
    rewards: Dict[str, List[float]],
    step: int,
    num_samples: int = 1,
    console: Optional[Console] = None,
) -> None:
    """
    Print a formatted sample of prompts, completions, and rewards.
    
    Args:
        prompts: List of prompts (strings or chat format)
        completions: List of completions 
        rewards: Dictionary of reward values
        step: Current training step
        num_samples: Number of samples to display
        console: Rich console instance (creates new if None)
    """
    if console is None:
        console = Console()
    
    table = Table(show_header=True, header_style="bold white", expand=True)
    
    # Add columns
    table.add_column("Prompt", style="bright_yellow", overflow="fold")
    table.add_column("Completion", style="bright_green", overflow="fold")
    
    # Add reward columns dynamically
    reward_keys = sorted(rewards.keys())
    for key in reward_keys:
        table.add_column(key.replace("_", " ").title(), style="bold cyan", justify="right")
    
    # Get samples to show
    samples_to_show = min(num_samples, len(prompts))
    
    for i in range(samples_to_show):
        prompt = prompts[i]
        completion = completions[i]
        
        # Format prompt
        formatted_prompt = _format_prompt(prompt)
        
        # Format completion
        formatted_completion = _format_completion(completion)
        
        # Get reward values for this sample
        row_values = [formatted_prompt, formatted_completion]
        for key in reward_keys:
            values = rewards.get(key, [])
            if i < len(values):
                row_values.append(Text(f"{values[i]:.3f}"))
            else:
                row_values.append(Text("N/A"))
        
        table.add_row(*row_values)
        
        if i < samples_to_show - 1:
            table.add_section()
    
    panel = Panel(
        table, 
        expand=False, 
        title=f"[bold]Step {step}[/bold]", 
        border_style="bold white"
    )
    console.print(panel)


def _format_prompt(prompt: Union[str, List[Dict[str, Any]]]) -> Text:
    """Format a prompt for display."""
    formatted_prompt = Text()
    
    if isinstance(prompt, str):
        formatted_prompt = Text(prompt)
    elif isinstance(prompt, list):
        # For chat format, show the last message content
        if prompt:
            last_message = prompt[-1]
            content = last_message.get("content", "")
            if isinstance(content, list):  # multimodal case
                # Extract text content
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content = item.get("text", "")
                        break
            formatted_prompt = Text(content, style="bright_yellow")
        else:
            formatted_prompt = Text("")
    else:
        formatted_prompt = Text(str(prompt))
    
    return formatted_prompt


def _format_completion(completion: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> Text:
    """Format a completion for display."""
    formatted_completion = Text()
    
    if isinstance(completion, str):
        formatted_completion = Text(completion)
    elif isinstance(completion, dict):
        # Handle single message dict
        role = completion.get("role", "")
        content = completion.get("content", "")
        style = "bright_cyan" if role == "assistant" else "bright_magenta"
        formatted_completion.append(f"{role}: ", style="bold")
        formatted_completion.append(content, style=style)
    elif isinstance(completion, list):
        # Handle list of message dicts
        for i, message in enumerate(completion):
            if i > 0:
                formatted_completion.append("\n\n")
            
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Set style based on role
            style = "bright_cyan" if role == "assistant" else "bright_magenta"
            
            formatted_completion.append(f"{role}: ", style="bold")
            formatted_completion.append(content, style=style)
    else:
        # Fallback
        formatted_completion = Text(str(completion))
    
    return formatted_completion


class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.DEBUG):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.logger.log(
            self.level, 
            f"{self.operation} took {duration:.3f}s",
            extra={'extra_data': {'operation': self.operation, 'duration': duration}}
        )


def log_generation_batch(
    logger: logging.Logger,
    batch_id: int,
    prompts: List[Any],
    completions: List[Any],
    rewards: Dict[str, List[float]],
    generation_time: float,
    **metadata
):
    """
    Log information about a generation batch in a structured way.
    
    Args:
        logger: Logger instance
        batch_id: Batch identifier
        prompts: List of prompts
        completions: List of completions
        rewards: Dictionary of reward scores
        generation_time: Time taken to generate
        **metadata: Additional metadata to log
    """
    # Calculate statistics
    avg_reward = sum(rewards.get("reward", [])) / len(rewards.get("reward", [1]))
    completion_lengths = []
    
    for completion in completions:
        if isinstance(completion, str):
            completion_lengths.append(len(completion.split()))
        elif isinstance(completion, list):
            # For chat format, sum content lengths
            length = sum(len(msg.get("content", "").split()) for msg in completion)
            completion_lengths.append(length)
    
    avg_length = sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0
    
    logger.info_structured(
        f"Batch {batch_id}: {len(prompts)} samples, "
        f"avg_reward={avg_reward:.3f}, avg_length={avg_length:.1f} tokens, "
        f"gen_time={generation_time:.2f}s",
        batch_id=batch_id,
        num_samples=len(prompts),
        avg_reward=avg_reward,
        avg_completion_length=avg_length,
        generation_time=generation_time,
        **metadata
    )