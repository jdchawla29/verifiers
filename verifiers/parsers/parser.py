import logging
from typing import Any, List, Dict, Callable

from verifiers import (
    ChatMessage,
    Messages,
)

class Parser:
    """
    Parser class for parsing LLM rollouts.

    Default behavior:
    - `parse` returns text as-is
    - `get_final_answer` returns the last message's content (or text if string)
    """

    def __init__(self, **kwargs):
        self.logger = logging.getLogger(f"verifiers.parsers.{self.__class__.__name__}")
        for key, value in kwargs.items():
            setattr(self, key, value)

    def parse(self, text: str) -> Any:
        self.logger.debug(f"Parsing text (length: {len(text)}): {text[:100]}..." if len(text) > 100 else f"Parsing text: {text}")
        return text
    
    def get_assistant_messages(self, completion: List[ChatMessage]) -> List[ChatMessage]:
        """Helper function to extract assistant messages from a completion."""
        messages = [msg for msg in completion if msg['role'] == 'assistant']
        self.logger.debug(f"Extracted {len(messages)} assistant messages from {len(completion)} total messages")
        return messages

    def get_system_messages(self, completion: List[ChatMessage]) -> List[ChatMessage]:
        """Helper function to extract system messages from a completion."""
        return [msg for msg in completion if msg['role'] == 'system']

    def get_user_messages(self, completion: List[ChatMessage]) -> List[ChatMessage]:
        """Helper function to extract user messages from a completion."""
        return [msg for msg in completion if msg['role'] == 'user']

    def get_tool_messages(self, completion: List[ChatMessage]) -> List[ChatMessage]:
        """Helper function to extract tool messages from a completion."""
        return [msg for msg in completion if msg['role'] == 'tool']
    
    def parse_answer(self, completion: Messages) -> str | None:
        self.logger.debug(f"Parsing answer from completion: {type(completion)}")
        if isinstance(completion, str):
            result = self.parse(completion)
            self.logger.debug(f"Parsed answer from string: {result}")
            return result
        else:
            self.logger.debug(f"Parsing answer from last message of {len(completion)} messages")
            result = self.parse(completion[-1]["content"])
            self.logger.debug(f"Parsed answer: {result}")
            return result
 
    def get_format_reward_func(self) -> Callable:
        """
        Reward function that checks if the final answer is formatted correctly.
        """
        def format_reward_func(completion: List[Dict[str, str]], **kwargs) -> float:
            return 1.0
        return format_reward_func
    

    