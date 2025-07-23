from typing import List, Callable
import logging

from verifiers import (
    ChatMessage,
    Parser,
)


class ThinkParser(Parser):
    def __init__(self,
                 extract_fn: Callable[[str], str] = lambda x: x,
                 **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(f"verifiers.parsers.{self.__class__.__name__}")
        self.extract_fn = extract_fn
        self.logger.debug(f"Initialized ThinkParser with extract_fn: {extract_fn}")

    def parse(self, text: str) -> str:
        self.logger.debug(f"Parsing text (length: {len(text)}): {text[:100]}..." if len(text) > 100 else f"Parsing text: {text}")
        if "</think>" in text:
            self.logger.debug("Found </think> tag, extracting content after it")
            text = text.split("</think>")[-1].strip()
            self.logger.debug(f"Extracted text after </think>: {text[:100]}..." if len(text) > 100 else f"Extracted text: {text}")
        result = self.extract_fn(text.strip())
        self.logger.debug(f"Final parsed result: {result[:100]}..." if len(result) > 100 else f"Final parsed result: {result}")
        return result

    def get_format_reward_func(self) -> Callable:
        """
        Return a reward function that checks if each message follows the format:
        <think>
        ...
        </think>
        ...
        """
        def follows_format(text: str) -> float:
            stripped_text = text.strip()
            starts_with_think = stripped_text.startswith("<think>")
            think_count = text.count("<think>")
            close_think_count = text.count("</think>")
            has_content_after = len(text.split("</think>")[-1]) > 0 if "</think>" in text else False
            
            self.logger.debug(
                f"Format check - starts_with_think: {starts_with_think}, "
                f"think_count: {think_count}, close_think_count: {close_think_count}, "
                f"has_content_after: {has_content_after}"
            )
            
            if (
                starts_with_think and 
                think_count == 1 and
                close_think_count == 1 and
                has_content_after
            ):
                return 1.0
            return 0.0

        def format_reward_func(completion: List[ChatMessage], **kwargs) -> float:
            messages = self.get_assistant_messages(completion)
            self.logger.debug(f"Calculating format reward for {len(messages)} messages")
            scores = [follows_format(m["content"]) for m in messages]
            avg_score = sum(scores) / len(messages) if messages else 0.0
            self.logger.debug(f"Format scores: {scores}, average: {avg_score}")
            return avg_score
        return format_reward_func