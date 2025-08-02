"""Utilities for handling processors and tokenizers uniformly."""

from typing import Any, Dict, List, Optional, Union

from transformers import PreTrainedTokenizerBase, ProcessorMixin


class ProcessorWrapper:
    """Unified interface for tokenizers and multimodal processors.

    This wrapper provides a consistent interface regardless of whether
    the underlying object is a tokenizer or a multimodal processor.
    """

    def __init__(
        self, processing_class: Union[PreTrainedTokenizerBase, ProcessorMixin]
    ):
        """Initialize the wrapper.

        Args:
            processing_class: Either a tokenizer or a processor
        """
        self.processing_class = processing_class

        # Determine if this is a multimodal processor or just a tokenizer
        if isinstance(processing_class, ProcessorMixin):
            self.tokenizer = processing_class.tokenizer
            self.processor = processing_class
            self.is_multimodal = True
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self.tokenizer = processing_class
            self.processor = None
            self.is_multimodal = False
        else:
            raise TypeError(
                f"processing_class must be either PreTrainedTokenizerBase or ProcessorMixin, "
                f"got {type(processing_class)}"
            )

    @property
    def pad_token(self) -> Optional[str]:
        """Get the padding token."""
        return self.tokenizer.pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        """Set the padding token."""
        self.tokenizer.pad_token = value

    @property
    def pad_token_id(self) -> Optional[int]:
        """Get the padding token ID."""
        return self.tokenizer.pad_token_id

    @property
    def eos_token(self) -> Optional[str]:
        """Get the end-of-sequence token."""
        return self.tokenizer.eos_token

    @property
    def eos_token_id(self) -> Optional[int]:
        """Get the end-of-sequence token ID."""
        return self.tokenizer.eos_token_id

    def ensure_pad_token(self) -> None:
        """Ensure a pad token is set, using eos_token as fallback."""
        if self.pad_token is None:
            self.pad_token = self.eos_token

    def apply_chat_template(
        self,
        messages: List[Dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Apply chat template to messages.

        Args:
            messages: List of message dictionaries
            tokenize: Whether to tokenize the output
            add_generation_prompt: Whether to add generation prompt
            **kwargs: Additional arguments for the tokenizer

        Returns:
            Templated string or token IDs
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Text to encode
            **kwargs: Additional arguments for encoding

        Returns:
            List of token IDs
        """
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            **kwargs: Additional arguments for decoding

        Returns:
            Decoded text
        """
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, sequences: List[List[int]], **kwargs) -> List[str]:
        """Decode multiple sequences of token IDs.

        Args:
            sequences: List of token ID sequences
            **kwargs: Additional arguments for decoding

        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(sequences, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """Forward calls to the underlying processing class.

        This allows the wrapper to be used exactly like the original
        tokenizer or processor.
        """
        return self.processing_class(*args, **kwargs)

    def get_base_processing_class(
        self,
    ) -> Union[PreTrainedTokenizerBase, ProcessorMixin]:
        """Get the underlying processing class.

        Returns:
            The wrapped tokenizer or processor
        """
        return self.processing_class
