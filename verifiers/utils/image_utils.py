"""Utilities for handling image data in verifiers."""

import base64
import io
from typing import Any

from PIL import Image


def pil_to_data_url(img: Image.Image, fmt: str | None = None) -> str:
    """Convert PIL Image to data URL for multimodal inputs.

    Args:
        img: PIL Image to convert
        fmt: Image format (defaults to image's format or PNG)

    Returns:
        Data URL string with base64-encoded image
    """
    buf = io.BytesIO()
    fmt = (fmt or img.format or "PNG").upper()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def format_openai_messages(
    prompts: list[list[Any]], images: list[list[Image.Image]]
) -> list[Any]:
    """Format multimodal chat messages for OpenAI API.

    Args:
        prompts: List of conversations, each with messages containing potential image placeholders
        images: List of image lists corresponding to each conversation

    Returns:
        Formatted conversations with images converted to data URLs
    """
    formatted_conversations = []

    for conv_prompts, conv_images in zip(prompts, images):
        img_iter = iter(conv_images)
        new_conv = []

        for msg in conv_prompts:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, list):
                new_parts = []
                for part in content:
                    # Check for placeholder URL pattern
                    if (part.get("type") == "image_url" and 
                        part.get("image_url", {}).get("url", "").startswith("placeholder://")):
                        img = next(img_iter)
                        data_url = pil_to_data_url(img)
                        new_parts.append(
                            {"type": "image_url", "image_url": {"url": data_url}}
                        )
                    else:
                        new_parts.append(part.copy())
                new_conv.append({"role": role, "content": new_parts})
            else:
                new_conv.append({"role": role, "content": content})

        formatted_conversations.append(new_conv)

    return formatted_conversations


def extract_text_from_multimodal_content(content: Any) -> str:
    """Extract text from multimodal message content.

    Args:
        content: Message content (string or list of content parts)

    Returns:
        Extracted text content
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    text_parts.append("[IMAGE]")
        
        return " ".join(text_parts) if text_parts else ""

    return ""


def has_images(batch: list[dict[str, Any]]) -> bool:
    """Check if a batch contains images.

    Args:
        batch: List of batch items

    Returns:
        True if any item has images
    """
    return any("images" in item for item in batch)


def extract_images_from_batch(
    batch: list[dict[str, Any]],
) -> list[list[Image.Image]] | None:
    """Extract images from a batch if present.

    Args:
        batch: List of batch items

    Returns:
        List of image lists or None if no images
    """
    if not has_images(batch):
        return None
    return [item.get("images", []) for item in batch]
