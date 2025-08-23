"""Tests for multimodal support in verifiers."""

import pytest
from PIL import Image
import base64
from datasets import Dataset

from verifiers import SingleTurnEnv, Parser, Rubric
from verifiers.utils.image_utils import (
    pil_to_data_url,
    format_openai_messages,
    extract_text_from_multimodal_content,
    extract_images_from_batch,
)


class TestMultimodalUtils:
    """Test multimodal utility functions."""

    def test_pil_to_data_url(self):
        """Test PIL image to data URL conversion."""
        # Create a small test image
        img = Image.new("RGB", (10, 10), color="red")

        # Convert to data URL
        data_url = pil_to_data_url(img)

        # Check format
        assert data_url.startswith("data:image/png;base64,")

        # Decode and verify it's valid base64
        base64_part = data_url.split(",")[1]
        decoded = base64.b64decode(base64_part)
        assert len(decoded) > 0

        # Test with specific format
        data_url_jpeg = pil_to_data_url(img, fmt="JPEG")
        assert data_url_jpeg.startswith("data:image/jpeg;base64,")

    def test_format_openai_messages(self):
        """Test formatting OpenAI-style chat messages with images."""
        # Create test data with multimodal content
        prompts = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": "placeholder://image"}},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this picture"},
                        {"type": "image_url", "image_url": {"url": "placeholder://image"}},
                    ],
                }
            ],
        ]

        # Create small test images
        img1 = Image.new("RGB", (10, 10), color="blue")
        img2 = Image.new("RGB", (10, 10), color="green")
        images = [[img1], [img2]]

        # Format messages
        formatted = format_openai_messages(prompts, images)

        # Check structure
        assert len(formatted) == 2

        # Check first message
        assert formatted[0][0]["role"] == "user"
        assert isinstance(formatted[0][0]["content"], list)
        assert len(formatted[0][0]["content"]) == 2  # text + image
        assert formatted[0][0]["content"][0]["type"] == "text"
        assert formatted[0][0]["content"][0]["text"] == "What is in this image?"
        assert formatted[0][0]["content"][1]["type"] == "image_url"
        assert formatted[0][0]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )

    def test_format_openai_messages_text_only(self):
        """Test formatting text-only messages (no image placeholders)."""
        prompts = [
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "World"}],
        ]

        # Empty images lists (not None, but empty lists for each conversation)
        images = [[], []]

        # Should return prompts unchanged since no images to format
        formatted = format_openai_messages(prompts, images)
        assert formatted == prompts

    def test_format_openai_messages_edge_cases(self):
        """Test edge cases for format_openai_messages."""
        # Test with empty images list for conversation with placeholders
        prompts = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "test"}, {"type": "image_url", "image_url": {"url": "placeholder://image"}}],
                }
            ]
        ]
        # Empty images list will cause StopIteration when trying to get image
        with pytest.raises(StopIteration):
            format_openai_messages(prompts, [[]])

        # Test with mismatched number of images and placeholders
        img = Image.new("RGB", (10, 10), color="red")

        # More placeholders than images - will raise StopIteration
        prompts_multi = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "test"},
                        {"type": "image_url", "image_url": {"url": "placeholder://image"}},
                        {"type": "image_url", "image_url": {"url": "placeholder://image"}},  # Two placeholders
                    ],
                }
            ]
        ]
        images_single = [[img]]  # Only one image

        # Should raise StopIteration when trying to get second image
        with pytest.raises(StopIteration):
            format_openai_messages(prompts_multi, images_single)

        # More images than placeholders - extra images ignored
        prompts_single = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "test"},
                        {"type": "image_url", "image_url": {"url": "placeholder://image"}},  # One placeholder
                    ],
                }
            ]
        ]
        images_multi = [[img, img]]  # Two images

        # Should only use first image, second is ignored
        formatted = format_openai_messages(prompts_single, images_multi)
        assert len(formatted[0][0]["content"]) == 2
        assert formatted[0][0]["content"][1]["type"] == "image_url"
        assert formatted[0][0]["content"][1]["image_url"]["url"].startswith(
            "data:image/"
        )

    def test_extract_text_from_multimodal_content(self):
        """Test text extraction from multimodal content."""
        # Simple string content
        assert extract_text_from_multimodal_content("Hello") == "Hello"

        # List with mixed content
        content = [
            {"type": "text", "text": "Part 1"},
            {"type": "image_url", "image_url": {"url": "placeholder://image"}},
            {"type": "text", "text": "Part 2"},
        ]
        assert extract_text_from_multimodal_content(content) == "Part 1 [IMAGE] Part 2"

        # Text item not first
        content_text_second = [
            {"type": "image_url", "image_url": {"url": "placeholder://image"}},
            {"type": "text", "text": "Found text"},
        ]
        assert extract_text_from_multimodal_content(content_text_second) == "[IMAGE] Found text"

        # Empty content
        assert extract_text_from_multimodal_content([]) == ""

        # Content without text
        content_no_text = [{"type": "image_url", "image_url": {"url": "placeholder://image"}}, {"type": "other"}]
        assert extract_text_from_multimodal_content(content_no_text) == "[IMAGE]"

        # None content
        assert extract_text_from_multimodal_content(None) == ""

    def test_pil_to_data_url_edge_cases(self):
        """Test PIL to data URL conversion edge cases."""
        # Test with different image modes
        img_rgba = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        data_url = pil_to_data_url(img_rgba)
        assert data_url.startswith("data:image/png;base64,")

        # Test with grayscale
        img_gray = Image.new("L", (10, 10), color=128)
        data_url = pil_to_data_url(img_gray)
        assert data_url.startswith("data:image/png;base64,")

        # Test with very small image
        img_tiny = Image.new("RGB", (1, 1), color="white")
        data_url = pil_to_data_url(img_tiny)
        assert data_url.startswith("data:image/png;base64,")

        # Test format parameter
        img = Image.new("RGB", (10, 10), color="green")
        data_url_jpeg = pil_to_data_url(img, fmt="JPEG")
        data_url_png = pil_to_data_url(img, fmt="PNG")
        assert data_url_jpeg.startswith("data:image/jpeg;base64,")
        assert data_url_png.startswith("data:image/png;base64,")
        assert (
            data_url_jpeg != data_url_png
        )  # Different formats should produce different results


class TestMultimodalEnvironment:
    """Test multimodal support in Environment class."""

    def test_process_chat_format_vllm_with_images(self, mock_processor):
        """Test processing chat format with images using vLLM methods."""
        # Create test environment
        dataset = Dataset.from_dict(
            {
                "prompt": [{"role": "user", "content": "test"}],
                "answer": ["test"],
                "images": [[Image.new("RGB", (10, 10), color="red")]],
            }
        )

        env = SingleTurnEnv(dataset=dataset, parser=Parser(), rubric=Rubric())

        prompt = [{"role": "user", "content": "What is this?"}]
        images = [Image.new("RGB", (10, 10), color="blue")]
        completion = [{"role": "assistant", "content": "A blue square"}]

        # Create mock vLLM response with tokens/logprobs
        from unittest.mock import Mock

        token_entries = [
            Mock(logprob=-0.1, token="token_id:1"),
            Mock(logprob=-0.2, token="token_id:2"),
        ]
        mock_choice = Mock()
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.content = token_entries
        mock_chat_completion = Mock()
        mock_chat_completion.choices = [mock_choice]
        state = {"responses": [mock_chat_completion]}

        # Process with images using vLLM method
        (
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            completion_logprobs,
            remaining_inputs,
        ) = env.process_chat_format_vllm(
            prompt,
            completion,
            state,
            mock_processor,
            mask_env_responses=False,
            images=images,
        )

        # Check outputs
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)
        assert isinstance(completion_logprobs, list)
        assert len(completion_logprobs) == len(completion_ids)
        # With images, the processor should have been called with images
        assert mock_processor.called
        # Check remaining_inputs contains pixel_values
        assert isinstance(remaining_inputs, dict)
        assert "pixel_values" in remaining_inputs

    def test_process_env_results_vllm_multimodal(self, mock_processor):
        """Test processing environment results with multimodal data using vLLM methods."""
        # Create test environment
        dataset = Dataset.from_dict(
            {"prompt": [{"role": "user", "content": "test"}], "answer": ["test"]}
        )

        env = SingleTurnEnv(dataset=dataset, parser=Parser(), rubric=Rubric())

        # Create test data
        prompts = [[{"role": "user", "content": "What is this?"}]]
        images = [[Image.new("RGB", (10, 10), color="green")]]
        completions = [[{"role": "assistant", "content": "A green square"}]]

        # Create mock vLLM response with tokens/logprobs
        from unittest.mock import Mock

        token_entries = [
            Mock(logprob=-0.1, token="token_id:1"),
            Mock(logprob=-0.2, token="token_id:2"),
            Mock(logprob=-0.3, token="token_id:3"),
        ]
        mock_choice = Mock()
        mock_choice.logprobs = Mock()
        mock_choice.logprobs.content = token_entries
        mock_chat_completion = Mock()
        mock_chat_completion.choices = [mock_choice]
        states = [{"responses": [mock_chat_completion]}]
        rewards = [1.0]

        # Process results using vLLM method
        results = env.process_env_results_vllm(
            prompts, completions, states, rewards, mock_processor, images=images
        )

        # Check outputs
        assert hasattr(results, "prompt_ids")
        assert hasattr(results, "completion_ids")
        assert hasattr(results, "completion_logprobs")
        assert hasattr(results, "rewards")
        assert len(results.prompt_ids) == 1
        assert len(results.completion_ids) == 1
        assert len(results.completion_logprobs) == 1
        assert results.rewards[0] == 1.0
        # With images, the processor should have been called
        assert mock_processor.called
        assert hasattr(results, "remaining_inputs")
        assert len(results.remaining_inputs) == 1
        # The processor returns pixel_values, not images
        assert "pixel_values" in results.remaining_inputs[0]


# Skip trainer tests for now - the existing test suite doesn't test trainers
# class TestMultimodalGRPOTrainer:
#     """Test multimodal support in GRPO trainer."""
#     # Trainer tests would go here if needed


class TestMultimodalIntegration:
    """Integration tests for multimodal workflows.

    The multimodal flow in verifiers works as follows:
    1. Data collator transforms prompts to have content as a list with text and image placeholders
    2. Images are stored separately in the "images" column
    3. When a_generate is called with a dataset containing images, format_openai_messages is invoked
    4. format_openai_messages replaces image placeholders with base64-encoded data URLs
    5. The formatted messages are sent to the API with proper multimodal structure
    """

    def test_data_collator_workflow(self):
        """Test a data collator workflow similar to docvqa example."""

        def data_collator(batch):
            # Handle batched format from dataset.map()
            prompts = []
            images = []

            for i in range(len(batch["question"])):
                # Create multimodal prompt format with image placeholders
                content_block = []
                content_block.append({"type": "text", "text": batch["question"][i]})
                content_block.append({"type": "image_url", "image_url": {"url": "placeholder://image"}})  # Image placeholder

                # Format the prompt with multimodal content
                prompts.append([{"role": "user", "content": content_block}])

                # Add the actual images
                images.append([Image.new("RGB", (10, 10), color="red")])

            # Return updated batch dict
            result = dict(batch)
            result["prompt"] = prompts
            result["images"] = images
            return result

        # Create datasets
        train_dataset = Dataset.from_dict(
            {
                "question": ["What is this?", "Describe the image"],
                "answer": [["red square"], ["a red square"]],
            }
        )

        eval_dataset = Dataset.from_dict(
            {
                "question": ["What color is it?", "What shape is it?"],
                "answer": [["red"], ["square"]],
            }
        )

        # Create environment with data collator on eval dataset
        env = SingleTurnEnv(
            dataset=train_dataset,  # This will be formatted but not data-collated
            eval_dataset=eval_dataset,  # This will have data_collator applied
            parser=Parser(),
            rubric=Rubric(),
            data_collator=data_collator,
        )

        # Check that data collator was applied to the eval_dataset
        # Since set_transform is used, the column won't appear in column_names
        # but will be present when we access the data
        assert len(env.eval_dataset) == 2

        # Access a sample to trigger the transform
        sample = env.eval_dataset[0]
        assert "images" in sample
        assert len(sample["images"]) == 1
        assert isinstance(sample["images"][0], Image.Image)

    @pytest.mark.asyncio
    async def test_multimodal_generation_flow(self, mock_openai_client):
        """Test the full multimodal generation flow with data collator."""
        # Configure mock client response
        mock_openai_client.set_response("It's a red square")

        # Create a data collator that formats prompts
        def collator(batch):
            # Handle batched format from dataset.map()
            prompts = []
            images = []

            for i in range(len(batch["question"])):
                # Create multimodal content structure
                content = [
                    {"type": "text", "text": batch["question"][i]},
                    {"type": "image_url", "image_url": {"url": "placeholder://image"}},  # Placeholder for image
                ]
                prompts.append([{"role": "user", "content": content}])
                # Create a simple PIL image
                img = Image.new("RGB", (10, 10), color="red")
                images.append([img])

            # Return updated batch dict
            result = dict(batch)
            result["prompt"] = prompts
            result["images"] = images
            return result

        # Create dataset
        base_dataset = Dataset.from_dict(
            {"question": ["What color is this?"], "answer": ["red"]}
        )

        # Create environment with data collator
        env = SingleTurnEnv(
            dataset=base_dataset,
            eval_dataset=base_dataset,  # data_collator applies to eval_dataset
            parser=Parser(),
            rubric=Rubric(),
            data_collator=collator,
        )

        # Run generation on eval dataset (which has data_collator applied)
        test_input = env.get_eval_dataset(n=1)

        results = await env.a_generate(
            test_input, client=mock_openai_client, model="test-model"
        )

        # Check results
        assert hasattr(results, "completion")
        assert len(results.completion) == 1
        assert results.completion[0][0]["content"] == "It's a red square"

        # Verify that the client was called
        mock_openai_client.chat.completions.create.assert_called_once()
        call_args = mock_openai_client.chat.completions.create.call_args

        # Get the messages argument (it's the first positional arg)
        messages = call_args[0][0] if call_args[0] else call_args[1]["messages"]

        # Check that messages were formatted for multimodal if images present
        assert len(messages) > 0

        # Find the user message (might not be the last if there's a system prompt)
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        assert len(user_messages) > 0
        last_user_msg = user_messages[-1]

        # With the data collator, prompts should have multimodal format
        # and when images are present, format_openai_messages should be called
        if "images" in test_input and test_input["images"]:
            # Check if the content is properly formatted as multimodal
            assert isinstance(last_user_msg["content"], list), (
                "Content should be a list for multimodal"
            )
            content_types = {item["type"] for item in last_user_msg["content"]}
            assert "text" in content_types, "Should have text content"
            assert "image_url" in content_types, "Should have image content"

            # Verify image was properly encoded as base64 data URL
            image_items = [
                item for item in last_user_msg["content"] if item["type"] == "image_url"
            ]
            assert len(image_items) > 0
            assert image_items[0]["image_url"]["url"].startswith("data:image/")
