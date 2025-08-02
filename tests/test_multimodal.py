"""Tests for multimodal support in verifiers."""

import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from PIL import Image
import base64
from datasets import Dataset

from verifiers import SingleTurnEnv, Parser, Rubric
from verifiers.utils.multimodal_utils import MultimodalHandler
from verifiers.trainers.async_batch_generator import AsyncBatchGenerator, BatchRequest


class MockProcessor:
    """Mock processor for multimodal tests that mimics AutoProcessor behavior."""

    def __init__(self):
        # Use MagicMock for tokenizer so it auto-creates attributes like pad_token
        self.tokenizer = MagicMock()
        self.tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        self.tokenizer.apply_chat_template.side_effect = (
            lambda messages, **kwargs: " ".join(
                [f"{msg['role']}: {msg['content']}" for msg in messages]
            )
        )
        # Set pad_token to None to match expected behavior
        self.tokenizer.pad_token = None

        # Processor should have chat_template
        self.chat_template = "test_template"

    def apply_chat_template(self, messages, **kwargs):
        """Apply chat template - processors have this method directly."""
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        """Process text and images like a real processor."""

        # Mock tensor class
        class MockTensor:
            def __init__(self, data):
                self._data = data

            def tolist(self):
                return self._data

            def __getitem__(self, idx):
                return self

        # Create a dict-like object that also supports attribute access
        class ProcessorOutput(dict):
            def __getattr__(self, key):
                return self[key]

            def __setattr__(self, key, value):
                self[key] = value

        # Return processor output with tensor values
        result = ProcessorOutput()
        if text:
            result["input_ids"] = [MockTensor(list(range(len(text.split()))))]
        if images:
            result["pixel_values"] = MockTensor([[1, 2, 3]])

        return result

    def batch_decode(self, token_ids, **kwargs):
        """Decode token ids to text."""
        return [f"decoded_{i}" for i in range(len(token_ids))]


class TestMultimodalUtils:
    """Test multimodal utility functions."""

    def test_pil_to_data_url(self):
        """Test PIL image to data URL conversion."""
        # Create a small test image
        img = Image.new("RGB", (10, 10), color="red")

        # Convert to data URL
        data_url = MultimodalHandler.pil_to_data_url(img)

        # Check format
        assert data_url.startswith("data:image/png;base64,")

        # Decode and verify it's valid base64
        base64_part = data_url.split(",")[1]
        decoded = base64.b64decode(base64_part)
        assert len(decoded) > 0

        # Test with specific format
        data_url_jpeg = MultimodalHandler.pil_to_data_url(img, fmt="JPEG")
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
                        {"type": "image"},
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this picture"},
                        {"type": "image"},
                    ],
                }
            ],
        ]

        # Create small test images
        img1 = Image.new("RGB", (10, 10), color="blue")
        img2 = Image.new("RGB", (10, 10), color="green")
        images = [[img1], [img2]]

        # Format messages
        formatted = MultimodalHandler.format_openai_messages(prompts, images)

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
        formatted = MultimodalHandler.format_openai_messages(prompts, images)
        assert formatted == prompts


class TestMultimodalEnvironment:
    """Test multimodal support in Environment class."""

    def test_process_chat_format_with_images(self):
        """Test processing chat format with images."""
        # Create test environment
        dataset = Dataset.from_dict(
            {
                "prompt": [{"role": "user", "content": "test"}],
                "answer": ["test"],
                "images": [[Image.new("RGB", (10, 10), color="red")]],
            }
        )

        env = SingleTurnEnv(dataset=dataset, parser=Parser(), rubric=Rubric())

        # Create mock processor
        mock_processor = MockProcessor()

        prompt = [{"role": "user", "content": "What is this?"}]
        images = [Image.new("RGB", (10, 10), color="blue")]
        completion = [{"role": "assistant", "content": "A blue square"}]

        # Process with images
        prompt_ids, prompt_mask, completion_ids, completion_mask, remaining_inputs = (
            env.process_chat_format(
                prompt, images, completion, mock_processor, mask_env_responses=False
            )
        )

        # Check outputs
        assert isinstance(prompt_ids, list)
        assert isinstance(completion_ids, list)
        assert remaining_inputs is not None
        # The processor returns pixel_values, not images
        assert "pixel_values" in remaining_inputs
        assert remaining_inputs["pixel_values"] is not None

    def test_process_env_results_multimodal(self):
        """Test processing environment results with multimodal data."""
        # Create test environment
        dataset = Dataset.from_dict(
            {"prompt": [{"role": "user", "content": "test"}], "answer": ["test"]}
        )

        env = SingleTurnEnv(dataset=dataset, parser=Parser(), rubric=Rubric())

        # Create mock processor
        mock_processor = MockProcessor()

        # Create test data
        prompts = [[{"role": "user", "content": "What is this?"}]]
        images = [[Image.new("RGB", (10, 10), color="green")]]
        completions = [[{"role": "assistant", "content": "A green square"}]]
        states = [{}]
        rewards = [1.0]

        # Process results
        results = env.process_env_results(
            prompts, images, completions, states, rewards, mock_processor
        )

        # Check outputs
        assert hasattr(results, "prompt_ids")
        assert hasattr(results, "completion_ids")
        assert hasattr(results, "remaining_inputs")
        assert len(results.remaining_inputs) == 1
        # The processor returns pixel_values, not images
        assert "pixel_values" in results.remaining_inputs[0]


# Skip trainer tests for now - the existing test suite doesn't test trainers
# class TestMultimodalGRPOTrainer:
#     """Test multimodal support in GRPO trainer."""
#     # Trainer tests would go here if needed


class TestAsyncBatchGeneratorMultimodal:
    """Test multimodal support in async batch generator."""

    @pytest.fixture
    def mock_env_multimodal(self):
        """Create a mock environment with multimodal support."""
        from verifiers.types import GenerateOutputs

        env = Mock()
        env.a_generate = AsyncMock(
            return_value=GenerateOutputs(
                prompt=[[{"role": "user", "content": "test"}]],
                answer=[""],
                task=["default"],
                info=[{}],
                completion=[[{"role": "assistant", "content": "response"}]],
                state=[{}],
                reward=[1.0],
                metrics={},
            )
        )
        from verifiers.types import ProcessedOutputs

        processed_output = ProcessedOutputs(
            prompt_ids=[[1, 2, 3]],
            prompt_mask=[[0, 0, 0]],
            completion_ids=[[4, 5, 6]],
            completion_mask=[[1, 1, 1]],
            completion_logprobs=[[0.0, 0.0, 0.0]],
            rewards=[1.0],
            remaining_inputs=[{"images": [Image.new("RGB", (10, 10), color="red")]}],
        )
        env.process_env_results = Mock(return_value=processed_output)
        env.process_env_results_vllm = Mock(return_value=processed_output)
        return env

    @pytest.mark.asyncio
    async def test_generate_batch_with_images(self, mock_env_multimodal):
        """Test async batch generation with images."""
        client_config = {
            "base_url": "http://test",
            "api_key": "test",
            "http_client_args": {"limits": {"max_connections": 10}, "timeout": 30.0},
        }

        generator = AsyncBatchGenerator(
            env=mock_env_multimodal,
            client_config=client_config,
            model_name="test-model",
            sampling_args={},
        )

        # Create batch request
        request = BatchRequest(
            batch_id=0,
            env_inputs={"prompt": [[{"role": "user", "content": "test"}]]},
            processing_class=MockProcessor(),
            mask_env_responses=False,
            max_seq_len=100,
            mask_truncated_completions=False,
            zero_truncated_completions=False,
            max_concurrent=10,
        )

        # Generate batch (mocked)
        with patch.object(generator, "client", Mock()):
            result = await generator._generate_batch_async(request)

        # Check result
        assert result.batch_id == 0
        assert hasattr(result.processed_results, "remaining_inputs")
        assert len(result.processed_results.remaining_inputs) == 1
        assert "images" in result.processed_results.remaining_inputs[0]


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
            processed = []
            for sample in batch:
                # Create multimodal prompt format with image placeholders
                content_block = []
                content_block.append({"type": "text", "text": sample["question"]})
                content_block.append({"type": "image"})  # Image placeholder

                # Format the prompt with multimodal content
                sample["prompt"] = [{"role": "user", "content": content_block}]

                # Add the actual images
                sample["images"] = [Image.new("RGB", (10, 10), color="red")]

                processed.append(sample)
            return processed

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
        # The eval_dataset should now have images
        assert "images" in env.eval_dataset
        assert len(env.eval_dataset["images"]) == 2

        # Verify the images were added correctly
        assert len(env.eval_dataset["images"][0]) == 1
        assert isinstance(env.eval_dataset["images"][0][0], Image.Image)

    @pytest.mark.asyncio
    async def test_multimodal_generation_flow(self, mock_openai_client):
        """Test the full multimodal generation flow with data collator."""
        # Configure mock client response
        mock_openai_client.set_response("It's a red square")

        # Create a data collator that formats prompts for multimodal
        def multimodal_collator(batch):
            processed = []
            for sample in batch:
                # Create multimodal content structure
                content = [
                    {"type": "text", "text": sample["question"]},
                    {"type": "image"},  # Placeholder for image
                ]
                sample["prompt"] = [{"role": "user", "content": content}]
                # Create a simple PIL image
                img = Image.new("RGB", (10, 10), color="red")
                sample["images"] = [img]
                processed.append(sample)
            return processed

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
            data_collator=multimodal_collator,
        )

        # Run generation on eval dataset (which has data_collator applied)
        test_input = env.get_eval_dataset(n=1)

        # Debug: Check what test_input looks like
        print(f"test_input type: {type(test_input)}")
        if isinstance(test_input, dict):
            print(f"test_input keys: {test_input.keys()}")
            for key, value in test_input.items():
                print(
                    f"  {key}: type={type(value)}, len={len(value) if hasattr(value, '__len__') else 'N/A'}"
                )
                if key == "prompt" and value:
                    print(f"    First prompt: {value[0]}")
                if key == "images" and value:
                    print(f"    First images: {value[0]}")
                    print(
                        f"    Image type: {type(value[0][0]) if value[0] else 'None'}"
                    )

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
