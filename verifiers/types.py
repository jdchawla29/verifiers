from typing import (
    Any,
    Annotated,
    Callable,
    Literal,
    Optional,
)

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,  # noqa: F401
)
from openai.types.chat.chat_completion_tool_param import (
    ChatCompletionToolParam,  # noqa: F401
)
from openai.types.completion import Completion
from openai.types.shared_params import (  # noqa: F401
    FunctionDefinition,
    FunctionParameters,
)
from pydantic import BaseModel, Field, SkipValidation, ConfigDict
from PIL import Image

# typing aliases
MessageType = Literal["chat", "completion"]
ModelResponse = Completion | ChatCompletion | None

# Use the OpenAI type with SkipValidation to avoid pydantic ValidatorIterator issues
# This maintains type safety while allowing multimodal messages to work properly
ChatMessage = Annotated[ChatCompletionMessageParam, SkipValidation]
Message = str | ChatMessage
Messages = str | list[ChatMessage]
Info = dict[str, Any]
State = dict[str, Any]
SamplingArgs = dict[str, Any]
RewardFunc = Callable[..., float]

# oai tools
JsonPrimitive = Literal["string", "number", "integer", "boolean", "array", "object"]


class GenerateInputs(BaseModel):
    """Pydantic model for generation inputs."""

    prompt: list[Messages]
    answer: list[str] | None = None
    info: list[dict] | None = None
    task: list[str] | None = None
    completion: list[Messages] | None = None


class GenerateOutputs(BaseModel):
    """Pydantic model for generation outputs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: list[Messages]
    completion: list[Messages]
    answer: list[str]
    state: list[State]
    info: list[Info]
    task: list[str]
    reward: list[float]
    metrics: dict[str, list[float]] = Field(default_factory=dict)
    images: Optional[list[list[Image.Image]]] = None


class RolloutScore(BaseModel):
    """Pydantic model for rollout scores."""

    reward: float
    metrics: dict[str, float] = Field(default_factory=dict)


class RolloutScores(BaseModel):
    """Pydantic model for rubric outputs."""

    reward: list[float]
    metrics: dict[str, list[float]] = Field(default_factory=dict)


class ProcessedOutputs(BaseModel):
    """Pydantic model for processed outputs."""

    prompt_ids: list[list[int]]
    prompt_mask: list[list[int]]
    completion_ids: list[list[int]]
    completion_mask: list[list[int]]
    completion_logprobs: list[list[float]]
    rewards: list[float]
    remaining_inputs: Optional[list[dict[str, Any]]] = None
