import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Tuple

from openai import AsyncOpenAI

from verifiers import (
    ChatCompletion,
    ChatMessage,
    Completion,
    Environment,
    Info,
    Message,
    Messages,
    MessageType,
    SamplingArgs,
    State,
)

class MultiTurnEnv(Environment):
    def __init__(self,
                 message_type: MessageType = 'chat',
                 max_turns: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.message_type = message_type
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.logger.debug(f"Initialized MultiTurnEnv with max_turns={max_turns}, message_type={message_type}")

    @abstractmethod
    def is_completed(self,
                     messages: Messages,
                     state: State,
                     **kwargs) -> bool:
        pass

    @abstractmethod
    def env_response(self,
                     messages: Messages,
                     state: State,
                     **kwargs) -> Tuple[Message, State]:
        """
        Generate a response from the environment (message, state).
        """
        pass

    async def rollout(self,
                      client: AsyncOpenAI,
                      model: str,
                      prompt: Messages,
                      answer: str = "",
                      task: str = "default",
                      info: Info = {},
                      sampling_args: SamplingArgs = {},
                      **kwargs) -> Tuple[Messages, State]:
        """
        Generate a multi-turn rollout with the environment (messages, state).
        """
        self.logger.debug(f"Starting rollout: model={model}, task={task}, answer_length={len(answer)}")
        is_completed = False
        state = {
            'prompt': prompt,
            'completion': [],
            'answer': answer,
            'task': task,
            'info': info,
            'responses': []
        }
        if self.message_type == 'chat':
            assert isinstance(prompt, list)
            completion = []
        else:
            assert isinstance(prompt, str)
            completion = ""
        rollout = deepcopy(prompt) 
        turn = 0
        while not is_completed:
            self.logger.debug(f"Turn {turn}: checking if completed")
            if self.is_completed(rollout, state, **kwargs):
                is_completed = True
                self.logger.debug(f"Rollout completed at turn {turn}")
                break
            self.logger.debug(f"Turn {turn}: getting model response")
            response = await self.get_model_response(
                prompt=rollout,
                client=client,
                model=model,
                sampling_args=sampling_args,
                message_type=self.message_type
            )
            state['responses'].append(response)
            self.logger.debug(f"Turn {turn}: received model response")
            if self.message_type == 'chat':
                assert isinstance(rollout, list)
                assert isinstance(completion, list)
                assert isinstance(response, ChatCompletion)
                response_text: str = response.choices[0].message.content or ""
                response_message: ChatMessage = {
                    "role": "assistant",
                    "content": response_text
                }
                rollout.append(response_message)
                completion.append(response_message)
            else:
                assert isinstance(rollout, str)
                assert isinstance(completion, str)
                assert isinstance(response, Completion)
                response_text: str = response.choices[0].text or ""
                rollout += response_text
                completion += response_text
            turn += 1
            if self.is_completed(rollout, state, **kwargs) or turn >= self.max_turns:
                is_completed = True
                self.logger.debug(f"Rollout completed: is_completed={self.is_completed(rollout, state, **kwargs)}, "
                                f"turn={turn}, max_turns={self.max_turns}")
            else:
                self.logger.debug(f"Turn {turn}: getting environment response")
                env_msg, state = self.env_response(rollout, state, **kwargs)
                self.logger.debug(f"Turn {turn}: environment response type={type(env_msg)}")
                if self.message_type == 'chat':
                    assert isinstance(env_msg, dict)
                    assert isinstance(rollout, list)
                    assert isinstance(completion, list)
                    rollout += [env_msg]
                    completion += [env_msg]
                else:
                    assert isinstance(env_msg, str)
                    assert isinstance(rollout, str)
                    assert isinstance(completion, str)
                    rollout += env_msg
                    completion += env_msg
        self.logger.debug(f"Rollout finished after {turn} turns")
        return completion, state