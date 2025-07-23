import inspect
import json
import logging
from typing import List, Dict, Any, Callable, Optional, Type, Tuple

from datasets import Dataset

from verifiers import (
    RewardFunc,
    ChatMessage,
    State,
    MultiTurnEnv,
)
from verifiers.parsers.smola_parser import SmolaParser
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE
from verifiers.rubrics.smola_tool_rubric import SmolaToolRubric

class SmolaToolEnv(MultiTurnEnv):
    def __init__(self,
                 dataset: Dataset | None = None,
                 eval_dataset: Dataset | None = None,
                 tools: List[Any] = [],
                 system_prompt: str = DEFAULT_TOOL_PROMPT_TEMPLATE,
                 few_shot: List[Dict[str, str]] = [],
                 mask_env_response: bool = True,
                 max_steps: int = 10, **kwargs):
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.logger.debug(f"Initializing SmolaToolEnv with {len(tools)} tools, max_steps={max_steps}, "
                         f"mask_env_response={mask_env_response}, dataset_size={len(dataset) if dataset else None}")
        # Format the system prompt with tool descriptions
        tool_descriptions = self._format_tool_descriptions(tools)
        formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=formatted_prompt,
            few_shot=few_shot,
            mask_env_response=mask_env_response,
            max_steps=max_steps,
            **kwargs
        )
        self.dataset_name = dataset
        self.max_steps = max_steps
        self.tools = {tool.name: tool for tool in tools}
        self.logger.debug(f"Registered tools: {list(self.tools.keys())}")
        self.rubric = SmolaToolRubric(tools=tools)
        self.llm_parser = SmolaParser(fields=["reasoning", ("tool", "answer")])
        self.env_parser = SmolaParser(fields=["result"])

    def _format_tool_descriptions(self, tools: List[Any]) -> str:
        """Formats tool schemas into a user-friendly description string."""
        self.logger.debug(f"Formatting descriptions for {len(tools)} tools")
        descriptions = []
        for tool in tools:
            desc = [f"{tool.name}: {tool.description}"]
            
            desc.append("\nArguments:")
            for arg_name, arg_info in tool.inputs.items():
                desc.append(f"  - {arg_name}: {arg_info['description']}")
            
            desc.append(f"\nReturns: {tool.output_type}")
            
            descriptions.append("\n".join(desc))
        
        return "\n\n".join(descriptions)

    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        return self.rubric.get_reward_weights()

    def _get_step_count(self, messages: List[ChatMessage]) -> int:
        """Count the number of tool uses in the message history, excluding few-shot examples."""
        step_count = 0
        self.logger.debug(f"Counting steps in {len(messages)} messages")
        
        # Skip messages that are part of few-shot examples
        # We need to determine where the actual conversation starts
        # System message + few-shot examples + user query = start of actual conversation
        conversation_start = 1  # Start after system message
        if self.few_shot:
            # Account for all few-shot messages
            conversation_start += len(self.few_shot)
        
        # Only count tool uses from the actual conversation
        for message in messages[conversation_start:]:
            if message.get("role") == "assistant":
                step_count += 1
        self.logger.debug(f"Counted {step_count} assistant messages (tool uses)")
        return step_count
    
    def is_completed(self, messages: List[ChatMessage], state: State, **kwargs: Any) -> bool:
        try:
            # Check if we've hit max steps by counting tool uses in the message history
            step_count = self._get_step_count(messages)
            if step_count > self.max_steps:
                self.logger.debug(f"Max steps reached: {step_count} > {self.max_steps}")
                return True
            
            parsed = self.llm_parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            has_answer = hasattr(parsed, 'answer') and parsed.answer is not None
            self.logger.debug(f"Checking completion: has_answer={has_answer}")
            return has_answer
        except Exception as e:
            self.logger.error(f"Failed to check if conversation is completed: {e}", exc_info=True)
            return False

    def call_tool(self, tool_json: str, **kwargs: Any) -> str:
        """Call a SmolaAgents Tool object based on JSON command."""
        self.logger.debug(f"Calling tool with JSON: {tool_json[:200]}..." if len(tool_json) > 200 else f"Calling tool with JSON: {tool_json}")
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return "Error: Tool command must be a JSON object, e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
            tool_name = command.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name', e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}'. " + "Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
            tool = self.tools[tool_name]
            tool_args = command.get("args", {})
            if isinstance(tool_args, str):
                return f"Error: Arguments for {tool_name} must be a JSON object matching the tool's input schema, not a string." 
            
            # Call the tool object with arguments
            self.logger.debug(f"Executing tool {tool_name} with args: {tool_args}")
            result = tool(**tool_args)
            self.logger.debug(f"Tool {tool_name} returned: {str(result)[:100]}..." if len(str(result)) > 100 else f"Tool {tool_name} returned: {str(result)}")
            return str(result)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse tool JSON: {e}", exc_info=True)
            return "Error: Invalid JSON format. Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
        except Exception as e:
            self.logger.error(f"Failed to execute tool: {e}", exc_info=True)
            return f"Error: {str(e)}. " + "Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"

    def env_response(self, messages: List[ChatMessage], state: State, **kwargs: Any) -> Tuple[ChatMessage, State]:
        self.logger.debug("Generating environment response")
        try:
            parsed = self.llm_parser.parse(messages[-1]["content"])
            self.logger.debug(f"Parsed content: has_tool={hasattr(parsed, 'tool') and parsed.tool is not None}")
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    response = {"role": "user", "content": self.env_parser.format(result=result)}
                    self.logger.debug(f"Returning tool result as environment response")
                    return response, {}
                else:
                    self.logger.debug("Tool returned empty output")
                    return {"role": "user", "content": "Error: Tool execution returned empty output."}, {}
        except Exception as e:
            self.logger.error(f"Failed to parse environment response: {e}", exc_info=True)
        self.logger.debug("No valid tool command found, returning error message")
        return {"role": "user", "content": "Error: Tool command not found or invalid XML format. Please ensure correct formatting."}, {}