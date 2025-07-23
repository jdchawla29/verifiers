import inspect
import json
import logging
from typing import List, Dict, Any, Callable, Tuple

from verifiers import (
    Message,
    Messages,
    MultiTurnEnv,
    RewardFunc,
    State,
    ToolRubric,
    XMLParser
)
from verifiers.prompts import DEFAULT_TOOL_PROMPT_TEMPLATE

def infer_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Infers a tool schema from a function's signature and docstring."""
    logger = logging.getLogger("verifiers.envs.tool_env")
    logger.debug(f"Inferring schema for function: {func.__name__}")
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""
    
    # Parse docstring sections
    doc_parts = doc.split("\n\n")
    description = doc_parts[0].strip()
    
    # Extract examples if present
    examples = []
    return_description = ""
    for part in doc_parts:
        if part.startswith("Examples:"):
            examples = [line.strip() for line in part.split("\n")[1:] if line.strip()]
        elif part.startswith("Returns:"):
            return_description = part.split("\n")[1].strip()

    return_type = str(sig.return_annotation.__name__ if sig.return_annotation != inspect.Parameter.empty else "any")

    logger.debug(f"return_description: {return_description} ({return_type})")
    # Build args schema
    args = {}
    for name, param in sig.parameters.items():
        param_doc = ""
        for part in doc_parts:
            if part.strip().startswith("Args:"):
                for line in part.split("\n")[1:]:
                    if line.strip().startswith(f"{name}:"):
                        param_doc = line.strip()[len(name)+1:].strip()
        
        args[name] = {
            "type": str(param.annotation.__name__ if param.annotation != inspect.Parameter.empty else "any"),
            "description": param_doc,
        }
        if param.default != inspect.Parameter.empty:
            args[name]["default"] = param.default
    
    schema = {
        "name": func.__name__,
        "description": description,
        "args": args,
        "returns": return_description + f" ({return_type})",
        "examples": examples
    }
    logger.debug(f"Generated schema for {func.__name__}: {schema}")
    return schema

def format_tool_descriptions(schemas: List[Dict[str, Any]]) -> str:
    """Formats tool schemas into a user-friendly description string."""
    descriptions = []
    for schema in schemas:
        desc = [f"{schema['name']}: {schema['description']}"]
        
        desc.append("\nArguments:")
        for arg_name, arg_info in schema['args'].items():
            default = f" (default: {arg_info['default']})" if 'default' in arg_info else ""
            desc.append(f"  - {arg_name}: {arg_info['description']}{default}")
        
        if schema['examples']:
            desc.append("\nExamples:")
            for example in schema['examples']:
                desc.append(f"  {example}")
        
        if schema['returns']:
            desc.append(f"\nReturns: {schema['returns']}")
        
        descriptions.append("\n".join(desc))
    
    return "\n\n".join(descriptions)

class ToolEnv(MultiTurnEnv):
    def __init__(self,
                 tools: List[Callable] = [],
                 system_prompt: str = DEFAULT_TOOL_PROMPT_TEMPLATE,
                 format_prompt: bool = True,
                 parser: XMLParser = XMLParser(fields=["think", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"]),
                 max_turns: int = 10, **kwargs):
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.logger.debug(f"Initializing ToolEnv with {len(tools)} tools, max_turns={max_turns}, format_prompt={format_prompt}")
        rubric = ToolRubric(tools=tools, parser=parser, env_parser=env_parser)
        self.tool_schemas = [infer_schema_from_function(tool) for tool in tools]
        self.tools = {tool.__name__: tool for tool in tools}
        self.logger.debug(f"Registered tools: {list(self.tools.keys())}")
        
        if format_prompt:
            tool_descriptions = format_tool_descriptions(self.tool_schemas)
            formatted_prompt = system_prompt.format(tool_descriptions=tool_descriptions)
        else:
            formatted_prompt = system_prompt
        super().__init__(
            system_prompt=formatted_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs
        )
        self.env_parser = env_parser

    def get_reward_funcs(self, **kwargs) -> List[RewardFunc]:
        return self.rubric.get_reward_funcs()
    
    def get_reward_weights(self, **kwargs) -> List[float]:
        return self.rubric.get_reward_weights()
 
    def is_completed(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> bool:
        has_answer = self.parser.parse_answer(messages) is not None
        self.logger.debug(f"Checking if completed: has_answer={has_answer}")
        return has_answer

    def call_tool(self, tool_json: str, max_chars: int = 1024, **kwargs) -> str:
        """Call a tool based on JSON command."""
        self.logger.debug(f"Calling tool with JSON: {tool_json[:200]}..." if len(tool_json) > 200 else f"Calling tool with JSON: {tool_json}")
        try:
            command = json.loads(tool_json)
            if not isinstance(command, dict):
                return "Error: Tool command must be a JSON object, e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
            tool_name = command.get("name")
            if not tool_name:
                return "Error: Tool command must specify 'name', e.g. '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
            if tool_name not in self.tools:
                return f"Error: Unknown tool '{tool_name}. " + "Please format your tool call as '{\"name\": \"tool_name\", \"args\": {\"arg1\": \"value1\", \"arg2\": \"value2\"}}'"
            
            tool_func = self.tools[tool_name]
            tool_args = command.get("args", {})
            if isinstance(tool_args, str):
                tool_schema = next((schema['args'] for schema in self.tool_schemas if schema['name'] == tool_name), None)
                return f"Error: Arguments for {tool_name} must be a JSON object with schema {tool_schema}, not a string." 
            
            # Call the tool function with arguments
            self.logger.debug(f"Executing tool {tool_name} with args: {tool_args}")
            result = tool_func(**tool_args)
            if max_chars > 0 and len(str(result)) > max_chars:
                result = str(result)[:max_chars] + "..."
                self.logger.debug(f"Tool result truncated to {max_chars} chars")
            self.logger.debug(f"Tool {tool_name} returned: {str(result)[:100]}..." if len(str(result)) > 100 else f"Tool {tool_name} returned: {str(result)}")
            return str(result)
        except Exception as e:
            self.logger.error(f"Failed to execute tool: {e}", exc_info=True)
            return f"Error: {str(e)}. " + "Please format your tool call as '{{\"name\": \"tool_name\", \"args\": {{\"arg1\": \"value1\", \"arg2\": \"value2\"}}}}'"

    def env_response(self,
                     messages: Messages, 
                     state: State,
                     **kwargs) -> Tuple[Message, State]:
        self.logger.debug("Generating environment response")
        try:
            parsed = self.parser.parse(messages[-1]['content'])
            self.logger.debug(f"Parsed content: has_tool={hasattr(parsed, 'tool') and parsed.tool is not None}")
            # Check if we got a valid tool field (not just None from failed parsing)
            if hasattr(parsed, 'tool') and parsed.tool is not None:
                result = self.call_tool(parsed.tool)
                if len(result.strip()) > 0:
                    response = {'role': 'user', 'content': self.env_parser.format(result=result)}
                    self.logger.debug(f"Returning tool result as environment response")
                    return response, state
                else:
                    self.logger.debug("Tool returned empty output")
                    return {'role': 'user', 'content': "Error: Tool execution returned empty output."}, state
        except Exception as e:
            self.logger.error(f"Failed to parse tool response: {e}", exc_info=True)
        self.logger.debug("No valid tool command found, returning error message")
        return {'role': 'user', 'content': "Error: Tool command not found or invalid XML format. Please ensure correct formatting."}, state