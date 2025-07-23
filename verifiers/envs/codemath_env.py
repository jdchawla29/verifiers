import subprocess
import logging
from typing import List, Dict, Any, Tuple

from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.parsers import XMLParser
from verifiers.prompts import CODE_PROMPT
from verifiers.rubrics import CodeMathRubric

class CodeMathEnv(MultiTurnEnv):
    def __init__(self,
                 system_prompt: str = CODE_PROMPT,
                 max_turns: int = 5,
                 **kwargs):
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.logger.debug(f"Initializing CodeMathEnv with max_turns={max_turns}")
        parser = XMLParser(fields=["think", ("code", "answer")])
        self.env_parser = XMLParser(fields=["output"])
        rubric = CodeMathRubric(parser=parser, env_parser=self.env_parser)
        super().__init__(
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs
        )
    
    def is_completed(self,
                    messages: List[Dict[str, str]],
                    state: Dict[str, Any],
                    **kwargs: Any) -> bool:
        try:
            parsed = self.parser.parse(messages[-1]["content"])
            # Check if we got a valid answer field (not just None from failed parsing)
            has_answer = hasattr(parsed, 'answer') and parsed.answer is not None
            self.logger.debug(f"Checking completion: has_answer={has_answer}")
            return has_answer
        except Exception as e:
            self.logger.error(f"Failed to check if conversation is completed: {e}", exc_info=True)
            return False

    def run_code(self, code: str, **kwargs: Any) -> str:
        self.logger.debug(f"Running code: {code[:200]}..." if len(code) > 200 else f"Running code: {code}")
        try:
            # Run the code block in subprocess with 10-second timeout
            result = subprocess.run(
                ['python', '-c', code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                text=True
            )
            if result.stderr:
                error_msg = f"Error: {result.stderr.strip()}"
                self.logger.debug(f"Code execution failed: {error_msg}")
                return error_msg
            output = result.stdout.strip() if result.stdout else ""
            self.logger.debug(f"Code output: {output[:100]}..." if len(output) > 100 else f"Code output: {output}")
            return output
        except subprocess.TimeoutExpired:
            self.logger.error("Code execution timed out after 10 seconds", exc_info=True)
            return "Error: Code execution timed out after 10 seconds"

    def env_response(self,
                     messages: List[Dict[str, str]],
                     state: Dict[str, Any],
                     **kwargs: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        self.logger.debug("Generating environment response")
        try:
            parsed = self.parser.parse(messages[-1]["content"])
            self.logger.debug(f"Parsed content: has_code={hasattr(parsed, 'code') and parsed.code is not None}")
            # Check if we got a valid code field (not just None from failed parsing)
            if hasattr(parsed, 'code') and parsed.code is not None:
                output = self.run_code(parsed.code)
                if len(output.strip()) > 0:
                    env_response = {"role": "user", "content": self.env_parser.format(output=output)}
                    self.logger.debug(f"Returning code output as environment response")
                    return env_response, state
                else:
                    self.logger.debug("Code returned empty output")
                    env_response = {"role": "user", "content": "Error: Code execution returned empty output."}
                    return env_response, state
        except Exception as e:
            self.logger.error(f"Failed to parse environment response: {e}", exc_info=True)
        self.logger.debug("No valid code found, returning error message")
        env_response = {"role": "user", "content": "Error: Code not found or invalid XML format. Please ensure correct formatting."}
        return env_response, state