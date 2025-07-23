import re
import logging
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
from types import SimpleNamespace

from verifiers import (
    ChatMessage,
    Messages,
    Parser,
)

class XMLParser(Parser):
    def __init__(self, fields: List[Union[str, Tuple[str, ...]]], answer_field: str = "answer"):
        """
        Initialize the parser with field definitions.
        
        Each field may be:
          - a string (e.g. "reasoning"): the XML tag is fixed.
          - a tuple of alternatives (e.g. ("code", "answer")): the first element is
            the canonical name used for formatting, and all elements are allowed tags
            when parsing.
            
        The schema is assumed to have no duplicate names.
        """
        super().__init__()
        self.logger = logging.getLogger(f"verifiers.parsers.{self.__class__.__name__}")
        self._fields: List[Tuple[str, List[str]]] = []  # List of (canonical, [alternatives])
        self.answer_field = answer_field
        seen = set()
        self.logger.debug(f"Initializing XMLParser with fields: {fields}, answer_field: {answer_field}")
        for field in fields:
            if isinstance(field, str):
                canonical = field
                alternatives = [field]
            elif isinstance(field, tuple):
                if not field:
                    raise ValueError("Field tuple cannot be empty.")
                canonical = field[0]
                if not all(isinstance(alt, str) for alt in field):
                    raise TypeError("All alternatives in a tuple must be strings.")
                alternatives = list(field)
            else:
                raise TypeError("Each field must be a string or a tuple of strings.")
            if canonical in seen:
                raise ValueError(f"Duplicate field name: {canonical}")
            seen.add(canonical)
            self._fields.append((canonical, alternatives))
        self.logger.debug(f"Parsed fields structure: {self._fields}")

    def parse(self, text: str, strip: bool = True) -> Any:
        """
        Parse the given XML string and return an object with attributes corresponding
        to all allowed tags in the schema.
        
        For each field defined:
          - If it is a simple field (e.g. 'reasoning'), the output object will have
            an attribute 'reasoning' set to the text content (or None if missing).
          - If it is defined with alternatives (e.g. ("code", "answer")), the output
            object will have attributes for *each* allowed tag name. For example,
            if the schema is ['reasoning', ('code', 'answer')], then both
            `result.code` and `result.answer` are always accessible. If a tag is not
            found in the XML, its corresponding attribute is set to None.
        """
        self.logger.debug(f"Parsing XML text (length: {len(text)}, strip: {strip}): {text[:200]}..." if len(text) > 200 else f"Parsing XML text: {text}")
        results: Dict[str, Optional[str]] = {}
        for canonical, alternatives in self._fields:
            # For each allowed alternative tag, search independently.
            for alt in alternatives:
                # Regex pattern to capture the content between the tags.
                pattern = rf"<{alt}>\s*(.*?)\s*</{alt}>"
                self.logger.debug(f"Searching for pattern: {pattern}")
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    content = match.group(1).strip() if strip else match.group(1)
                    self.logger.debug(f"Found match for '{alt}': {content[:100]}..." if len(content) > 100 else f"Found match for '{alt}': {content}")
                    results[alt] = content
                else:
                    self.logger.debug(f"No match found for '{alt}'")
                    results[alt] = None
        parsed_result = SimpleNamespace(**results)
        self.logger.debug(f"Parsed result attributes: {list(results.keys())} with values: {[(k, v[:50] + '...' if v and len(v) > 50 else v) for k, v in results.items()]}")
        return parsed_result

    def parse_answer(self, completion: Messages) -> str | None:
        """Extract the last answer from a completion."""
        self.logger.debug(f"Parsing answer from completion (type: {type(completion)}, answer_field: {self.answer_field})")
        if isinstance(completion, str):
            parsed = self.parse(completion)
            answer = getattr(parsed, self.answer_field, None) if hasattr(parsed, self.answer_field) else None
            self.logger.debug(f"Extracted answer from string: {answer}")
            return answer
        else:
            assistant_msgs = self.get_assistant_messages(completion)
            self.logger.debug(f"Searching for answer in {len(assistant_msgs)} assistant messages")
            for i, msg in enumerate(reversed(assistant_msgs)):
                parsed = self.parse(msg['content'])
                if parsed and hasattr(parsed, self.answer_field) and getattr(parsed, self.answer_field) is not None:
                    answer = getattr(parsed, self.answer_field)
                    self.logger.debug(f"Found answer in message {len(assistant_msgs) - i}: {answer}")
                    return answer
        self.logger.debug("No answer found in completion")
        return None

    def get_format_str(self) -> str:
        """
        Return a string that describes the format of the XML.
        """
        format_str = ""
        for field in self._fields:
            if len(field[1]) > 1:
                options = " | ".join(field[1])
                format_str += f"<[ {options} ]>\n...\n</[ {options} ]>\n"
            else:
                format_str += f"<{field[0]}>\n...\n</{field[0]}>\n"
        return format_str.strip()

    def get_format_reward_func(self) -> Callable:
        """
        Return a reward function that checks if messages follow the expected format.
        
        The function does not make assumptions about which fields should start/end the message
        or the specific order of fields. It checks that:
        - At least one field from the schema is present in each message
        - Fields have proper content and spacing
        """
        def format_reward_func(completion: List[ChatMessage]):
            """Reward function that checks if each step follows the expected format."""
            model_messages = self.get_assistant_messages(completion)
            if not model_messages:
                return 0.0
            
            # Calculate format adherence for each message
            format_scores = []
            for msg in model_messages:
                content = msg['content']
                parsed = self.parse(content)
                parsed_no_strip = self.parse(content, strip=False)
                
                # Check if the message has at least one valid field
                has_any_field = False
                fields_with_content = 0
                total_fields = 0
                
                # Keep track of which expected fields are present
                expected_field_count = len(self._fields)  # Total number of expected field sets
                present_field_sets = set()  # Which field sets have at least one alternative present
                
                # Check proper spacing for fields
                has_correct_spacing = True
                
                for i, (canonical, alternatives) in enumerate(self._fields):
                    field_set_present = False
                    for alt in alternatives:
                        if hasattr(parsed, alt) and getattr(parsed, alt) is not None:
                            has_any_field = True
                            fields_with_content += 1
                            total_fields += 1
                            field_set_present = True
                            
                            # Check if field exists in non-stripped version too (proper spacing)
                            if not (hasattr(parsed_no_strip, alt) and 
                                    getattr(parsed_no_strip, alt) is not None):
                                has_correct_spacing = False
                        elif content.count(f"<{alt}>") > 0 or content.count(f"</{alt}>") > 0:
                            # Tag exists but content wasn't properly parsed
                            total_fields += 1
                            field_set_present = True
                    
                    # If any alternative from this field set was present, count it
                    if field_set_present:
                        present_field_sets.add(i)
                
                # Calculate format score components
                format_score = 0.0
                self.logger.debug(f"Message format check - has_any_field: {has_any_field}, fields_with_content: {fields_with_content}, total_fields: {total_fields}, present_field_sets: {present_field_sets}")
                
                # Check if any field from the first field set starts the message
                starts_with_any_field = False
                first_field_set = self._fields[0][1]  # Get alternatives for first field set
                for alt in first_field_set:
                    if content.strip().startswith(f"<{alt}>"):
                        starts_with_any_field = True
                        break
                
                # Check if any field from the last field set ends the message
                ends_with_any_field = False
                last_field_set = self._fields[-1][1]  # Get alternatives for last field set
                for alt in last_field_set:
                    if content.strip().endswith(f"</{alt}>"):
                        ends_with_any_field = True
                        break
                
                # Weight the score based on different criteria
                if has_any_field:
                    # Calculate the proportion of expected field sets that are present
                    field_set_ratio = len(present_field_sets) / expected_field_count
                    format_score += 0.4 * field_set_ratio
                
                if has_correct_spacing:
                    format_score += 0.2
                
                if starts_with_any_field:
                    format_score += 0.2
                    
                if ends_with_any_field:
                    format_score += 0.2
                
                format_scores.append(format_score)
                self.logger.debug(f"Message format score: {format_score}")
            
            # Return average format adherence
            if not format_scores:
                self.logger.debug("No format scores calculated, returning 0.0")
                return 0.0
            avg_score = sum(format_scores) / len(format_scores)
            self.logger.debug(f"Average format score across {len(format_scores)} messages: {avg_score}")
            return avg_score
        
        return format_reward_func

    def get_fields(self) -> List[str]:
        """Return a list of the canonical field names (in order)."""
        return [canonical for canonical, _ in self._fields]
    
    def format(self, **kwargs) -> str:
        """
        Format the provided keyword arguments into an XML string.
        
        For fields with alternatives (tuple), the canonical name (the first element)
        is used as the XML tag. The method looks for a provided value using any of the
        allowed names (preferring the canonical if present).
        
        Example usage:
            parser = XMLParser(['reasoning', ('code', 'answer')])
            formatted_str = parser.format(reasoning="...", code="...")
        """
        self.logger.debug(f"Formatting XML with kwargs: {list(kwargs.keys())}")
        parts = []
        for canonical, alternatives in self._fields:
            value = None
            # Look for a provided value using any of the acceptable keys,
            # preferring the canonical name if it exists.
            if canonical in kwargs:
                value = kwargs[canonical]
            else:
                for alt in alternatives:
                    if alt in kwargs:
                        value = kwargs[alt]
                        break
            if value is None:
                self.logger.error(f"Missing value for field '{canonical}' (allowed: {alternatives})")
                raise ValueError(f"Missing value for field '{canonical}' (allowed: {alternatives})")
            # Use the canonical name as the tag for formatting.
            self.logger.debug(f"Adding field '{canonical}' with value length: {len(str(value))}")
            parts.append(f"<{canonical}>\n{value}\n</{canonical}>")
        formatted = "\n".join(parts)
        self.logger.debug(f"Formatted XML output (length: {len(formatted)})")
        return formatted