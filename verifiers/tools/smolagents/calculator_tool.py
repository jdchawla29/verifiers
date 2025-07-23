from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Import from smolagents package if available, otherwise use local stub
try:
    from smolagents.tools import Tool
except ImportError:
    # Create a stub Tool class for environments without smolagents
    class Tool:
        def __init__(self, *args, **kwargs):
            self.is_initialized = True
            
        def forward(self, *args, **kwargs):
            raise NotImplementedError("This is a stub - real implementation requires smolagents")


class CalculatorTool(Tool):
    """A calculator tool for evaluating mathematical expressions."""
    
    name = "calculator"
    description = "Evaluates a single line of Python math expression. No imports or variables allowed."
    inputs = {
        "expression": {
            "type": "string",
            "description": "A mathematical expression using only numbers and basic operators (+,-,*,/,**,())"
        }
    }
    output_type = "string"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allowed = set("0123456789+-*/.() ")
        self.is_initialized = True
    
    def forward(self, expression: str) -> str:
        """Evaluates a single line of Python math expression. No imports or variables allowed.

        Args:
            expression: A mathematical expression using only numbers and basic operators (+,-,*,/,**,())

        Returns:
            The result of the calculation or an error message

        Examples:
            "2 + 2" -> "4"
            "3 * (17 + 4)" -> "63"
            "100 / 5" -> "20.0"
        """
        logger.debug(f"CalculatorTool.forward() called with expression: '{expression}'")
        
        invalid_chars = [c for c in expression if c not in self.allowed]
        if invalid_chars:
            logger.debug(f"Invalid characters found in expression: {invalid_chars}")
            return "Error: Invalid characters in expression"
        
        logger.debug("Expression validation passed, all characters are allowed")
        
        try:
            # Safely evaluate the expression with no access to builtins
            logger.debug(f"Evaluating expression: {expression}")
            result = eval(expression, {"__builtins__": {}}, {})
            result_str = str(result)
            logger.debug(f"Calculation successful: {expression} = {result_str}")
            return result_str
        except Exception as e:
            logger.debug(f"Calculation failed: {type(e).__name__}: {str(e)}")
            logger.error(f"Failed to evaluate expression '{expression}': {e}", exc_info=True)
            return f"Error: {str(e)}"