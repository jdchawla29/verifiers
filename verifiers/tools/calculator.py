import logging

logger = logging.getLogger(__name__)

def calculator(expression: str) -> str:
    """Evaluates a single line of Python math expression. No imports or variables allowed.

    Args:
        expression (str): A mathematical expression using only numbers and basic operators (+,-,*,/,**,())

    Returns:
        The result of the calculation or an error message

    Examples:
        "2 + 2" -> "4"
        "3 * (17 + 4)" -> "63"
        "100 / 5" -> "20.0"
    """
    logger.debug(f"Calculator called with expression: '{expression}'")
    
    allowed = set("0123456789+-*/.() ")
    invalid_chars = [c for c in expression if c not in allowed]
    
    if invalid_chars:
        logger.debug(f"Invalid characters found: {invalid_chars}")
        return "Error: Invalid characters in expression"
    
    logger.debug("Expression contains only valid characters")
    
    try:
        logger.debug(f"Evaluating expression: {expression}")
        result = eval(expression, {"__builtins__": {}}, {})
        result_str = str(result)
        logger.debug(f"Calculation successful: {expression} = {result_str}")
        return result_str
    except Exception as e:
        logger.debug(f"Calculation failed: {type(e).__name__}: {str(e)}")
        logger.error(f"Failed to evaluate expression '{expression}': {e}", exc_info=True)
        return f"Error: {str(e)}" 