import logging

logger = logging.getLogger(__name__)

def python(code: str) -> str:
    """Evaluates a block of Python code and returns output of print() statements. Allowed libraries: astropy, biopython, networkx, numpy, scipy, sympy.
    
    Args:
        code (str): A block of Python code

    Returns:
        The output of the code (truncated to 1000 chars) or an error message

    Examples:
        {"code": "import numpy as np; print(np.array([1, 2, 3]) + np.array([4, 5, 6]))"} -> "[5 7 9]"
        {"code": "import scipy; print(scipy.linalg.inv(np.array([[1, 2], [3, 4]])))"} -> "[[-2.   1. ] [ 1.5 -0.5]]"
        {"code": "import sympy; x, y = sympy.symbols('x y'); print(sympy.integrate(x**2, x))"} -> "x**3/3"
    """

    import subprocess
    import time
    
    # Log the full input code
    logger.debug(f"Python tool called with code (length={len(code)}): {code}")
    
    start_time = time.time()
    try:
        # Run the code block in subprocess with 10-second timeout
        logger.debug("Executing Python code in subprocess with 10-second timeout")
        result = subprocess.run(
            ['python', '-c', code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            text=True
        )
        
        execution_time = time.time() - start_time
        logger.debug(f"Python code execution completed in {execution_time:.3f} seconds")
        logger.debug(f"Return code: {result.returncode}")
        logger.debug(f"Stdout length: {len(result.stdout) if result.stdout else 0}")
        logger.debug(f"Stderr length: {len(result.stderr) if result.stderr else 0}")
        
        if result.stderr:
            logger.debug(f"Python execution stderr: {result.stderr.strip()}")
            return f"Error: {result.stderr.strip()}"
            
        output = result.stdout.strip() if result.stdout else ""
        logger.debug(f"Python execution stdout (first 200 chars): {output[:200]}")
        
        if len(output) > 1000:
            logger.debug(f"Truncating output from {len(output)} to 1000 characters")
            output = output[:1000] + "... (truncated to 1000 chars)"
        
        return output
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        logger.debug(f"Python code execution timed out after {execution_time:.3f} seconds")
        logger.error(f"Python code execution timed out after 10 seconds for code: {code[:100]}...", exc_info=True)
        return "Error: Code execution timed out after 10 seconds"
    except Exception as e:
        execution_time = time.time() - start_time
        logger.debug(f"Unexpected error during Python execution after {execution_time:.3f} seconds: {type(e).__name__}: {str(e)}")
        logger.error(f"Unexpected error during Python execution: {e}", exc_info=True)
        return f"Error: {str(e)}"