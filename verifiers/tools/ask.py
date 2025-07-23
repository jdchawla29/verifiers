import os
import logging

logger = logging.getLogger(__name__)

BASE_URL = "https://api.deepinfra.com/v1/openai"
API_KEY = os.getenv("DEEPINFRA_API_KEY")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def get_url_markdown(url: str) -> str:
    """Get contents of URL as nicely formatted markdown."""
    import requests
    from markdownify import markdownify as md
    import time
    
    logger.debug(f"Fetching URL: {url}")
    start_time = time.time()
    
    try:
        logger.debug("Making HTTP GET request with 30s timeout")
        response = requests.get(url, timeout=30)
        
        fetch_time = time.time() - start_time
        logger.debug(f"HTTP request completed in {fetch_time:.3f} seconds")
        logger.debug(f"Response status code: {response.status_code}")
        logger.debug(f"Response content length: {len(response.text)} chars")
        
        response.raise_for_status()
        
        logger.debug("Converting HTML to markdown")
        markdown_content = md(response.text)
        logger.debug(f"Markdown conversion complete, length: {len(markdown_content)} chars")
        
        return markdown_content
    except requests.exceptions.Timeout:
        logger.debug(f"URL fetch timed out after {time.time() - start_time:.3f} seconds")
        logger.error(f"Failed to fetch URL '{url}': Request timed out", exc_info=True)
        return "Error: Request timed out"
    except requests.exceptions.HTTPError as e:
        logger.debug(f"HTTP error: {e.response.status_code} - {e.response.reason}")
        logger.error(f"Failed to fetch URL '{url}': HTTP {e.response.status_code}", exc_info=True)
        return f"Error: HTTP {e.response.status_code} - {e.response.reason}"
    except Exception as e:
        logger.debug(f"Unexpected error fetching URL: {type(e).__name__}: {str(e)}")
        logger.error(f"Failed to fetch URL '{url}': {e}", exc_info=True)
        return f"Error: {str(e)}"

def ask(question: str, url: str) -> str:
    """Ask a question about a web page returned from search results.
    
    Args:
        question: The question to be answered (by an LLM who will be given the web page contents)
        url: The URL of the web page to query

    Returns:
        A LLM-generated answer to the question based on the web page contents.

    Examples:
        {"question": "What is the capital of France?", "url": "https://en.wikipedia.org/wiki/France"} -> "The capital of France is Paris."
        {"question": "How many people live in the United States?", "url": "https://en.wikipedia.org/wiki/United_States"} -> "The population of the United States is approximately 340 million people."
    """
    import time
    
    logger.debug(f"Ask tool called with question: '{question}', url: '{url}'")
    
    # Fetch and truncate contents
    logger.debug("Fetching URL contents")
    contents = get_url_markdown(url)[:50000]
    logger.debug(f"URL contents fetched, truncated to {len(contents)} chars")

    if contents.startswith("Error:"):
        logger.debug(f"URL fetch failed: {contents}")
        return "Error: Failed to fetch URL contents."

    from openai import OpenAI
    
    logger.debug(f"Creating OpenAI client with base_url: {BASE_URL}, model: {MODEL_NAME}")
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    prompt = f"""Answer the following question based on the provided web page contents:

    Question: {question}

    Page: {url}

    Page contents:
    {contents}
    """
    
    logger.debug(f"Prompt length: {len(prompt)} chars")
    logger.debug(f"Sending request to LLM with max_tokens=4000")
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        
        llm_time = time.time() - start_time
        logger.debug(f"LLM response received in {llm_time:.3f} seconds")
        
        answer = response.choices[0].message.content or "Error: No response from model."
        logger.debug(f"LLM response length: {len(answer)} chars")
        logger.debug(f"LLM response preview (first 200 chars): {answer[:200]}...")
        
        return answer
    except Exception as e:
        llm_time = time.time() - start_time
        logger.debug(f"LLM request failed after {llm_time:.3f} seconds: {type(e).__name__}: {str(e)}")
        logger.error(f"Failed to get answer from LLM for question '{question}' about URL '{url}': {e}", exc_info=True)
        return f"Error: {str(e)}"