import logging

logger = logging.getLogger(__name__)

def search_ddg(query: str, num_results: int = 5) -> str:
    """Searches DuckDuckGo and returns concise summaries of top results.
    
    Args:
        query (str): The search query string
        num_results (int): Number of results to return (default: 5, max: 10)
        
    Returns:
        Formatted string with bullet points of top results, each with title and brief summary
        
    Examples:
        {"query": "who invented the lightbulb", "num_results": 3}
    """
    
    logger.debug(f"DuckDuckGo search called with query: '{query}', num_results: {num_results}")
    actual_num_results = min(num_results, 10)
    if actual_num_results != num_results:
        logger.debug(f"Clamping num_results from {num_results} to {actual_num_results} (max: 10)")

    try:
        from duckduckgo_search import DDGS # type: ignore
        logger.debug("Creating DDGS client")
        with DDGS() as ddgs:
            logger.debug(f"Executing DuckDuckGo search with max_results={actual_num_results}")
            results = list(ddgs.text(query, max_results=actual_num_results))
            
            logger.debug(f"DuckDuckGo returned {len(results)} results")
            if not results:
                logger.debug("No results found from DuckDuckGo")
                return "No results found"

            summaries = []
            for i, r in enumerate(results):
                logger.debug(f"Processing result {i+1}: {r.get('title', 'No title')[:50]}...")
                title = r['title']
                snippet = r['body'][:200].rsplit('.', 1)[0] + '.'
                summaries.append(f"• {title}\n  {snippet}")

            result_text = "\n\n".join(summaries)
            logger.debug(f"Formatted {len(summaries)} search results, total length: {len(result_text)} chars")
            return result_text
    except Exception as e:
        logger.debug(f"DuckDuckGo search failed: {type(e).__name__}: {str(e)}")
        logger.error(f"Failed to search DuckDuckGo for query '{query}': {e}", exc_info=True)
        return f"Error: {str(e)}" 
    
def search(query: str) -> str:
    """Searches the web and returns summaries of top results.
    
    Args:
        query: The search query string

    Returns:
        Formatted string with bullet points of top 3 results, each with title, source, url, and brief summary

    Examples:
        {"query": "who invented the lightbulb"} -> ["Thomas Edison (1847-1931) - Inventor of the lightbulb", ...]
        {"query": "what is the capital of France"} -> ["Paris is the capital of France", ...]
        {"query": "when was the Declaration of Independence signed"} -> ["The Declaration of Independence was signed on July 4, 1776", ...]
    """
    from typing import List, Dict, Any
    import time
    
    logger.debug(f"Brave search called with query: '{query}'")
    start_time = time.time()

    try:
        from brave import Brave # type: ignore
        logger.debug("Creating Brave client")
        brave = Brave()
        
        logger.debug("Executing Brave search with count=10, raw=True")
        results = brave.search(q=query, count=10, raw=True) # type: ignore
        
        search_time = time.time() - start_time
        logger.debug(f"Brave search completed in {search_time:.3f} seconds")
        logger.debug(f"Response structure: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        
        web_results = results.get('web', {}).get('results', []) # type: ignore
        logger.debug(f"Found {len(web_results)} web results")
        
        if not web_results:
            logger.debug("No web results found in Brave response")
            return "No results found"

        summaries = []
        for i, r in enumerate(web_results):
            logger.debug(f"Processing result {i+1}: has_profile={bool('profile' in r)}, url={r.get('url', 'No URL')[:50]}...")
            if 'profile' not in r:
                logger.debug(f"Skipping result {i+1}: no profile data")
                continue
                
            header = f"{r['profile']['name']} ({r['profile']['long_name']})"
            title = r['title']
            snippet = r['description'][:300] + " ..."
            url = r['url'] 
            summaries.append(f"•  {header}\n   {title}\n   {snippet}\n   {url}")
            
            if len(summaries) >= 3:
                logger.debug("Reached 3 summaries, stopping")
                break

        result_text = "\n\n".join(summaries[:3])
        logger.debug(f"Returning {len(summaries[:3])} formatted results, total length: {len(result_text)} chars")
        return result_text
    except Exception as e:
        search_time = time.time() - start_time
        logger.debug(f"Brave search failed after {search_time:.3f} seconds: {type(e).__name__}: {str(e)}")
        logger.error(f"Failed to search Brave for query '{query}': {e}", exc_info=True)
        return f"Error: {str(e)}"