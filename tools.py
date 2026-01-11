"""
Tools for the RSS chatbot - web search, article fetching, etc.
"""

import requests
from typing import Optional, List, Dict
from langchain_core.tools import tool
from ddgs import DDGS  # pip install ddgs
from bs4 import BeautifulSoup
import json
from datetime import datetime

# Initialize search tools

@tool
def search_web(query: str) -> str:
    """
    Search the web for current information using DuckDuckGo.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as formatted text
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return "\n".join(f"{r.get('title')} - {r.get('href')}" for r in results) or "No results."
    except Exception as e:
        return f"Search error: {e}"

@tool
def get_current_crypto_price(symbol: str) -> str:
    """
    Get current cryptocurrency price from a public API.
    
    Args:
        symbol: Crypto symbol like 'bitcoin', 'ethereum', 'btc', 'eth'
        
    Returns:
        Current price information
    """
    try:
        # Map common symbols
        symbol_map = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'ada': 'cardano',
            'sol': 'solana',
            'dot': 'polkadot'
        }
        
        crypto_id = symbol_map.get(symbol.lower(), symbol.lower())
        
        # Use CoinGecko API (free, no key required)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies=usd&include_24hr_change=true"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if crypto_id in data:
            price_data = data[crypto_id]
            price = price_data.get('usd', 'N/A')
            change_24h = price_data.get('usd_24h_change', 'N/A')
            
            change_str = f"{change_24h:+.2f}%" if isinstance(change_24h, (int, float)) else "N/A"
            
            return f"💰 {crypto_id.title()} ({symbol.upper()}):\n• Current Price: ${price:,.2f}\n• 24h Change: {change_str}\n• Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
        else:
            return f"❌ Cryptocurrency '{symbol}' not found. Try: bitcoin, ethereum, cardano, solana, etc."
            
    except Exception as e:
        return f"❌ Failed to get crypto price: {e}"

@tool
def fetch_webpage_content(url: str) -> str:
    """
    Fetch and extract text content from a webpage.
    
    Args:
        url: The webpage URL to fetch
        
    Returns:
        Extracted text content
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Truncate if too long
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return f"📄 Content from {url}:\n\n{text}"
        
    except Exception as e:
        return f"❌ Failed to fetch webpage: {e}"

@tool
def search_crypto_news(query: str = "cryptocurrency") -> str:
    """
    Search for latest cryptocurrency news from the web.
    
    Args:
        query: Search query for crypto news
        
    Returns:
        Latest crypto news results
    """
    try:
        q = f"{query} crypto news latest"
        with DDGS() as ddgs:
            results = list(ddgs.text(q, max_results=5))
        return "\n".join(f"{r.get('title')} - {r.get('href')}" for r in results) or "No results."
    except Exception as e:
        return f"Search error: {e}"

@tool
def get_current_time(dummy_param: str = "time") -> str:
    """
    Get the current date and time.
    
    Args:
        dummy_param: Unused parameter to satisfy validation (default: "time")
        
    Returns:
        Current timestamp
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return f"🕐 Current time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}\n📅 Today is: {now.strftime('%A, %B %d, %Y')}"

# List of all available tools
available_tools = [
    search_web,
    get_current_crypto_price,
    fetch_webpage_content,
    search_crypto_news,
    get_current_time
]