"""
Wikipedia API connector for retrieving evidence from Wikipedia articles.
Uses the Wikipedia REST API for search and content retrieval.
"""

import requests
import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
from datetime import datetime
import re
from bs4 import BeautifulSoup

from .base_connector import BaseConnector, EvidenceItem
from config import get_source_credibility

class WikipediaConnector(BaseConnector):
    """Connector for Wikipedia API."""
    
    def __init__(self):
        super().__init__("wikipedia", "https://en.wikipedia.org/api/rest_v1")
        self.search_url = "https://en.wikipedia.org/w/api.php"
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def search(self, query: str, max_results: int = 5) -> List[EvidenceItem]:
        """
        Search Wikipedia for articles related to the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of EvidenceItem objects with Wikipedia content
        """
        try:
            # First, search for relevant page titles
            page_titles = await self._search_pages(query, max_results)
            
            if not page_titles:
                return []
            
            # Get content for each page
            evidence_items = []
            for title in page_titles[:max_results]:
                content = await self._get_page_content(title)
                if content:
                    evidence_items.append(content)
            
            return evidence_items
            
        except Exception as e:
            print(f"Error searching Wikipedia: {e}")
            return []
    
    async def _search_pages(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for Wikipedia page titles matching the query.
        
        Args:
            query: Search query
            limit: Maximum number of titles to return
            
        Returns:
            List of page titles
        """
        session = await self._get_session()
        
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srprop": "snippet|titlesnippet"
        }
        
        try:
            async with session.get(self.search_url, params=params) as response:
                data = await response.json()
                
                if "query" in data and "search" in data["query"]:
                    return [item["title"] for item in data["query"]["search"]]
                
        except Exception as e:
            print(f"Error searching Wikipedia pages: {e}")
        
        return []
    
    async def _get_page_content(self, title: str) -> Optional[EvidenceItem]:
        """
        Get content from a Wikipedia page.
        
        Args:
            title: Wikipedia page title
            
        Returns:
            EvidenceItem with page content or None if error
        """
        session = await self._get_session()
        
        # Get page extract
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "exintro": True,
            "explaintext": True,
            "exsectionformat": "plain",
            "titles": title,
            "inprop": "url"
        }
        
        try:
            async with session.get(self.search_url, params=params) as response:
                data = await response.json()
                
                if "query" not in data or "pages" not in data["query"]:
                    return None
                
                pages = data["query"]["pages"]
                page_id = list(pages.keys())[0]
                
                if page_id == "-1":  # Page not found
                    return None
                
                page = pages[page_id]
                extract = page.get("extract", "")
                page_url = page.get("fullurl", "")
                
                if not extract:
                    return None
                
                # Clean and truncate the extract
                extract = self._clean_text(extract)
                
                return EvidenceItem(
                    text=extract,
                    source="wikipedia",
                    title=title,
                    url=page_url,
                    credibility_score=get_source_credibility("wikipedia"),
                    metadata={
                        "page_id": page_id,
                        "source_type": "encyclopedia"
                    }
                )
                
        except Exception as e:
            print(f"Error getting Wikipedia page content for '{title}': {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean Wikipedia text content.
        
        Args:
            text: Raw text from Wikipedia
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove citation markers like [1], [citation needed], etc.
        text = re.sub(r'\[[\d\s,]+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text)
        text = re.sub(r'\[clarification needed\]', '', text)
        
        # Remove disambiguation notes
        text = re.sub(r'\(disambiguation\)', '', text)
        
        return text.strip()
    
    def is_available(self) -> bool:
        """
        Check if Wikipedia API is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            response = requests.get(
                self.search_url,
                params={"action": "query", "format": "json", "meta": "siteinfo"},
                timeout=5
            )
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None



