"""
arXiv API connector for retrieving academic papers and abstracts.
Uses the arXiv API for searching scientific literature.
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any
from datetime import datetime
import re

from .base_connector import BaseConnector, EvidenceItem
from config import get_source_credibility

class ArxivConnector(BaseConnector):
    """Connector for arXiv API."""
    
    def __init__(self):
        super().__init__("arxiv", "http://export.arxiv.org/api/query")
        self.session = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def search(self, query: str, max_results: int = 5) -> List[EvidenceItem]:
        """
        Search arXiv for papers related to the query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of EvidenceItem objects with arXiv paper abstracts
        """
        try:
            session = await self._get_session()
            
            # Prepare search parameters
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            async with session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    print(f"arXiv API returned status {response.status}")
                    return []
                
                xml_content = await response.text()
                return self._parse_arxiv_response(xml_content)
                
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[EvidenceItem]:
        """
        Parse XML response from arXiv API.
        
        Args:
            xml_content: XML response string
            
        Returns:
            List of EvidenceItem objects
        """
        evidence_items = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Find all entry elements
            entries = root.findall('atom:entry', namespaces)
            
            for entry in entries:
                try:
                    # Extract paper information
                    title_elem = entry.find('atom:title', namespaces)
                    title = title_elem.text.strip() if title_elem is not None else "Unknown Title"
                    
                    summary_elem = entry.find('atom:summary', namespaces)
                    abstract = summary_elem.text.strip() if summary_elem is not None else ""
                    
                    # Get paper URL
                    id_elem = entry.find('atom:id', namespaces)
                    paper_url = id_elem.text if id_elem is not None else ""
                    
                    # Get publication date
                    published_elem = entry.find('atom:published', namespaces)
                    pub_date = None
                    if published_elem is not None:
                        try:
                            pub_date = datetime.fromisoformat(
                                published_elem.text.replace('Z', '+00:00')
                            )
                        except:
                            pass
                    
                    # Get authors
                    authors = []
                    author_elems = entry.findall('atom:author', namespaces)
                    for author_elem in author_elems:
                        name_elem = author_elem.find('atom:name', namespaces)
                        if name_elem is not None:
                            authors.append(name_elem.text)
                    
                    # Get categories
                    categories = []
                    category_elems = entry.findall('atom:category', namespaces)
                    for cat_elem in category_elems:
                        term = cat_elem.get('term')
                        if term:
                            categories.append(term)
                    
                    if abstract:
                        # Clean the abstract
                        abstract = self._clean_text(abstract)
                        
                        # Create evidence item
                        evidence_item = EvidenceItem(
                            text=abstract,
                            source="arxiv",
                            title=title,
                            url=paper_url,
                            publication_date=pub_date,
                            credibility_score=get_source_credibility("arxiv"),
                            metadata={
                                "authors": authors,
                                "categories": categories,
                                "source_type": "academic_paper",
                                "paper_id": self._extract_arxiv_id(paper_url)
                            }
                        )
                        
                        evidence_items.append(evidence_item)
                
                except Exception as e:
                    print(f"Error parsing arXiv entry: {e}")
                    continue
        
        except Exception as e:
            print(f"Error parsing arXiv XML response: {e}")
        
        return evidence_items
    
    def _extract_arxiv_id(self, url: str) -> Optional[str]:
        """
        Extract arXiv ID from URL.
        
        Args:
            url: arXiv paper URL
            
        Returns:
            arXiv ID or None
        """
        match = re.search(r'arxiv\.org/abs/([^/]+)', url)
        return match.group(1) if match else None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean arXiv abstract text.
        
        Args:
            text: Raw abstract text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common LaTeX commands that might appear
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Remove dollar signs (LaTeX math mode)
        text = re.sub(r'\$+', '', text)
        
        return text.strip()
    
    def is_available(self) -> bool:
        """
        Check if arXiv API is available.
        
        Returns:
            True if available, False otherwise
        """
        try:
            import requests
            response = requests.get(
                self.base_url,
                params={"search_query": "test", "max_results": 1},
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



