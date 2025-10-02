"""
Evidence retriever that orchestrates data source queries and semantic search.
Combines multiple data sources and provides unified evidence retrieval.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import time

from data_sources import WikipediaConnector, ArxivConnector, EvidenceItem
from .semantic_search import SemanticSearchEngine
from .embedding_manager import EmbeddingManager
from config import CONFIG

class EvidenceRetriever:
    """Orchestrates evidence retrieval from multiple sources with semantic search."""
    
    def __init__(self):
        """Initialize the evidence retriever."""
        self.embedding_manager = EmbeddingManager()
        self.search_engine = SemanticSearchEngine(self.embedding_manager)
        
        # Initialize data source connectors
        self.connectors = {
            "wikipedia": WikipediaConnector(),
            "arxiv": ArxivConnector()
        }
        
        # TODO: Add more connectors (CDC, data.gov, news feeds)
        # self.connectors["cdc"] = CDCConnector()
        # self.connectors["news"] = NewsConnector()
        
        print(f"Initialized evidence retriever with {len(self.connectors)} data sources")
    
    async def retrieve_evidence(self, claim: str, max_results_per_source: int = None,
                              use_semantic_search: bool = True) -> List[EvidenceItem]:
        """
        Retrieve evidence for a given claim from all available sources.
        
        Args:
            claim: The claim to find evidence for
            max_results_per_source: Maximum results per data source
            use_semantic_search: Whether to use semantic search for ranking
            
        Returns:
            List of EvidenceItem objects, ranked by relevance
        """
        max_results = max_results_per_source or CONFIG.retrieval.max_results_per_source
        
        print(f"Retrieving evidence for claim: '{claim[:100]}...'")
        start_time = time.time()
        
        # Step 1: Query all data sources in parallel
        all_evidence = await self._query_all_sources(claim, max_results)
        
        if not all_evidence:
            print("No evidence found from any source")
            return []
        
        print(f"Retrieved {len(all_evidence)} evidence items from {len(self.connectors)} sources")
        
        # Step 2: Use semantic search for ranking if requested
        if use_semantic_search and all_evidence:
            ranked_evidence = await self._rank_evidence_semantically(claim, all_evidence)
        else:
            # Simple ranking by source credibility
            ranked_evidence = sorted(all_evidence, 
                                   key=lambda x: x.credibility_score, reverse=True)
        
        retrieval_time = time.time() - start_time
        print(f"Evidence retrieval completed in {retrieval_time:.2f} seconds")
        
        return ranked_evidence
    
    async def _query_all_sources(self, claim: str, max_results: int) -> List[EvidenceItem]:
        """
        Query all data sources in parallel.
        
        Args:
            claim: The claim to search for
            max_results: Maximum results per source
            
        Returns:
            Combined list of evidence items from all sources
        """
        tasks = []
        
        for source_name, connector in self.connectors.items():
            if connector.is_available():
                task = asyncio.create_task(
                    self._safe_query_source(connector, claim, max_results),
                    name=f"query_{source_name}"
                )
                tasks.append(task)
            else:
                print(f"Skipping {source_name} - not available")
        
        if not tasks:
            print("No available data sources")
            return []
        
        # Wait for all queries to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        all_evidence = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error in task {tasks[i].get_name()}: {result}")
            elif isinstance(result, list):
                all_evidence.extend(result)
        
        return all_evidence
    
    async def _safe_query_source(self, connector, claim: str, max_results: int) -> List[EvidenceItem]:
        """
        Safely query a data source with error handling.
        
        Args:
            connector: Data source connector
            claim: The claim to search for
            max_results: Maximum results to return
            
        Returns:
            List of evidence items or empty list on error
        """
        try:
            return await connector.search(claim, max_results)
        except Exception as e:
            print(f"Error querying {connector.source_name}: {e}")
            return []
    
    async def _rank_evidence_semantically(self, claim: str, 
                                        evidence_items: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Rank evidence items using semantic similarity to the claim.
        
        Args:
            claim: The original claim
            evidence_items: List of evidence items to rank
            
        Returns:
            Evidence items ranked by semantic similarity
        """
        if not evidence_items:
            return []
        
        print("Ranking evidence using semantic similarity...")
        
        # Clear and rebuild search index with current evidence
        self.search_engine.clear_index()
        success = self.search_engine.add_evidence(evidence_items)
        
        if not success:
            print("Failed to build semantic search index, using credibility ranking")
            return sorted(evidence_items, key=lambda x: x.credibility_score, reverse=True)
        
        # Search for evidence similar to the claim
        search_results = self.search_engine.search(
            claim, 
            k=len(evidence_items),
            similarity_threshold=0.0  # Get all items with scores
        )
        
        # Extract ranked evidence items
        ranked_evidence = [item for item, score in search_results]
        
        # Add any items that weren't returned by search (shouldn't happen)
        returned_items = set(id(item) for item, _ in search_results)
        missing_items = [item for item in evidence_items if id(item) not in returned_items]
        ranked_evidence.extend(missing_items)
        
        print(f"Ranked {len(ranked_evidence)} evidence items by semantic similarity")
        return ranked_evidence
    
    def get_evidence_by_source(self, claim: str, source_name: str, 
                             max_results: int = 5) -> List[EvidenceItem]:
        """
        Get evidence from a specific source only.
        
        Args:
            claim: The claim to search for
            source_name: Name of the source (e.g., "wikipedia", "arxiv")
            max_results: Maximum results to return
            
        Returns:
            List of evidence items from the specified source
        """
        if source_name not in self.connectors:
            print(f"Unknown source: {source_name}")
            return []
        
        connector = self.connectors[source_name]
        if not connector.is_available():
            print(f"Source {source_name} is not available")
            return []
        
        # Run async query in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(connector.search(claim, max_results))
        finally:
            loop.close()
    
    def get_available_sources(self) -> List[str]:
        """
        Get list of available data sources.
        
        Returns:
            List of available source names
        """
        available = []
        for name, connector in self.connectors.items():
            if connector.is_available():
                available.append(name)
        return available
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retrieval system.
        
        Returns:
            Dictionary with retrieval statistics
        """
        available_sources = self.get_available_sources()
        search_stats = self.search_engine.get_statistics()
        
        return {
            "total_sources": len(self.connectors),
            "available_sources": available_sources,
            "embedding_model": self.embedding_manager.get_model_info(),
            "search_engine": search_stats
        }
    
    async def close(self):
        """Close all connections and clean up resources."""
        print("Closing evidence retriever...")
        
        # Close data source connections
        for connector in self.connectors.values():
            if hasattr(connector, 'close'):
                await connector.close()
        
        print("Evidence retriever closed")



