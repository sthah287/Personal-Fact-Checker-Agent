"""
Quick test script to verify the system works without heavy model downloads.
"""

import asyncio
from data_sources.wikipedia_connector import WikipediaConnector

async def quick_test():
    """Quick test of basic functionality."""
    print("QUICK TEST - Personal Fact-Checker Components")
    print("=" * 50)
    
    # Test 1: Configuration
    print("\n1. Testing configuration...")
    try:
        from config import CONFIG
        print(f"   [OK] Config loaded successfully")
        print(f"   - Embedding model: {CONFIG.models.embedding_model}")
        print(f"   - NLI model: {CONFIG.models.nli_model_actual}")
        print(f"   - Device: {CONFIG.models.device}")
    except Exception as e:
        print(f"   [ERROR] Config error: {e}")
        return
    
    # Test 2: Wikipedia connector
    print("\n2. Testing Wikipedia connector...")
    try:
        wiki = WikipediaConnector()
        if wiki.is_available():
            print("   [OK] Wikipedia API is available")
            
            # Test search
            print("   - Searching for 'Earth'...")
            results = await wiki.search("Earth", max_results=2)
            print(f"   [OK] Found {len(results)} results")
            
            if results:
                print(f"   - First result: {results[0].title}")
                print(f"   - Source: {results[0].source}")
                print(f"   - Text preview: {results[0].text[:100]}...")
        else:
            print("   [ERROR] Wikipedia API not available")
    except Exception as e:
        print(f"   [ERROR] Wikipedia error: {e}")
    
    # Test 3: arXiv connector
    print("\n3. Testing arXiv connector...")
    try:
        from data_sources.arxiv_connector import ArxivConnector
        arxiv = ArxivConnector()
        if arxiv.is_available():
            print("   [OK] arXiv API is available")
            
            # Test search
            print("   - Searching for 'machine learning'...")
            results = await arxiv.search("machine learning", max_results=1)
            print(f"   [OK] Found {len(results)} results")
            
            if results:
                print(f"   - First result: {results[0].title[:50]}...")
                print(f"   - Source: {results[0].source}")
        else:
            print("   [ERROR] arXiv API not available")
    except Exception as e:
        print(f"   [ERROR] arXiv error: {e}")
    
    print("\n" + "=" * 50)
    print("QUICK TEST COMPLETED")
    print("\nTo run the full demo with AI models:")
    print("  python demo.py")
    print("\nTo run the web interface:")
    print("  python main.py --web")
    print("\nTo run interactive CLI:")
    print("  python main.py --cli --interactive")

if __name__ == "__main__":
    asyncio.run(quick_test())
