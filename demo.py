"""
Demo script for the Personal Fact-Checker Agent.
Shows example usage and tests the system with sample claims.
"""

import asyncio
import time
from verification.fact_checker import FactChecker

# Sample claims for testing
DEMO_CLAIMS = [
    "The Earth is round",
    "COVID-19 vaccines are effective at preventing severe illness",
    "Climate change is caused by human activities",
    "The Great Wall of China is visible from space",
    "Drinking 8 glasses of water per day is necessary for good health",
    "Artificial intelligence will replace all human jobs by 2030"
]

async def demo_single_claim(fact_checker: FactChecker, claim: str):
    """Demo fact-checking a single claim."""
    print(f"\n{'='*80}")
    print(f"CHECKING CLAIM: {claim}")
    print(f"{'='*80}")
    
    start_time = time.time()
    result = await fact_checker.check_claim(claim, max_evidence_per_source=3, show_steps=True)
    end_time = time.time()
    
    print(f"\nVERDICT: {result['verdict']}")
    print(f"CONFIDENCE: {result['confidence']:.1%}")
    print(f"TIME: {end_time - start_time:.2f} seconds")
    print(f"EVIDENCE: {result['total_evidence_found']} items analyzed")
    
    if result.get('evidence'):
        print(f"\nTOP EVIDENCE:")
        for i, evidence in enumerate(result['evidence'][:2], 1):
            support_type = "SUPPORTS" if evidence['supports_claim'] else "REFUTES" if evidence['refutes_claim'] else "NEUTRAL"
            print(f"  {i}. {support_type} | {evidence['source'].upper()} | {evidence['nli_confidence']:.1%}")
            print(f"     {evidence['title']}")
            print(f"     {evidence['text'][:150]}...")
    
    return result

async def run_demo():
    """Run the complete demo."""
    print("PERSONAL FACT-CHECKER AGENT DEMO")
    print("="*60)
    print("This demo will test the fact-checker with several sample claims.")
    print("Each claim will be verified against Wikipedia and arXiv sources.")
    print("="*60)
    
    # Initialize fact checker
    print("\nInitializing Fact Checker...")
    fact_checker = FactChecker()
    
    # Get system info
    system_info = fact_checker.get_system_info()
    available_sources = system_info.get("available_sources", [])
    print(f"System ready! Available sources: {', '.join(available_sources)}")
    
    # Test each demo claim
    results = []
    total_start_time = time.time()
    
    for i, claim in enumerate(DEMO_CLAIMS, 1):
        print(f"\n\nDEMO {i}/{len(DEMO_CLAIMS)}")
        try:
            result = await demo_single_claim(fact_checker, claim)
            results.append(result)
        except Exception as e:
            print(f"Error checking claim: {e}")
            results.append({"error": str(e), "claim": claim})
        
        # Small delay between claims
        if i < len(DEMO_CLAIMS):
            print("\nWaiting 2 seconds before next claim...")
            await asyncio.sleep(2)
    
    total_end_time = time.time()
    
    # Summary
    print(f"\n\n{'='*80}")
    print("DEMO SUMMARY")
    print(f"{'='*80}")
    
    successful_results = [r for r in results if not r.get('error')]
    
    print(f"Total Claims Tested: {len(DEMO_CLAIMS)}")
    print(f"Successful Checks: {len(successful_results)}")
    print(f"Total Time: {total_end_time - total_start_time:.2f} seconds")
    print(f"Average Time per Claim: {(total_end_time - total_start_time) / len(DEMO_CLAIMS):.2f} seconds")
    
    if successful_results:
        # Verdict distribution
        verdicts = {}
        confidences = []
        evidence_counts = []
        
        for result in successful_results:
            verdict = result['verdict']
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            confidences.append(result['confidence'])
            evidence_counts.append(result['total_evidence_found'])
        
        print(f"\nVERDICT DISTRIBUTION:")
        for verdict, count in verdicts.items():
            print(f"  {verdict}: {count} ({count/len(successful_results)*100:.1f}%)")
        
        print(f"\nSTATISTICS:")
        print(f"  Average Confidence: {sum(confidences)/len(confidences):.1%}")
        print(f"  Average Evidence per Claim: {sum(evidence_counts)/len(evidence_counts):.1f}")
        print(f"  Confidence Range: {min(confidences):.1%} - {max(confidences):.1%}")
    
    # Individual results
    print(f"\nINDIVIDUAL RESULTS:")
    for i, result in enumerate(results, 1):
        if result.get('error'):
            print(f"  {i}. ERROR: {result['claim'][:50]}...")
        else:
            print(f"  {i}. {result['verdict']} ({result['confidence']:.1%}) - {result['claim'][:50]}...")
    
    # Close fact checker
    await fact_checker.close()
    
    print(f"\nDemo completed! Thank you for trying the Personal Fact-Checker Agent.")
    print("To run the web interface: python main.py --web")
    print("To run interactive CLI: python main.py --cli --interactive")

def main():
    """Main demo entry point."""
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")

if __name__ == "__main__":
    main()
