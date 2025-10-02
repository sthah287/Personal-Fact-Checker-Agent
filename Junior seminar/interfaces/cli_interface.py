"""
Command-line interface for the Personal Fact-Checker Agent.
Provides a simple CLI for testing and debugging fact-checking functionality.
"""

import asyncio
import argparse
import json
import sys
from typing import Dict, Any
import time

from verification.fact_checker import FactChecker
from config import CONFIG

class CLIInterface:
    """Command-line interface for fact-checking."""
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.fact_checker = None
    
    async def initialize(self):
        """Initialize the fact checker."""
        if self.fact_checker is None:
            print("Initializing Fact Checker...")
            self.fact_checker = FactChecker()
            print("Fact Checker ready!\n")
    
    async def check_claim(self, claim: str, max_evidence: int = 5, 
                         show_steps: bool = False, output_format: str = "text") -> Dict[str, Any]:
        """
        Check a single claim.
        
        Args:
            claim: Claim to verify
            max_evidence: Maximum evidence per source
            show_steps: Whether to show reasoning steps
            output_format: Output format ("text", "json")
            
        Returns:
            Fact-check result dictionary
        """
        await self.initialize()
        
        result = await self.fact_checker.check_claim(
            claim, max_evidence, show_steps
        )
        
        if output_format == "json":
            print(json.dumps(result, indent=2, default=str))
        else:
            self._print_text_result(result)
        
        return result
    
    def _print_text_result(self, result: Dict[str, Any]):
        """Print result in human-readable text format."""
        print("\n" + "="*80)
        print("FACT-CHECK RESULT")
        print("="*80)
        
        # Basic info
        print(f"Claim: {result['claim']}")
        print(f"Verdict: {result['verdict']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Processing Time: {result['processing_time']:.2f} seconds")
        print(f"Evidence Found: {result['total_evidence_found']} items")
        
        # Reasoning
        print(f"\nReasoning: {result['reasoning']}")
        
        # Evidence
        if result.get('evidence'):
            print(f"\n{'='*40}")
            print("EVIDENCE ANALYSIS")
            print("="*40)
            
            for i, evidence in enumerate(result['evidence'], 1):
                print(f"\n--- Evidence {i} ---")
                print(f"Source: {evidence['source'].title()}")
                print(f"Title: {evidence['title']}")
                
                if evidence['supports_claim']:
                    print("Type: ✅ SUPPORTS claim")
                elif evidence['refutes_claim']:
                    print("Type: ❌ REFUTES claim")
                else:
                    print("Type: ➖ NEUTRAL")
                
                print(f"NLI Confidence: {evidence['nli_confidence']:.1%}")
                print(f"Source Credibility: {evidence['credibility_score']:.1%}")
                
                # Truncate long text
                text = evidence['text']
                if len(text) > 200:
                    text = text[:197] + "..."
                print(f"Text: {text}")
                
                if evidence.get('url'):
                    print(f"URL: {evidence['url']}")
        
        # Steps (if requested)
        if result.get('steps'):
            print(f"\n{'='*40}")
            print("REASONING STEPS")
            print("="*40)
            
            for i, step in enumerate(result['steps'], 1):
                print(f"{i}. {step}")
        
        # Metadata
        if result.get('metadata'):
            metadata = result['metadata']
            print(f"\n{'='*40}")
            print("DETAILED ANALYSIS")
            print("="*40)
            
            if 'source_distribution' in metadata:
                print("Source Distribution:")
                for source, count in metadata['source_distribution'].items():
                    print(f"  - {source.title()}: {count} items")
            
            if 'nli_label_distribution' in metadata:
                print("NLI Label Distribution:")
                for label, count in metadata['nli_label_distribution'].items():
                    print(f"  - {label}: {count} items")
            
            print(f"Support Score: {metadata.get('support_score', 0):.2f}")
            print(f"Refute Score: {metadata.get('refute_score', 0):.2f}")
            print(f"Net Score: {metadata.get('net_score', 0):.2f}")
        
        print("\n" + "="*80)
    
    async def interactive_mode(self):
        """Run interactive mode where user can enter multiple claims."""
        await self.initialize()
        
        print("="*60)
        print("PERSONAL FACT-CHECKER - INTERACTIVE MODE")
        print("="*60)
        print("Enter claims to verify. Type 'quit' or 'exit' to stop.")
        print("Commands:")
        print("  - 'help': Show this help")
        print("  - 'status': Show system status")
        print("  - 'config': Show configuration")
        print("  - 'steps on/off': Toggle reasoning steps")
        print("  - 'evidence <num>': Set max evidence per source")
        print()
        
        show_steps = False
        max_evidence = 5
        
        while True:
            try:
                user_input = input("Enter claim (or command): ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  - 'help': Show this help")
                    print("  - 'status': Show system status")
                    print("  - 'config': Show configuration")
                    print("  - 'steps on/off': Toggle reasoning steps")
                    print("  - 'evidence <num>': Set max evidence per source")
                    print("  - 'quit'/'exit': Exit interactive mode")
                    continue
                
                elif user_input.lower() == 'status':
                    await self._show_status()
                    continue
                
                elif user_input.lower() == 'config':
                    self._show_config(show_steps, max_evidence)
                    continue
                
                elif user_input.lower().startswith('steps '):
                    setting = user_input.lower().split()[1]
                    if setting == 'on':
                        show_steps = True
                        print("✅ Reasoning steps enabled")
                    elif setting == 'off':
                        show_steps = False
                        print("❌ Reasoning steps disabled")
                    else:
                        print("Usage: steps on/off")
                    continue
                
                elif user_input.lower().startswith('evidence '):
                    try:
                        num = int(user_input.split()[1])
                        if 1 <= num <= 20:
                            max_evidence = num
                            print(f"✅ Max evidence per source set to {num}")
                        else:
                            print("Evidence count must be between 1 and 20")
                    except (IndexError, ValueError):
                        print("Usage: evidence <number>")
                    continue
                
                # Process as claim
                print(f"\nChecking claim: {user_input}")
                await self.check_claim(user_input, max_evidence, show_steps)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    async def _show_status(self):
        """Show system status."""
        try:
            system_info = self.fact_checker.get_system_info()
            
            print("\n" + "="*40)
            print("SYSTEM STATUS")
            print("="*40)
            
            available_sources = system_info.get("available_sources", [])
            retrieval_stats = system_info.get("retrieval_stats", {})
            
            print(f"Available Sources: {', '.join(available_sources)}")
            print(f"Total Sources: {retrieval_stats.get('total_sources', 0)}")
            
            nli_info = system_info.get("nli_model_info", {})
            print(f"NLI Model: {nli_info.get('model_name', 'Unknown')}")
            print(f"Device: {nli_info.get('device', 'Unknown')}")
            
            embedding_info = retrieval_stats.get("embedding_model", {})
            print(f"Embedding Model: {embedding_info.get('model_name', 'Unknown')}")
            
        except Exception as e:
            print(f"Error getting system status: {e}")
    
    def _show_config(self, show_steps: bool, max_evidence: int):
        """Show current configuration."""
        print("\n" + "="*40)
        print("CURRENT CONFIGURATION")
        print("="*40)
        print(f"Show Steps: {show_steps}")
        print(f"Max Evidence per Source: {max_evidence}")
        print(f"True Threshold: {CONFIG.true_threshold}")
        print(f"False Threshold: {CONFIG.false_threshold}")
        print(f"Similarity Threshold: {CONFIG.retrieval.similarity_threshold}")
    
    async def close(self):
        """Close the CLI and clean up resources."""
        if self.fact_checker:
            await self.fact_checker.close()

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Personal Fact-Checker Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m interfaces.cli_interface --claim "The Earth is round"
  python -m interfaces.cli_interface --interactive
  python -m interfaces.cli_interface --claim "COVID vaccines work" --steps --json
        """
    )
    
    parser.add_argument(
        "--claim", "-c",
        type=str,
        help="Claim to fact-check"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--steps", "-s",
        action="store_true",
        help="Show reasoning steps"
    )
    
    parser.add_argument(
        "--evidence", "-e",
        type=int,
        default=5,
        help="Maximum evidence items per source (default: 5)"
    )
    
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.claim and not args.interactive:
        parser.error("Must specify either --claim or --interactive")
    
    if args.claim and args.interactive:
        parser.error("Cannot use both --claim and --interactive")
    
    # Run CLI
    cli = CLIInterface()
    
    try:
        if args.interactive:
            asyncio.run(cli.interactive_mode())
        else:
            output_format = "json" if args.json else "text"
            asyncio.run(cli.check_claim(
                args.claim, args.evidence, args.steps, output_format
            ))
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        # Clean up
        try:
            asyncio.run(cli.close())
        except:
            pass

if __name__ == "__main__":
    main()



