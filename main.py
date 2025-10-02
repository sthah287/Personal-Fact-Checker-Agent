"""
Main entry point for the Personal Fact-Checker Agent.
Provides options to run the Gradio web interface or CLI.
"""

import argparse
import sys
import asyncio
from typing import Optional

from interfaces.gradio_app import GradioApp
from interfaces.cli_interface import CLIInterface
from config import CONFIG

def run_gradio_app(port: Optional[int] = None, share: bool = False, debug: bool = False):
    """
    Run the Gradio web application.
    
    Args:
        port: Port to run on (default from config)
        share: Whether to create a public link
        debug: Whether to run in debug mode
    """
    print("Starting Personal Fact-Checker Web Interface...")
    
    app = GradioApp()
    
    try:
        app.launch(
            port=port or CONFIG.ui.gradio_port,
            share=share,
            debug=debug
        )
    except KeyboardInterrupt:
        print("\nShutting down web interface...")
    except Exception as e:
        print(f"Error running web interface: {e}")
        sys.exit(1)
    finally:
        app.close()

def run_cli(claim: Optional[str] = None, interactive: bool = False, 
           steps: bool = False, evidence: int = 5, json_output: bool = False):
    """
    Run the CLI interface.
    
    Args:
        claim: Single claim to check
        interactive: Whether to run in interactive mode
        steps: Whether to show reasoning steps
        evidence: Maximum evidence per source
        json_output: Whether to output JSON
    """
    cli = CLIInterface()
    
    try:
        if interactive:
            print("Starting Personal Fact-Checker Interactive CLI...")
            asyncio.run(cli.interactive_mode())
        elif claim:
            output_format = "json" if json_output else "text"
            asyncio.run(cli.check_claim(claim, evidence, steps, output_format))
        else:
            print("Error: Must specify either a claim or interactive mode")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        try:
            asyncio.run(cli.close())
        except:
            pass

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Personal Fact-Checker Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run web interface
  python main.py --web
  python main.py --web --port 8080 --share
  
  # Run CLI with single claim
  python main.py --cli --claim "The Earth is round"
  python main.py --cli --claim "COVID vaccines work" --steps --json
  
  # Run interactive CLI
  python main.py --cli --interactive
        """
    )
    
    # Interface selection
    interface_group = parser.add_mutually_exclusive_group(required=True)
    interface_group.add_argument(
        "--web", "-w",
        action="store_true",
        help="Run Gradio web interface"
    )
    interface_group.add_argument(
        "--cli", "-c",
        action="store_true",
        help="Run command-line interface"
    )
    
    # Web interface options
    web_group = parser.add_argument_group("Web Interface Options")
    web_group.add_argument(
        "--port", "-p",
        type=int,
        help=f"Port to run web interface on (default: {CONFIG.ui.gradio_port})"
    )
    web_group.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the web interface"
    )
    web_group.add_argument(
        "--debug",
        action="store_true",
        help="Run web interface in debug mode"
    )
    
    # CLI options
    cli_group = parser.add_argument_group("CLI Options")
    cli_group.add_argument(
        "--claim",
        type=str,
        help="Claim to fact-check (for single-use CLI)"
    )
    cli_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run CLI in interactive mode"
    )
    cli_group.add_argument(
        "--steps", "-s",
        action="store_true",
        help="Show reasoning steps"
    )
    cli_group.add_argument(
        "--evidence", "-e",
        type=int,
        default=5,
        help="Maximum evidence items per source (default: 5)"
    )
    cli_group.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results in JSON format"
    )
    
    args = parser.parse_args()
    
    # Validate CLI arguments
    if args.cli and not args.claim and not args.interactive:
        parser.error("CLI mode requires either --claim or --interactive")
    
    if args.cli and args.claim and args.interactive:
        parser.error("Cannot use both --claim and --interactive")
    
    # Run the appropriate interface
    if args.web:
        run_gradio_app(
            port=args.port,
            share=args.share,
            debug=args.debug
        )
    elif args.cli:
        run_cli(
            claim=args.claim,
            interactive=args.interactive,
            steps=args.steps,
            evidence=args.evidence,
            json_output=args.json
        )

if __name__ == "__main__":
    main()



