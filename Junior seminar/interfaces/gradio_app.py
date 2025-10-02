"""
Gradio web interface for the Personal Fact-Checker Agent.
Provides an intuitive web UI for claim verification with detailed results display.
"""

import gradio as gr
import asyncio
import json
from typing import Dict, Any, Tuple, List
import time

from verification.fact_checker import FactChecker
from config import CONFIG

class GradioApp:
    """Gradio web application for fact-checking."""
    
    def __init__(self):
        """Initialize the Gradio app."""
        self.fact_checker = None
        self.app = None
        self._initialize_app()
    
    def _initialize_app(self):
        """Initialize the Gradio interface."""
        # Custom CSS for better styling
        css = """
        .verdict-true { background-color: #d4edda !important; border-color: #c3e6cb !important; color: #155724 !important; }
        .verdict-false { background-color: #f8d7da !important; border-color: #f5c6cb !important; color: #721c24 !important; }
        .verdict-unverified { background-color: #fff3cd !important; border-color: #ffeaa7 !important; color: #856404 !important; }
        .evidence-item { border: 1px solid #dee2e6; border-radius: 0.25rem; padding: 1rem; margin: 0.5rem 0; }
        .evidence-supporting { border-left: 4px solid #28a745; }
        .evidence-refuting { border-left: 4px solid #dc3545; }
        .evidence-neutral { border-left: 4px solid #6c757d; }
        .confidence-high { color: #28a745; font-weight: bold; }
        .confidence-medium { color: #ffc107; font-weight: bold; }
        .confidence-low { color: #dc3545; font-weight: bold; }
        """
        
        with gr.Blocks(css=css, title="Personal Fact-Checker Agent") as self.app:
            gr.Markdown("""
            # 🔍 Personal Fact-Checker Agent
            
            Enter a claim below to verify it against trusted sources including Wikipedia, arXiv, and more.
            The system will analyze evidence and provide a verdict with confidence scores.
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    claim_input = gr.Textbox(
                        label="Claim to Verify",
                        placeholder="Enter a factual claim to check (e.g., 'The Earth is round', 'COVID-19 vaccines are effective')",
                        lines=3,
                        max_lines=5
                    )
                    
                    with gr.Row():
                        check_btn = gr.Button("🔍 Check Claim", variant="primary", size="lg")
                        show_steps = gr.Checkbox(label="Show reasoning steps", value=False)
                        max_evidence = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Max evidence per source"
                        )
                
                with gr.Column(scale=1):
                    # System info
                    gr.Markdown("### System Status")
                    system_status = gr.HTML()
            
            # Results section
            with gr.Row():
                with gr.Column():
                    # Main verdict
                    verdict_display = gr.HTML()
                    
                    # Evidence section
                    evidence_display = gr.HTML()
                    
                    # Steps section (hidden by default)
                    steps_display = gr.HTML(visible=False)
                    
                    # Raw JSON (for debugging)
                    with gr.Accordion("Raw Results (JSON)", open=False):
                        json_output = gr.JSON()
            
            # Event handlers
            check_btn.click(
                fn=self._check_claim_wrapper,
                inputs=[claim_input, show_steps, max_evidence],
                outputs=[verdict_display, evidence_display, steps_display, json_output]
            )
            
            show_steps.change(
                fn=lambda show: gr.update(visible=show),
                inputs=[show_steps],
                outputs=[steps_display]
            )
            
            # Load system status on startup
            self.app.load(
                fn=self._get_system_status,
                outputs=[system_status]
            )
    
    async def _initialize_fact_checker(self):
        """Initialize the fact checker (async)."""
        if self.fact_checker is None:
            self.fact_checker = FactChecker()
    
    def _check_claim_wrapper(self, claim: str, show_steps: bool, max_evidence: int) -> Tuple[str, str, str, Dict]:
        """
        Wrapper function to handle async fact-checking in Gradio.
        
        Args:
            claim: Claim to check
            show_steps: Whether to show reasoning steps
            max_evidence: Maximum evidence per source
            
        Returns:
            Tuple of (verdict_html, evidence_html, steps_html, json_result)
        """
        if not claim.strip():
            return (
                self._create_error_html("Please enter a claim to verify."),
                "",
                "",
                {"error": "No claim provided"}
            )
        
        try:
            # Run async fact-checking
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Initialize fact checker if needed
                loop.run_until_complete(self._initialize_fact_checker())
                
                # Check the claim
                result = loop.run_until_complete(
                    self.fact_checker.check_claim(claim, max_evidence, show_steps)
                )
                
                # Generate HTML displays
                verdict_html = self._create_verdict_html(result)
                evidence_html = self._create_evidence_html(result)
                steps_html = self._create_steps_html(result) if show_steps else ""
                
                return verdict_html, evidence_html, steps_html, result
                
            finally:
                loop.close()
                
        except Exception as e:
            error_msg = f"Error during fact-checking: {str(e)}"
            return (
                self._create_error_html(error_msg),
                "",
                "",
                {"error": error_msg}
            )
    
    def _create_verdict_html(self, result: Dict[str, Any]) -> str:
        """Create HTML for verdict display."""
        if result.get("error"):
            return self._create_error_html(result.get("reasoning", "Unknown error"))
        
        verdict = result["verdict"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]
        processing_time = result.get("processing_time", 0)
        
        # Determine CSS class based on verdict
        if verdict == "Likely True":
            css_class = "verdict-true"
            icon = "✅"
        elif verdict == "Likely False":
            css_class = "verdict-false"
            icon = "❌"
        else:
            css_class = "verdict-unverified"
            icon = "❓"
        
        # Confidence styling
        if confidence >= 0.7:
            conf_class = "confidence-high"
        elif confidence >= 0.4:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"
        
        html = f"""
        <div class="verdict-card {css_class}" style="padding: 1.5rem; border-radius: 0.5rem; margin: 1rem 0;">
            <h2 style="margin-top: 0;">{icon} {verdict}</h2>
            <p class="{conf_class}">Confidence: {confidence:.1%}</p>
            <p><strong>Reasoning:</strong> {reasoning}</p>
            <p><small>Processing time: {processing_time:.2f} seconds | Evidence analyzed: {result.get('total_evidence_found', 0)}</small></p>
        </div>
        """
        
        return html
    
    def _create_evidence_html(self, result: Dict[str, Any]) -> str:
        """Create HTML for evidence display."""
        if result.get("error") or not result.get("evidence"):
            return "<p>No evidence to display.</p>"
        
        evidence_items = result["evidence"]
        
        html = "<h3>📋 Evidence Analysis</h3>"
        
        for i, evidence in enumerate(evidence_items, 1):
            # Determine evidence type styling
            if evidence["supports_claim"]:
                border_class = "evidence-supporting"
                type_icon = "✅"
                type_text = "Supports"
            elif evidence["refutes_claim"]:
                border_class = "evidence-refuting"
                type_icon = "❌"
                type_text = "Refutes"
            else:
                border_class = "evidence-neutral"
                type_icon = "➖"
                type_text = "Neutral"
            
            # Format source link
            source_link = evidence.get("url", "")
            if source_link:
                source_display = f'<a href="{source_link}" target="_blank">{evidence["source"].title()}</a>'
            else:
                source_display = evidence["source"].title()
            
            html += f"""
            <div class="evidence-item {border_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <strong>{type_icon} {type_text} Claim</strong>
                    <span style="font-size: 0.9em; color: #6c757d;">
                        NLI: {evidence["nli_confidence"]:.1%} | Source: {source_display}
                    </span>
                </div>
                <h4 style="margin: 0.5rem 0;">{evidence["title"]}</h4>
                <p style="margin-bottom: 0.5rem;">{evidence["text"]}</p>
                <div style="font-size: 0.8em; color: #6c757d;">
                    Credibility: {evidence["credibility_score"]:.1%} | 
                    NLI Label: {evidence["nli_label"]}
                </div>
            </div>
            """
        
        return html
    
    def _create_steps_html(self, result: Dict[str, Any]) -> str:
        """Create HTML for reasoning steps display."""
        if not result.get("steps"):
            return ""
        
        html = "<h3>🔍 Reasoning Steps</h3><ol>"
        
        for step in result["steps"]:
            html += f"<li>{step}</li>"
        
        html += "</ol>"
        return html
    
    def _create_error_html(self, error_msg: str) -> str:
        """Create HTML for error display."""
        return f"""
        <div style="background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; padding: 1rem; border-radius: 0.25rem;">
            <strong>❌ Error:</strong> {error_msg}
        </div>
        """
    
    def _get_system_status(self) -> str:
        """Get system status HTML."""
        try:
            # Try to get system info (this will initialize components if needed)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self._initialize_fact_checker())
                system_info = self.fact_checker.get_system_info()
                
                available_sources = system_info.get("available_sources", [])
                total_sources = len(system_info.get("retrieval_stats", {}).get("total_sources", 0))
                
                status_html = f"""
                <div style="padding: 1rem; background-color: #d1ecf1; border-radius: 0.25rem;">
                    <h4 style="margin-top: 0;">✅ System Ready</h4>
                    <p><strong>Sources:</strong> {len(available_sources)}/{total_sources} available</p>
                    <p><strong>Available:</strong> {', '.join(available_sources)}</p>
                    <p><strong>Model:</strong> {system_info.get('nli_model_info', {}).get('model_name', 'Unknown')}</p>
                </div>
                """
                
                return status_html
                
            finally:
                loop.close()
                
        except Exception as e:
            return f"""
            <div style="padding: 1rem; background-color: #f8d7da; border-radius: 0.25rem;">
                <h4 style="margin-top: 0;">❌ System Error</h4>
                <p>Error initializing system: {str(e)}</p>
            </div>
            """
    
    def launch(self, share: bool = None, port: int = None, debug: bool = False):
        """
        Launch the Gradio app.
        
        Args:
            share: Whether to create a public link
            port: Port to run on
            debug: Whether to run in debug mode
        """
        share = share if share is not None else CONFIG.ui.gradio_share
        port = port if port is not None else CONFIG.ui.gradio_port
        
        print(f"Launching Gradio app on port {port}...")
        
        self.app.launch(
            share=share,
            server_port=port,
            debug=debug,
            show_error=True
        )
    
    def close(self):
        """Close the app and clean up resources."""
        if self.fact_checker:
            # Note: This should be called in an async context
            # asyncio.run(self.fact_checker.close())
            pass



