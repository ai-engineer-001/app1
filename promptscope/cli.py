"""
Command-line interface for PromptScope.

Provides comprehensive CLI commands for prompt analysis, comparison,
evaluation, improvement, and workflow scoring.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from . import PromptScope
from .auth import AuthManager
from .apis import APIManager
from .analysis import PromptAnalyzer
from .compare import PromptComparator
from .evaluate import ResponseEvaluator
from .improve import PromptImprover
from .scoring import WorkflowScorer

# Initialize Typer app
app = typer.Typer(
    name="promptscope",
    help="🎯 PromptScope: Production-grade prompt analysis and optimization toolkit",
    add_completion=False
)

console = Console()

# Global state
_auth_manager = None
_api_manager = None

def get_auth_manager() -> AuthManager:
    """Get or create global auth manager."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager

def get_api_manager(model: str = "qwen/qwen-2.5-vl-32b-instruct", 
                   provider: str = "openrouter") -> APIManager:
    """Get or create global API manager."""
    global _api_manager
    if _api_manager is None:
        auth_manager = get_auth_manager()
        _api_manager = APIManager(model=model, provider=provider, auth=auth_manager)
    return _api_manager

@app.command()
def setup(
    interactive: bool = typer.Option(True, "--interactive/--non-interactive", 
                                   help="Run interactive setup wizard")
):
    """🔧 Set up PromptScope authentication and configuration."""
    console.print("🚀 PromptScope Setup", style="bold blue")
    
    auth_manager = get_auth_manager()
    
    if interactive:
        auth_manager.setup_interactive()
    else:
        # Show current status
        status = auth_manager.list_configured_providers()
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="white")
        
        for provider, configured in status.items():
            status_text = "✅ Configured" if configured else "❌ Not configured"
            table.add_row(provider.upper(), status_text)
        
        console.print("\n🔑 Authentication Status:", style="bold")
        console.print(table)
        console.print("\nRun 'promptscope setup --interactive' to configure providers.")

@app.command()
def analyze(
    input_file: str = typer.Argument(..., help="Path to JSONL file with prompts"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("console", "--format", "-f", 
                                   help="Output format: console, json, csv, txt"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    visualize: bool = typer.Option(False, "--visualize", help="Generate visualization plots")
):
    """📊 Analyze prompts for patterns, structure, and optimization opportunities."""
    
    console.print(f"🔍 Analyzing prompts from {input_file}...", style="blue")
    
    try:
        # Initialize analyzer
        api_manager = get_api_manager() if Path(input_file).exists() else None
        analyzer = PromptAnalyzer(api_manager)
        
        # Analyze prompts
        report = analyzer.analyze_file(input_file, display=not json_output)
        
        # Handle output
        if json_output or format_type == "json":
            result = json.dumps(report.__dict__, indent=2, default=str)
            if output:
                Path(output).write_text(result)
                console.print(f"✅ Results saved to {output}", style="green")
            else:
                print(result)
        
        elif output:
            analyzer.export_report(report, output, format_type)
        
        if visualize:
            console.print("📈 Visualization feature coming soon!", style="yellow")
    
    except Exception as e:
        console.print(f"❌ Analysis failed: {e}", style="red")
        raise typer.Exit(1)

@app.command()
def compare(
    prompt1: str = typer.Argument(..., help="First prompt (text or file path)"),
    prompt2: str = typer.Argument(..., help="Second prompt (text or file path)"),
    semantic: bool = typer.Option(True, "--semantic/--no-semantic", 
                                 help="Include semantic similarity analysis"),
    side_by_side: bool = typer.Option(False, "--side-by-side", 
                                     help="Display prompts side by side"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colored output"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("console", "--format", "-f",
                                   help="Output format: console, json, txt"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON")
):
    """🔍 Compare two prompts for semantic and structural differences."""
    
    console.print("🔄 Comparing prompts...", style="blue")
    
    try:
        # Load prompts (handle both text and file paths)
        def load_prompt(prompt_input: str) -> str:
            if Path(prompt_input).exists():
                return Path(prompt_input).read_text().strip()
            return prompt_input
        
        prompt1_text = load_prompt(prompt1)
        prompt2_text = load_prompt(prompt2)
        
        # Initialize comparator
        api_manager = get_api_manager() if semantic else None
        comparator = PromptComparator(api_manager)
        
        # Compare prompts
        result = comparator.compare(
            prompt1_text, 
            prompt2_text,
            display=not json_output,
            side_by_side=side_by_side,
            show_token_diff=not no_color
        )
        
        # Handle output
        if json_output or format_type == "json":
            result_json = json.dumps(result.__dict__, indent=2, default=str)
            if output:
                Path(output).write_text(result_json)
                console.print(f"✅ Results saved to {output}", style="green")
            else:
                print(result_json)
        
        elif output:
            comparator.export_comparison(result, output, format_type)
    
    except Exception as e:
        console.print(f"❌ Comparison failed: {e}", style="red")
        raise typer.Exit(1)

@app.command()
def evaluate(
    input_file: str = typer.Argument(..., help="Path to JSONL file with responses"),
    models: str = typer.Option("openrouter", "--models", "-m", 
                              help="Comma-separated list of models to use"),
    metrics: str = typer.Option("all", "--metrics", 
                               help="Comma-separated metrics: relevance,hallucination,toxicity,fluency"),
    relevance: bool = typer.Option(False, "--relevance", help="Evaluate relevance"),
    hallucination: bool = typer.Option(False, "--hallucination", help="Check for hallucinations"),
    toxicity: bool = typer.Option(False, "--toxicity", help="Evaluate toxicity"),
    fluency: bool = typer.Option(False, "--fluency", help="Evaluate fluency"),
    all_metrics: bool = typer.Option(False, "--all-metrics", help="Evaluate all metrics"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("console", "--format", "-f",
                                   help="Output format: console, json, csv, txt"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    debug: bool = typer.Option(False, "--debug", help="Show detailed debug information")
):
    """📈 Evaluate LLM responses for quality, relevance, and safety."""
    
    console.print(f"📊 Evaluating responses from {input_file}...", style="blue")
    
    try:
        # Parse metrics
        if all_metrics or metrics == "all":
            metrics_list = ['relevance', 'hallucination', 'toxicity', 'fluency', 'coherence', 'completeness']
        else:
            metrics_list = []
            if relevance or "relevance" in metrics:
                metrics_list.append('relevance')
            if hallucination or "hallucination" in metrics:
                metrics_list.append('hallucination')
            if toxicity or "toxicity" in metrics:
                metrics_list.append('toxicity')
            if fluency or "fluency" in metrics:
                metrics_list.append('fluency')
            
            # Add any additional metrics from the metrics string
            if metrics and metrics != "all":
                additional_metrics = [m.strip() for m in metrics.split(',')]
                metrics_list.extend(additional_metrics)
        
        if not metrics_list:
            metrics_list = ['relevance', 'toxicity', 'fluency']  # Default metrics
        
        # Initialize evaluator
        model_list = [m.strip() for m in models.split(',')]
        # Use first model for API manager
        api_manager = get_api_manager(model=model_list[0].split('/')[-1]) if model_list else None
        evaluator = ResponseEvaluator(api_manager)
        
        # Evaluate responses
        report = evaluator.evaluate_file(input_file, metrics=metrics_list)
        
        # Handle output
        if json_output or format_type == "json":
            result_json = json.dumps(report.__dict__, indent=2, default=str)
            if output:
                Path(output).write_text(result_json)
                console.print(f"✅ Results saved to {output}", style="green")
            else:
                print(result_json)
        
        elif output:
            evaluator.export_evaluation(report, output, format_type)
    
    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}", style="red")
        if debug:
            raise
        raise typer.Exit(1)

@app.command()
def improve(
    prompt_input: str = typer.Argument(..., help="Prompt text or file path"),
    model: str = typer.Option("qwen/qwen-2.5-vl-32b-instruct", "--model", "-m",
                             help="Model to use for improvements"),
    provider: str = typer.Option("openrouter", "--provider", "-p",
                                help="API provider to use"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("console", "--format", "-f",
                                   help="Output format: console, json, txt"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    versions: int = typer.Option(2, "--versions", "-v", 
                                help="Number of improved versions to generate")
):
    """✨ Improve prompts using AI-powered suggestions."""
    
    console.print("🔧 Analyzing and improving prompt...", style="blue")
    
    try:
        # Load prompt (handle both text and file paths)
        if Path(prompt_input).exists():
            prompt_text = Path(prompt_input).read_text().strip()
        else:
            prompt_text = prompt_input
        
        # Initialize improver
        api_manager = get_api_manager(model=model, provider=provider)
        improver = PromptImprover(api_manager)
        
        # Improve prompt
        report = improver.improve(prompt_text, display=not json_output)
        
        # Handle output
        if json_output or format_type == "json":
            result_json = json.dumps(report.__dict__, indent=2, default=str)
            if output:
                Path(output).write_text(result_json)
                console.print(f"✅ Results saved to {output}", style="green")
            else:
                print(result_json)
        
        elif output:
            improver.export_improvement(report, output, format_type)
    
    except Exception as e:
        console.print(f"❌ Improvement failed: {e}", style="red")
        raise typer.Exit(1)

@app.command("score-chain")
def score_chain(
    workflow_file: str = typer.Argument(..., help="Path to JSON workflow file"),
    model: str = typer.Option("default", "--model", "-m", 
                             help="Model for cost estimation"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    format_type: str = typer.Option("console", "--format", "-f",
                                   help="Output format: console, json, txt"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    show_tree: bool = typer.Option(True, "--tree/--no-tree", 
                                  help="Show dependency tree visualization")
):
    """🔗 Score and analyze prompt workflow chains."""
    
    console.print(f"📊 Analyzing workflow from {workflow_file}...", style="blue")
    
    try:
        # Initialize scorer
        api_manager = get_api_manager() if Path(workflow_file).exists() else None
        scorer = WorkflowScorer(api_manager)
        
        # Score workflow
        report = scorer.score_file(workflow_file, model=model, display=not json_output)
        
        # Handle output
        if json_output or format_type == "json":
            result_json = json.dumps(report.__dict__, indent=2, default=str)
            if output:
                Path(output).write_text(result_json)
                console.print(f"✅ Results saved to {output}", style="green")
            else:
                print(result_json)
        
        elif output:
            scorer.export_workflow_report(report, output, format_type)
    
    except Exception as e:
        console.print(f"❌ Workflow scoring failed: {e}", style="red")
        raise typer.Exit(1)

@app.command("test-connection")
def test_connection(
    provider: str = typer.Option("openrouter", "--provider", "-p", 
                                help="Provider to test"),
    model: str = typer.Option("qwen/qwen-2.5-vl-32b-instruct", "--model", "-m",
                             help="Model to test")
):
    """🔌 Test API connection to LLM providers."""
    
    console.print(f"🔌 Testing connection to {provider}...", style="blue")
    
    try:
        api_manager = get_api_manager(model=model, provider=provider)
        success = api_manager.test_connection(provider)
        
        if success:
            console.print(f"✅ Connection to {provider} successful!", style="green")
        else:
            console.print(f"❌ Connection to {provider} failed!", style="red")
            raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"❌ Connection test failed: {e}", style="red")
        raise typer.Exit(1)

@app.command()
def status():
    """📋 Show PromptScope configuration and status."""
    
    console.print("📋 PromptScope Status", style="bold blue")
    console.print("="*40)
    
    # Authentication status
    auth_manager = get_auth_manager()
    provider_status = auth_manager.list_configured_providers()
    
    console.print("\n🔑 Authentication Status:", style="bold")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Models Available", style="dim white")
    
    for provider, configured in provider_status.items():
        status_text = "✅ Configured" if configured else "❌ Not configured"
        
        if configured:
            try:
                api_manager = get_api_manager(provider=provider)
                models = api_manager.get_available_models(provider)
                models_text = f"{len(models)} models" if models else "Unknown"
            except:
                models_text = "Connection error"
        else:
            models_text = "N/A"
        
        table.add_row(provider.upper(), status_text, models_text)
    
    console.print(table)
    
    # Configuration info
    config = auth_manager.load_config()
    console.print(f"\n⚙️ Default Provider: {config.get('default_provider', 'Not set')}")
    console.print(f"⚙️ Default Model: {config.get('default_model', 'Not set')}")
    console.print(f"⚙️ Offline Fallback: {'Enabled' if config.get('offline_fallback', False) else 'Disabled'}")

@app.command()
def version():
    """📦 Show PromptScope version information."""
    
    from . import __version__, __author__
    
    panel = Panel(
        f"[bold blue]PromptScope[/bold blue] v{__version__}\n"
        f"Production-grade prompt analysis toolkit\n\n"
        f"Author: {__author__}\n"
        f"Python: {sys.version.split()[0]}\n",
        title="🎯 PromptScope",
        border_style="blue"
    )
    
    console.print(panel)

# Example usage commands
@app.command()
def examples():
    """📚 Show usage examples for PromptScope commands."""
    
    console.print("📚 PromptScope Usage Examples", style="bold blue")
    console.print("="*50)
    
    examples = [
        {
            "title": "📊 Analyze Prompts",
            "command": "promptscope analyze prompts.jsonl --json --output analysis.json",
            "description": "Analyze prompts from JSONL file and save results as JSON"
        },
        {
            "title": "🔍 Compare Prompts", 
            "command": "promptscope compare 'old prompt' 'new prompt' --side-by-side",
            "description": "Compare two prompts with side-by-side display"
        },
        {
            "title": "📈 Evaluate Responses",
            "command": "promptscope evaluate responses.jsonl --all-metrics --models openai,anthropic",
            "description": "Evaluate responses using multiple models and all metrics"
        },
        {
            "title": "✨ Improve Prompts",
            "command": "promptscope improve prompt.txt --model gemini-pro --versions 3",
            "description": "Generate 3 improved versions using Gemini Pro"
        },
        {
            "title": "🔗 Score Workflow",
            "command": "promptscope score-chain workflow.json --model gpt-4 --tree",
            "description": "Score workflow chain with cost estimation and dependency tree"
        },
        {
            "title": "🔌 Test Connection",
            "command": "promptscope test-connection --provider openrouter",
            "description": "Test API connection to OpenRouter"
        }
    ]
    
    for example in examples:
        console.print(f"\n{example['title']}", style="bold green")
        console.print(f"  Command: [cyan]{example['command']}[/cyan]")
        console.print(f"  {example['description']}")
    
    console.print("\n💡 Pro Tips:", style="bold yellow")
    console.print("  • Use --json for programmatic output")
    console.print("  • Set up authentication with: promptscope setup")
    console.print("  • Check status with: promptscope status")
    console.print("  • Most commands support --output to save results")

if __name__ == "__main__":
    app()