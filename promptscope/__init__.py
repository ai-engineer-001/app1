"""
PromptScope: Production-grade Python module for prompt analysis, comparison, 
response evaluation, and improvement.

This module provides tools for analyzing, comparing, and improving prompts used
with Large Language Models (LLMs). It supports multiple LLM providers and 
includes both online and offline analysis capabilities.
"""

from typing import Optional

__version__ = "1.0.0"
__author__ = "PromptScope Team"
__email__ = "team@promptscope.ai"

from .analysis import PromptAnalyzer
from .compare import PromptComparator  
from .evaluate import ResponseEvaluator
from .improve import PromptImprover
from .scoring import WorkflowScorer
from .apis import APIManager
from .auth import AuthManager

# Main interface class
class PromptScope:
    """
    Main interface for PromptScope functionality.
    
    Provides a unified interface to all prompt analysis, comparison, 
    evaluation, and improvement features.
    """
    
    def __init__(self, model: str = "qwen2.5-vl-32b-instruct", provider: str = "openrouter", 
                 api_manager: Optional[APIManager] = None):
        """
        Initialize PromptScope with specified model and provider.
        
        Args:
            model: Name of the LLM model to use
            provider: API provider (openrouter, openai, anthropic, etc.)
            api_manager: Optional pre-configured API manager (for testing)
        """
        if api_manager:
            self.api = api_manager
            self.auth = api_manager.auth
        else:
            self.auth = AuthManager()
            # Only initialize API manager if we have credentials, otherwise leave as None
            try:
                self.api = APIManager(model=model, provider=provider, auth=self.auth)
            except (EOFError, KeyboardInterrupt):
                # In non-interactive mode, set API manager to None
                self.api = None
        
        self.analyzer = PromptAnalyzer(self.api)
        self.comparator = PromptComparator(self.api)
        self.evaluator = ResponseEvaluator(self.api) 
        self.improver = PromptImprover(self.api)
        self.scorer = WorkflowScorer(self.api)
    
    def analyze(self, input_path: str, **kwargs):
        """Analyze prompts from JSONL file."""
        return self.analyzer.analyze_file(input_path, **kwargs)
    
    def compare(self, prompt1: str, prompt2: str, **kwargs):
        """Compare two prompts."""
        return self.comparator.compare(prompt1, prompt2, **kwargs)
    
    def evaluate(self, input_path: str, **kwargs):
        """Evaluate responses from JSONL file."""
        return self.evaluator.evaluate_file(input_path, **kwargs)
    
    def improve(self, prompt: str, **kwargs):
        """Improve a prompt using LLM suggestions."""
        return self.improver.improve(prompt, **kwargs)
    
    def score_workflow(self, workflow_path: str, **kwargs):
        """Score a workflow chain."""
        return self.scorer.score_file(workflow_path, **kwargs)

# Convenience imports for direct usage
__all__ = [
    "PromptScope",
    "PromptAnalyzer", 
    "PromptComparator",
    "ResponseEvaluator",
    "PromptImprover", 
    "WorkflowScorer",
    "APIManager",
    "AuthManager",
]