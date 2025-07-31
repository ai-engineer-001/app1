"""
Workflow scoring module for evaluating prompt chains and workflows.

Analyzes prompt workflows for coherence, redundancy, cost estimation,
and overall effectiveness.
"""

import json
import re
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from .apis import APIManager

console = Console()

@dataclass
class PromptStep:
    """Individual step in a prompt workflow."""
    step_id: str
    prompt: str
    expected_output: Optional[str] = None
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class WorkflowMetrics:
    """Metrics for a prompt workflow."""
    coherence_score: float
    redundancy_score: float
    complexity_score: float
    cost_estimate: float
    efficiency_score: float
    overall_score: float

@dataclass
class StepAnalysis:
    """Analysis of individual workflow step."""
    step_id: str
    length_tokens: int
    complexity: float
    clarity_score: float
    dependency_issues: List[str]
    optimization_suggestions: List[str]

@dataclass
class WorkflowReport:
    """Complete workflow scoring report."""
    workflow_name: str
    total_steps: int
    metrics: WorkflowMetrics
    step_analyses: List[StepAnalysis]
    dependency_graph: Dict[str, List[str]]
    bottlenecks: List[str]
    optimization_recommendations: List[str]
    cost_breakdown: Dict[str, float]

class WorkflowScorer:
    """
    Comprehensive workflow scoring and optimization system.
    
    Evaluates prompt workflows for coherence, efficiency, cost,
    and provides optimization recommendations.
    """
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        """
        Initialize workflow scorer.
        
        Args:
            api_manager: Optional API manager for cost estimation
        """
        self.api_manager = api_manager
        
        # Token pricing (approximate costs per 1000 tokens)
        self.pricing = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-3.5-turbo': {'input': 0.001, 'output': 0.002},
            'claude-3-opus': {'input': 0.015, 'output': 0.075},
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
            'gemini-pro': {'input': 0.0005, 'output': 0.0015},
            'qwen-2.5-vl': {'input': 0.0008, 'output': 0.002},  # OpenRouter pricing
            'default': {'input': 0.002, 'output': 0.004}
        }
    
    def score_workflow(self, workflow: Dict[str, Any], **kwargs) -> WorkflowReport:
        """
        Score a complete workflow.
        
        Args:
            workflow: Workflow definition with steps and dependencies
            **kwargs: Additional scoring options
            
        Returns:
            Complete workflow report
        """
        console.print("📊 Analyzing workflow...", style="blue")
        
        # Parse workflow steps
        steps = self._parse_workflow_steps(workflow)
        
        # Analyze each step
        step_analyses = []
        for step in steps:
            analysis = self._analyze_step(step)
            step_analyses.append(analysis)
        
        # Calculate workflow metrics
        metrics = self._calculate_workflow_metrics(steps, step_analyses)
        
        # Analyze dependencies
        dependency_graph = self._build_dependency_graph(steps)
        bottlenecks = self._identify_bottlenecks(dependency_graph, step_analyses)
        
        # Generate recommendations
        recommendations = self._generate_optimization_recommendations(
            steps, step_analyses, metrics, bottlenecks
        )
        
        # Calculate cost breakdown
        cost_breakdown = self._calculate_cost_breakdown(step_analyses, **kwargs)
        
        report = WorkflowReport(
            workflow_name=workflow.get('name', 'Unnamed Workflow'),
            total_steps=len(steps),
            metrics=metrics,
            step_analyses=step_analyses,
            dependency_graph=dependency_graph,
            bottlenecks=bottlenecks,
            optimization_recommendations=recommendations,
            cost_breakdown=cost_breakdown
        )
        
        # Display results
        if kwargs.get('display', True):
            self._display_workflow_report(report)
        
        return report
    
    def score_file(self, file_path: str, **kwargs) -> WorkflowReport:
        """
        Score workflow from JSON file.
        
        Args:
            file_path: Path to JSON file containing workflow definition
            **kwargs: Additional scoring options
            
        Returns:
            Workflow report
        """
        try:
            with open(file_path, 'r') as f:
                workflow = json.load(f)
            return self.score_workflow(workflow, **kwargs)
        except Exception as e:
            console.print(f"❌ Error loading workflow file: {e}", style="red")
            raise
    
    def _parse_workflow_steps(self, workflow: Dict[str, Any]) -> List[PromptStep]:
        """Parse workflow definition into PromptStep objects."""
        steps = []
        
        workflow_steps = workflow.get('steps', [])
        if isinstance(workflow_steps, dict):
            # Handle dict format: {"step1": {...}, "step2": {...}}
            for step_id, step_data in workflow_steps.items():
                step = PromptStep(
                    step_id=step_id,
                    prompt=step_data.get('prompt', ''),
                    expected_output=step_data.get('expected_output'),
                    dependencies=step_data.get('dependencies', []),
                    metadata=step_data.get('metadata', {})
                )
                steps.append(step)
        
        elif isinstance(workflow_steps, list):
            # Handle list format: [{"id": "step1", "prompt": "...", ...}, ...]
            for i, step_data in enumerate(workflow_steps):
                step = PromptStep(
                    step_id=step_data.get('id', f'step_{i}'),
                    prompt=step_data.get('prompt', ''),
                    expected_output=step_data.get('expected_output'),
                    dependencies=step_data.get('dependencies', []),
                    metadata=step_data.get('metadata', {})
                )
                steps.append(step)
        
        return steps
    
    def _analyze_step(self, step: PromptStep) -> StepAnalysis:
        """Analyze individual workflow step."""
        # Estimate token count (rough approximation: 1 token ≈ 4 characters)
        length_tokens = len(step.prompt) // 4
        
        # Calculate complexity based on prompt characteristics
        complexity = self._calculate_prompt_complexity(step.prompt)
        
        # Calculate clarity score
        clarity_score = self._calculate_clarity_score(step.prompt)
        
        # Check for dependency issues
        dependency_issues = self._check_dependency_issues(step)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_step_optimizations(step, complexity, clarity_score)
        
        return StepAnalysis(
            step_id=step.step_id,
            length_tokens=length_tokens,
            complexity=complexity,
            clarity_score=clarity_score,
            dependency_issues=dependency_issues,
            optimization_suggestions=optimization_suggestions
        )
    
    def _calculate_prompt_complexity(self, prompt: str) -> float:
        """Calculate complexity score for a prompt (0-10)."""
        if not prompt:
            return 0.0
        
        complexity_factors = []
        
        # Length factor
        word_count = len(prompt.split())
        length_complexity = min(1.0, word_count / 200)  # Normalize to max complexity at 200 words
        complexity_factors.append(length_complexity)
        
        # Structural complexity
        has_structure = bool(re.search(r'\n\s*[-*]\s+|\n\s*\d+\.\s+', prompt))
        structure_complexity = 0.8 if has_structure else 0.3
        complexity_factors.append(structure_complexity)
        
        # Instruction complexity
        instruction_count = len(re.findall(r'\b(?:must|should|don\'t|avoid|only|exactly)\b', prompt.lower()))
        instruction_complexity = min(1.0, instruction_count / 5)
        complexity_factors.append(instruction_complexity)
        
        # Variable/placeholder complexity
        variable_count = len(re.findall(r'\{[^}]+\}|\[[^\]]+\]|\$\w+', prompt))
        variable_complexity = min(1.0, variable_count / 3)
        complexity_factors.append(variable_complexity)
        
        # Conditional complexity
        conditional_count = len(re.findall(r'\b(?:if|when|unless|in case)\b', prompt.lower()))
        conditional_complexity = min(1.0, conditional_count / 2)
        complexity_factors.append(conditional_complexity)
        
        # Average complexity
        avg_complexity = sum(complexity_factors) / len(complexity_factors)
        return round(avg_complexity * 10, 2)
    
    def _calculate_clarity_score(self, prompt: str) -> float:
        """Calculate clarity score for a prompt (0-10)."""
        if not prompt:
            return 0.0
        
        clarity_factors = []
        
        # Check for vague language
        vague_patterns = [r'\b(?:some|many|various|several|thing|stuff)\b']
        vague_count = sum(len(re.findall(pattern, prompt.lower())) for pattern in vague_patterns)
        vague_penalty = min(1.0, vague_count / 3)
        clarity_factors.append(1.0 - vague_penalty)
        
        # Check for clear action verbs
        action_verbs = re.findall(r'\b(?:write|create|generate|analyze|explain|describe|list)\b', prompt.lower())
        action_clarity = min(1.0, len(set(action_verbs)) / 2)
        clarity_factors.append(action_clarity)
        
        # Check for specific examples
        has_examples = bool(re.search(r'example:|for instance:|such as', prompt.lower()))
        example_clarity = 1.0 if has_examples else 0.6
        clarity_factors.append(example_clarity)
        
        # Check for format specification
        has_format = bool(re.search(r'format:|json|xml|csv|markdown|bullet', prompt.lower()))
        format_clarity = 1.0 if has_format else 0.7
        clarity_factors.append(format_clarity)
        
        # Average clarity
        avg_clarity = sum(clarity_factors) / len(clarity_factors)
        return round(avg_clarity * 10, 2)
    
    def _check_dependency_issues(self, step: PromptStep) -> List[str]:
        """Check for potential dependency issues."""
        issues = []
        
        if step.dependencies:
            # Check for circular dependencies (simplified check)
            if step.step_id in step.dependencies:
                issues.append("Self-dependency detected")
            
            # Check for missing dependency information
            if len(step.dependencies) > 3:
                issues.append("High dependency count may indicate complexity")
        
        # Check for implicit dependencies in prompt text
        implicit_refs = re.findall(r'(?:previous|above|earlier|from step)', step.prompt.lower())
        if implicit_refs and not step.dependencies:
            issues.append("Implicit dependencies detected but not declared")
        
        return issues
    
    def _generate_step_optimizations(self, step: PromptStep, complexity: float, clarity: float) -> List[str]:
        """Generate optimization suggestions for a step."""
        suggestions = []
        
        if complexity > 7.0:
            suggestions.append("Consider breaking this step into smaller, simpler steps")
        
        if clarity < 6.0:
            suggestions.append("Improve clarity by adding specific examples and clearer instructions")
        
        if len(step.prompt) > 1000:
            suggestions.append("Prompt is very long - consider condensing or restructuring")
        
        if not re.search(r'format:|json|xml|csv', step.prompt.lower()):
            suggestions.append("Specify expected output format for more consistent results")
        
        if step.dependencies and len(step.dependencies) > 2:
            suggestions.append("High dependency count - consider consolidating or reordering steps")
        
        # Check for missing context passing
        if step.dependencies and not re.search(r'\{[^}]+\}|\[[^\]]+\]', step.prompt):
            suggestions.append("Add placeholders for data from dependent steps")
        
        return suggestions
    
    def _calculate_workflow_metrics(self, steps: List[PromptStep], step_analyses: List[StepAnalysis]) -> WorkflowMetrics:
        """Calculate overall workflow metrics."""
        if not steps or not step_analyses:
            return WorkflowMetrics(0, 0, 0, 0, 0, 0)
        
        # Coherence score - based on dependency structure and step clarity
        coherence_score = self._calculate_coherence_score(steps, step_analyses)
        
        # Redundancy score - detect similar or overlapping steps
        redundancy_score = self._calculate_redundancy_score(steps)
        
        # Complexity score - average step complexity
        complexity_score = np.mean([analysis.complexity for analysis in step_analyses])
        
        # Cost estimate - total estimated tokens
        total_tokens = sum(analysis.length_tokens for analysis in step_analyses)
        cost_estimate = self._estimate_cost(total_tokens)
        
        # Efficiency score - inverse of redundancy and complexity
        efficiency_score = 10.0 - (redundancy_score * 0.4 + complexity_score * 0.6)
        efficiency_score = max(0, efficiency_score)
        
        # Overall score - weighted average
        overall_score = (
            coherence_score * 0.3 +
            (10.0 - redundancy_score) * 0.2 +
            (10.0 - complexity_score) * 0.2 +
            efficiency_score * 0.3
        )
        
        return WorkflowMetrics(
            coherence_score=round(coherence_score, 2),
            redundancy_score=round(redundancy_score, 2),
            complexity_score=round(complexity_score, 2),
            cost_estimate=round(cost_estimate, 4),
            efficiency_score=round(efficiency_score, 2),
            overall_score=round(overall_score, 2)
        )
    
    def _calculate_coherence_score(self, steps: List[PromptStep], step_analyses: List[StepAnalysis]) -> float:
        """Calculate workflow coherence score."""
        if not steps:
            return 0.0
        
        coherence_factors = []
        
        # Dependency coherence - well-structured dependencies
        has_dependencies = any(step.dependencies for step in steps)
        if has_dependencies:
            dependency_coherence = 8.0  # Good if dependencies are explicit
        else:
            dependency_coherence = 5.0 if len(steps) > 1 else 8.0  # Neutral for single step
        coherence_factors.append(dependency_coherence)
        
        # Step clarity coherence - average clarity across steps
        avg_clarity = np.mean([analysis.clarity_score for analysis in step_analyses])
        coherence_factors.append(avg_clarity)
        
        # Flow coherence - check for logical progression
        flow_score = 7.0  # Default good score
        if len(steps) > 1:
            # Simple heuristic: steps with dependencies should come after their dependencies
            step_positions = {step.step_id: i for i, step in enumerate(steps)}
            flow_violations = 0
            
            for step in steps:
                if step.dependencies:
                    for dep in step.dependencies:
                        if dep in step_positions and step_positions[dep] >= step_positions[step.step_id]:
                            flow_violations += 1
            
            if flow_violations > 0:
                flow_score = max(3.0, 7.0 - flow_violations * 2)
        
        coherence_factors.append(flow_score)
        
        return sum(coherence_factors) / len(coherence_factors)
    
    def _calculate_redundancy_score(self, steps: List[PromptStep]) -> float:
        """Calculate redundancy score (higher = more redundant)."""
        if len(steps) <= 1:
            return 0.0
        
        redundancy_score = 0.0
        
        # Compare prompts pairwise for similarity
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        try:
            prompts = [step.prompt for step in steps]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = vectorizer.fit_transform(prompts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Calculate average similarity (excluding diagonal)
            similarities = []
            for i in range(len(steps)):
                for j in range(i + 1, len(steps)):
                    similarities.append(similarity_matrix[i][j])
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            redundancy_score = avg_similarity * 10  # Scale to 0-10
            
        except Exception:
            # Fallback to simple word overlap
            redundancy_score = self._calculate_word_overlap_redundancy(steps)
        
        return min(10.0, redundancy_score)
    
    def _calculate_word_overlap_redundancy(self, steps: List[PromptStep]) -> float:
        """Fallback redundancy calculation using word overlap."""
        if len(steps) <= 1:
            return 0.0
        
        # Simple word overlap approach
        step_words = []
        for step in steps:
            words = set(step.prompt.lower().split())
            step_words.append(words)
        
        overlaps = []
        for i in range(len(step_words)):
            for j in range(i + 1, len(step_words)):
                overlap = len(step_words[i].intersection(step_words[j]))
                union = len(step_words[i].union(step_words[j]))
                if union > 0:
                    overlaps.append(overlap / union)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0.0
        return avg_overlap * 10
    
    def _estimate_cost(self, total_tokens: int, model: str = 'default') -> float:
        """Estimate cost for workflow execution."""
        pricing = self.pricing.get(model, self.pricing['default'])
        
        # Assume 70% input tokens, 30% output tokens
        input_tokens = total_tokens * 0.7
        output_tokens = total_tokens * 0.3
        
        cost = (input_tokens / 1000 * pricing['input'] + 
                output_tokens / 1000 * pricing['output'])
        
        return cost
    
    def _build_dependency_graph(self, steps: List[PromptStep]) -> Dict[str, List[str]]:
        """Build dependency graph from workflow steps."""
        graph = {}
        
        for step in steps:
            graph[step.step_id] = step.dependencies or []
        
        return graph
    
    def _identify_bottlenecks(self, dependency_graph: Dict[str, List[str]], 
                             step_analyses: List[StepAnalysis]) -> List[str]:
        """Identify potential bottlenecks in the workflow."""
        bottlenecks = []
        
        # Find steps that many other steps depend on
        dependency_counts = {}
        for step_id, deps in dependency_graph.items():
            for dep in deps:
                dependency_counts[dep] = dependency_counts.get(dep, 0) + 1
        
        # Steps with high dependency counts are potential bottlenecks
        for step_id, count in dependency_counts.items():
            if count > 2:  # More than 2 steps depend on this step
                bottlenecks.append(f"{step_id}: {count} dependent steps")
        
        # Steps with high complexity are also bottlenecks
        analysis_by_id = {analysis.step_id: analysis for analysis in step_analyses}
        for step_id, analysis in analysis_by_id.items():
            if analysis.complexity > 8.0:
                bottlenecks.append(f"{step_id}: high complexity ({analysis.complexity})")
        
        return bottlenecks
    
    def _generate_optimization_recommendations(self, steps: List[PromptStep],
                                               step_analyses: List[StepAnalysis],
                                               metrics: WorkflowMetrics,
                                               bottlenecks: List[str]) -> List[str]:
        """Generate workflow optimization recommendations."""
        recommendations = []
        
        # Coherence recommendations
        if metrics.coherence_score < 6.0:
            recommendations.append("Improve workflow coherence by clarifying step dependencies and instructions")
        
        # Redundancy recommendations
        if metrics.redundancy_score > 6.0:
            recommendations.append("Reduce redundancy by consolidating similar steps or removing duplicates")
        
        # Complexity recommendations
        if metrics.complexity_score > 7.0:
            recommendations.append("Simplify complex steps by breaking them into smaller, focused tasks")
        
        # Efficiency recommendations
        if metrics.efficiency_score < 5.0:
            recommendations.append("Improve efficiency by streamlining the workflow and reducing unnecessary steps")
        
        # Bottleneck recommendations
        if bottlenecks:
            recommendations.append(f"Address {len(bottlenecks)} identified bottlenecks to improve workflow flow")
        
        # Cost recommendations
        if metrics.cost_estimate > 1.0:  # High cost threshold
            recommendations.append("Consider cost optimization by reducing prompt lengths or using cheaper models")
        
        # Step-specific recommendations
        high_complexity_steps = [a for a in step_analyses if a.complexity > 8.0]
        if high_complexity_steps:
            recommendations.append(f"Simplify {len(high_complexity_steps)} high-complexity steps")
        
        low_clarity_steps = [a for a in step_analyses if a.clarity_score < 5.0]
        if low_clarity_steps:
            recommendations.append(f"Improve clarity of {len(low_clarity_steps)} unclear steps")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _calculate_cost_breakdown(self, step_analyses: List[StepAnalysis], **kwargs) -> Dict[str, float]:
        """Calculate detailed cost breakdown."""
        model = kwargs.get('model', 'default')
        pricing = self.pricing.get(model, self.pricing['default'])
        
        breakdown = {}
        total_cost = 0.0
        
        for analysis in step_analyses:
            step_cost = self._estimate_cost(analysis.length_tokens, model)
            breakdown[analysis.step_id] = step_cost
            total_cost += step_cost
        
        breakdown['total'] = total_cost
        breakdown['per_execution'] = total_cost
        breakdown['per_100_executions'] = total_cost * 100
        
        return breakdown
    
    def _display_workflow_report(self, report: WorkflowReport) -> None:
        """Display workflow report in rich format."""
        console.print(f"\n📊 Workflow Analysis: {report.workflow_name}", style="bold blue")
        console.print("="*60)
        
        # Overall metrics
        console.print(f"\n🎯 Overall Score: {report.metrics.overall_score}/10", 
                     style="bold green" if report.metrics.overall_score >= 7 else "bold yellow" if report.metrics.overall_score >= 4 else "bold red")
        
        # Metrics table
        metrics_table = Table(show_header=True, header_style="bold magenta")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", style="white")
        metrics_table.add_column("Status", style="white")
        
        metrics_data = [
            ("Coherence", report.metrics.coherence_score, "Good" if report.metrics.coherence_score >= 6 else "Needs Work"),
            ("Redundancy", report.metrics.redundancy_score, "Low" if report.metrics.redundancy_score <= 4 else "High"),
            ("Complexity", report.metrics.complexity_score, "Manageable" if report.metrics.complexity_score <= 6 else "High"),
            ("Efficiency", report.metrics.efficiency_score, "Good" if report.metrics.efficiency_score >= 6 else "Poor"),
            ("Est. Cost", f"${report.metrics.cost_estimate:.4f}", "Per execution")
        ]
        
        for metric, score, status in metrics_data:
            metrics_table.add_row(metric, str(score), status)
        
        console.print("\n", metrics_table)
        
        # Step analysis
        if report.step_analyses:
            console.print(f"\n📋 Step Analysis ({report.total_steps} steps):", style="bold")
            
            step_table = Table(show_header=True, header_style="bold magenta")
            step_table.add_column("Step ID", style="cyan")
            step_table.add_column("Tokens", style="white")
            step_table.add_column("Complexity", style="white")
            step_table.add_column("Clarity", style="white")
            step_table.add_column("Issues", style="red")
            
            for analysis in report.step_analyses[:10]:  # Show first 10 steps
                issues_text = f"{len(analysis.optimization_suggestions)} suggestions"
                if analysis.dependency_issues:
                    issues_text += f", {len(analysis.dependency_issues)} dep issues"
                
                step_table.add_row(
                    analysis.step_id,
                    str(analysis.length_tokens),
                    f"{analysis.complexity}/10",
                    f"{analysis.clarity_score}/10",
                    issues_text
                )
            
            console.print(step_table)
            
            if len(report.step_analyses) > 10:
                console.print(f"... and {len(report.step_analyses) - 10} more steps")
        
        # Bottlenecks
        if report.bottlenecks:
            console.print(f"\n⚠️ Bottlenecks ({len(report.bottlenecks)}):", style="bold yellow")
            for bottleneck in report.bottlenecks:
                console.print(f"  • {bottleneck}")
        
        # Cost breakdown
        if report.cost_breakdown:
            console.print("\n💰 Cost Breakdown:", style="bold")
            console.print(f"  Per execution: ${report.cost_breakdown.get('total', 0):.4f}")
            console.print(f"  Per 100 executions: ${report.cost_breakdown.get('per_100_executions', 0):.2f}")
        
        # Recommendations
        if report.optimization_recommendations:
            console.print("\n💡 Optimization Recommendations:", style="bold green")
            for i, rec in enumerate(report.optimization_recommendations, 1):
                console.print(f"  {i}. {rec}")
        
        # Dependency tree (if not too complex)
        if report.dependency_graph and len(report.dependency_graph) <= 10:
            console.print("\n🌳 Dependency Structure:", style="bold")
            self._display_dependency_tree(report.dependency_graph)
        
        console.print("\n" + "="*60)
    
    def _display_dependency_tree(self, dependency_graph: Dict[str, List[str]]) -> None:
        """Display dependency graph as a tree."""
        tree = Tree("Workflow Steps")
        
        # Find root nodes (steps with no dependencies)
        all_steps = set(dependency_graph.keys())
        all_deps = set()
        for deps in dependency_graph.values():
            all_deps.update(deps)
        
        root_nodes = all_steps - all_deps
        
        def add_node(parent, step_id, visited=None):
            if visited is None:
                visited = set()
            
            if step_id in visited:
                parent.add(f"{step_id} (circular)")
                return
            
            visited.add(step_id)
            node = parent.add(step_id)
            
            # Add children (steps that depend on this step)
            for child_step, child_deps in dependency_graph.items():
                if step_id in child_deps:
                    add_node(node, child_step, visited.copy())
        
        # Add root nodes to tree
        for root in root_nodes:
            add_node(tree, root)
        
        # Add orphaned nodes (shouldn't happen in well-formed workflows)
        orphans = all_steps - root_nodes
        for step in orphans:
            if step not in str(tree):  # Simple check to avoid duplicates
                tree.add(f"{step} (orphaned)")
        
        console.print(tree)
    
    def export_workflow_report(self, report: WorkflowReport, output_path: str, format_type: str = "json") -> None:
        """
        Export workflow report to file.
        
        Args:
            report: Workflow report to export
            output_path: Output file path
            format_type: Export format ('json', 'txt')
        """
        output_path = Path(output_path)
        
        try:
            if format_type == "json":
                with open(output_path, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
            
            elif format_type == "txt":
                with open(output_path, 'w') as f:
                    f.write(f"WORKFLOW ANALYSIS REPORT: {report.workflow_name}\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Total Steps: {report.total_steps}\n")
                    f.write(f"Overall Score: {report.metrics.overall_score}/10\n\n")
                    
                    f.write("METRICS:\n")
                    f.write(f"  Coherence: {report.metrics.coherence_score}/10\n")
                    f.write(f"  Redundancy: {report.metrics.redundancy_score}/10\n")
                    f.write(f"  Complexity: {report.metrics.complexity_score}/10\n")
                    f.write(f"  Efficiency: {report.metrics.efficiency_score}/10\n")
                    f.write(f"  Est. Cost: ${report.metrics.cost_estimate:.4f}\n\n")
                    
                    if report.bottlenecks:
                        f.write("BOTTLENECKS:\n")
                        for bottleneck in report.bottlenecks:
                            f.write(f"  - {bottleneck}\n")
                        f.write("\n")
                    
                    f.write("RECOMMENDATIONS:\n")
                    for i, rec in enumerate(report.optimization_recommendations, 1):
                        f.write(f"  {i}. {rec}\n")
            
            console.print(f"✅ Workflow report exported to {output_path}", style="green")
            
        except Exception as e:
            console.print(f"❌ Failed to export workflow report: {e}", style="red")