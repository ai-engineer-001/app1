"""
Prompt improvement module using LLM-powered suggestions.

Analyzes prompts for clarity, specificity, and effectiveness, then provides
AI-generated suggestions for improvement.
"""

import re
import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table

from .apis import APIManager, LLMRequest

console = Console()

@dataclass
class ImprovementSuggestion:
    """Individual improvement suggestion."""
    category: str
    issue: str
    suggestion: str
    example: Optional[str] = None
    priority: str = "medium"  # high, medium, low

@dataclass
class ImprovedPrompt:
    """Improved version of a prompt."""
    original: str
    improved: str
    improvements_made: List[str]
    confidence_score: float

@dataclass
class ImprovementReport:
    """Complete prompt improvement analysis."""
    original_prompt: str
    analysis: Dict[str, Any]
    suggestions: List[ImprovementSuggestion]
    improved_versions: List[ImprovedPrompt]
    overall_assessment: str
    recommendations: List[str]

class PromptImprover:
    """
    AI-powered prompt improvement system.
    
    Analyzes prompts for common issues and generates improved versions
    using LLM capabilities with structured improvement frameworks.
    """
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        """
        Initialize prompt improver.
        
        Args:
            api_manager: API manager for LLM-powered improvements
        """
        self.api_manager = api_manager
        
        # Improvement analysis prompt template
        self.analysis_prompt = """
        Analyze the following prompt for potential improvements across these dimensions:

        **PROMPT TO ANALYZE:**
        {prompt}

        **ANALYSIS FRAMEWORK:**
        1. **CLARITY**: Is the prompt clear and unambiguous?
        2. **SPECIFICITY**: Does it provide enough specific details and context?
        3. **STRUCTURE**: Is it well-organized and easy to follow?
        4. **COMPLETENESS**: Does it include all necessary information?
        5. **ACTIONABILITY**: Are the instructions actionable and concrete?
        6. **EXAMPLES**: Would examples or demonstrations help?
        7. **CONSTRAINTS**: Are important constraints and requirements specified?
        8. **TONE**: Is the tone appropriate for the intended use?

        For each dimension, provide:
        - Current status (Good/Needs Improvement/Poor)
        - Specific issues identified
        - Concrete suggestions for improvement
        - Priority level (High/Medium/Low)

        Respond in a structured format that clearly separates each dimension.
        """
        
        # Improvement generation prompt template
        self.improvement_prompt = """
        Based on this analysis, create 2-3 improved versions of the original prompt.

        **ORIGINAL PROMPT:**
        {original_prompt}

        **IDENTIFIED ISSUES:**
        {issues}

        **IMPROVEMENT GUIDELINES:**
        1. Make instructions more specific and actionable
        2. Add relevant context and constraints
        3. Improve structure and organization
        4. Include examples where helpful
        5. Clarify expected output format
        6. Remove ambiguity and vague language

        For each improved version, provide:
        - The complete improved prompt
        - List of specific improvements made
        - Confidence score (0-10) for this version

        Focus on practical improvements that will lead to better LLM responses.
        """
        
        # Common prompt issues patterns
        self.issue_patterns = {
            'vague_language': [
                r'\b(?:some|many|various|several|different)\b',
                r'\b(?:good|nice|great|awesome|cool)\b',
                r'\b(?:thing|stuff|something|anything)\b'
            ],
            'unclear_instructions': [
                r'\b(?:maybe|perhaps|could|might)\b',
                r'\b(?:try to|attempt to)\b',
                r'or something'
            ],
            'missing_context': [
                r'^(?:write|create|make|generate)\b',  # Starts with action but no context
            ],
            'no_examples': [
                r'(?:like|such as|for example)',  # Mentions examples but doesn't provide them
            ],
            'unclear_format': [
                r'(?:format|structure|organize)',  # Mentions format but doesn't specify
            ],
            'missing_constraints': [
                r'(?:short|long|brief|detailed)',  # Vague length constraints
            ]
        }
    
    def improve(self, prompt: str, **kwargs) -> ImprovementReport:
        """
        Analyze and improve a prompt.
        
        Args:
            prompt: Original prompt to improve
            **kwargs: Additional options
            
        Returns:
            Complete improvement report
        """
        console.print("🔍 Analyzing prompt for improvements...", style="blue")
        
        # Analyze the prompt
        analysis = self._analyze_prompt(prompt)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(prompt, analysis)
        
        # Create improved versions
        improved_versions = []
        if self.api_manager:
            improved_versions = self._generate_improved_versions(prompt, suggestions)
        else:
            # Fallback to heuristic improvements
            improved_versions = self._create_heuristic_improvements(prompt, suggestions)
        
        # Generate overall assessment and recommendations
        overall_assessment = self._generate_overall_assessment(analysis, suggestions)
        recommendations = self._generate_recommendations(suggestions)
        
        report = ImprovementReport(
            original_prompt=prompt,
            analysis=analysis,
            suggestions=suggestions,
            improved_versions=improved_versions,
            overall_assessment=overall_assessment,
            recommendations=recommendations
        )
        
        # Display results
        if kwargs.get('display', True):
            self._display_improvement_report(report)
        
        return report
    
    def improve_file(self, file_path: str, **kwargs) -> ImprovementReport:
        """
        Improve prompt from file.
        
        Args:
            file_path: Path to file containing prompt
            **kwargs: Additional options
            
        Returns:
            Improvement report
        """
        try:
            prompt = Path(file_path).read_text().strip()
            return self.improve(prompt, **kwargs)
        except Exception as e:
            console.print(f"❌ Error reading file: {e}", style="red")
            raise
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt for common issues and characteristics."""
        analysis = {
            'length': {
                'chars': len(prompt),
                'words': len(prompt.split()),
                'sentences': len(re.split(r'[.!?]+', prompt))
            },
            'structure': self._analyze_structure(prompt),
            'clarity': self._analyze_clarity(prompt),
            'specificity': self._analyze_specificity(prompt),
            'actionability': self._analyze_actionability(prompt),
            'completeness': self._analyze_completeness(prompt)
        }
        
        return analysis
    
    def _analyze_structure(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt structure."""
        structure_analysis = {
            'has_sections': bool(re.search(r'\n\s*[-*]\s+|\n\s*\d+\.\s+', prompt)),
            'has_examples': bool(re.search(r'example|for instance|such as', prompt.lower())),
            'has_constraints': bool(re.search(r'must|should|don\'t|avoid|only|exactly', prompt.lower())),
            'paragraph_count': len(prompt.split('\n\n')),
            'question_count': prompt.count('?'),
            'instruction_markers': len(re.findall(r'^(?:please|do|don\'t|make|create)', prompt.lower(), re.MULTILINE))
        }
        
        # Assess structure quality
        structure_score = 0
        if structure_analysis['has_sections']:
            structure_score += 2
        if structure_analysis['has_examples']:
            structure_score += 2
        if structure_analysis['has_constraints']:
            structure_score += 1
        if structure_analysis['paragraph_count'] > 1:
            structure_score += 1
        
        structure_analysis['score'] = min(10, structure_score)
        structure_analysis['quality'] = 'good' if structure_score >= 4 else 'fair' if structure_score >= 2 else 'poor'
        
        return structure_analysis
    
    def _analyze_clarity(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt clarity."""
        clarity_issues = []
        
        # Check for vague language
        for pattern in self.issue_patterns['vague_language']:
            matches = re.findall(pattern, prompt.lower())
            if matches:
                clarity_issues.append(f"Vague language: {', '.join(set(matches))}")
        
        # Check for unclear instructions
        for pattern in self.issue_patterns['unclear_instructions']:
            matches = re.findall(pattern, prompt.lower())
            if matches:
                clarity_issues.append(f"Unclear instructions: {', '.join(set(matches))}")
        
        # Check sentence complexity
        sentences = re.split(r'[.!?]+', prompt)
        long_sentences = [s for s in sentences if len(s.split()) > 30]
        
        if long_sentences:
            clarity_issues.append(f"Long sentences detected ({len(long_sentences)})")
        
        clarity_score = max(0, 10 - len(clarity_issues) * 2)
        
        return {
            'issues': clarity_issues,
            'score': clarity_score,
            'quality': 'good' if clarity_score >= 7 else 'fair' if clarity_score >= 4 else 'poor'
        }
    
    def _analyze_specificity(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt specificity."""
        specificity_indicators = {
            'numbers': len(re.findall(r'\b\d+\b', prompt)),
            'specific_terms': len(re.findall(r'\b(?:exactly|specifically|precisely|detailed|comprehensive)\b', prompt.lower())),
            'examples_provided': len(re.findall(r'example:|for instance:|such as [^,\n]+', prompt.lower())),
            'format_specified': bool(re.search(r'format:|json|xml|csv|markdown|bullet|list', prompt.lower())),
            'context_provided': bool(re.search(r'context:|background:|scenario:', prompt.lower()))
        }
        
        specificity_score = sum([
            min(2, specificity_indicators['numbers']),
            specificity_indicators['specific_terms'],
            specificity_indicators['examples_provided'] * 2,
            2 if specificity_indicators['format_specified'] else 0,
            1 if specificity_indicators['context_provided'] else 0
        ])
        
        specificity_score = min(10, specificity_score)
        
        return {
            'indicators': specificity_indicators,
            'score': specificity_score,
            'quality': 'good' if specificity_score >= 6 else 'fair' if specificity_score >= 3 else 'poor'
        }
    
    def _analyze_actionability(self, prompt: str) -> Dict[str, Any]:
        """Analyze how actionable the prompt is."""
        action_verbs = re.findall(r'\b(?:write|create|generate|make|build|design|analyze|explain|describe|list|summarize|compare|evaluate)\b', prompt.lower())
        
        actionability_score = min(10, len(set(action_verbs)) * 2)
        
        has_clear_task = bool(action_verbs)
        has_deliverable = bool(re.search(r'output|result|response|answer|solution', prompt.lower()))
        
        return {
            'action_verbs': list(set(action_verbs)),
            'has_clear_task': has_clear_task,
            'has_deliverable': has_deliverable,
            'score': actionability_score,
            'quality': 'good' if actionability_score >= 6 else 'fair' if actionability_score >= 3 else 'poor'
        }
    
    def _analyze_completeness(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt completeness."""
        completeness_elements = {
            'task_definition': bool(re.search(r'^(?:write|create|generate|make|analyze)', prompt.lower())),
            'context_provided': bool(re.search(r'context|background|scenario|situation', prompt.lower())),
            'constraints_specified': bool(re.search(r'must|should|don\'t|limit|maximum|minimum', prompt.lower())),
            'format_specified': bool(re.search(r'format|structure|json|xml|list|bullet', prompt.lower())),
            'examples_included': bool(re.search(r'example:|for instance:', prompt.lower())),
            'success_criteria': bool(re.search(r'good|quality|criteria|requirements', prompt.lower()))
        }
        
        completeness_score = sum(completeness_elements.values()) * 2
        missing_elements = [element for element, present in completeness_elements.items() if not present]
        
        return {
            'elements': completeness_elements,
            'missing_elements': missing_elements,
            'score': completeness_score,
            'quality': 'good' if completeness_score >= 8 else 'fair' if completeness_score >= 5 else 'poor'
        }
    
    def _generate_suggestions(self, prompt: str, analysis: Dict[str, Any]) -> List[ImprovementSuggestion]:
        """Generate improvement suggestions based on analysis."""
        suggestions = []
        
        # Structure suggestions
        if analysis['structure']['score'] < 6:
            if not analysis['structure']['has_sections']:
                suggestions.append(ImprovementSuggestion(
                    category="Structure",
                    issue="Lacks clear organization",
                    suggestion="Break the prompt into clear sections or numbered steps",
                    example="1. Task description\n2. Requirements\n3. Output format",
                    priority="high"
                ))
            
            if not analysis['structure']['has_examples']:
                suggestions.append(ImprovementSuggestion(
                    category="Structure", 
                    issue="No examples provided",
                    suggestion="Add concrete examples to illustrate expectations",
                    example="Example: Input: [sample] → Output: [expected result]",
                    priority="medium"
                ))
        
        # Clarity suggestions
        if analysis['clarity']['score'] < 7:
            for issue in analysis['clarity']['issues']:
                suggestions.append(ImprovementSuggestion(
                    category="Clarity",
                    issue=issue,
                    suggestion="Replace vague terms with specific, concrete language",
                    priority="high"
                ))
        
        # Specificity suggestions
        if analysis['specificity']['score'] < 6:
            if not analysis['specificity']['indicators']['format_specified']:
                suggestions.append(ImprovementSuggestion(
                    category="Specificity",
                    issue="Output format not specified",
                    suggestion="Clearly specify the expected output format",
                    example="Provide response in JSON format with keys: [list keys]",
                    priority="medium"
                ))
            
            if analysis['specificity']['indicators']['examples_provided'] == 0:
                suggestions.append(ImprovementSuggestion(
                    category="Specificity",
                    issue="No concrete examples provided",
                    suggestion="Include specific examples to clarify expectations",
                    priority="medium"
                ))
        
        # Actionability suggestions
        if analysis['actionability']['score'] < 6:
            if not analysis['actionability']['has_clear_task']:
                suggestions.append(ImprovementSuggestion(
                    category="Actionability",
                    issue="Task not clearly defined",
                    suggestion="Start with a clear action verb and specific task description",
                    example="Write a detailed analysis of...",
                    priority="high"
                ))
        
        # Completeness suggestions
        if analysis['completeness']['score'] < 8:
            for missing_element in analysis['completeness']['missing_elements']:
                element_suggestions = {
                    'context_provided': "Add relevant context or background information",
                    'constraints_specified': "Include important constraints and limitations",
                    'format_specified': "Specify the expected output format",
                    'examples_included': "Provide concrete examples",
                    'success_criteria': "Define what constitutes a good response"
                }
                
                if missing_element in element_suggestions:
                    suggestions.append(ImprovementSuggestion(
                        category="Completeness",
                        issue=f"Missing: {missing_element.replace('_', ' ')}",
                        suggestion=element_suggestions[missing_element],
                        priority="medium"
                    ))
        
        return suggestions
    
    def _generate_improved_versions(self, prompt: str, suggestions: List[ImprovementSuggestion]) -> List[ImprovedPrompt]:
        """Generate improved versions using LLM."""
        if not self.api_manager:
            return []
        
        # Prepare issues summary for the LLM
        issues_summary = []
        for suggestion in suggestions:
            issues_summary.append(f"- {suggestion.category}: {suggestion.issue} → {suggestion.suggestion}")
        
        improvement_request = self.improvement_prompt.format(
            original_prompt=prompt,
            issues="\n".join(issues_summary)
        )
        
        try:
            request = LLMRequest(
                prompt=improvement_request,
                max_tokens=1500,
                temperature=0.7
            )
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(self.api_manager.call_llm(request))
                return self._parse_improved_versions(response.content, prompt)
            finally:
                loop.close()
                
        except Exception as e:
            console.print(f"⚠️ LLM improvement failed: {e}", style="yellow")
            return self._create_heuristic_improvements(prompt, suggestions)
    
    def _parse_improved_versions(self, llm_response: str, original_prompt: str) -> List[ImprovedPrompt]:
        """Parse LLM response to extract improved prompt versions."""
        improved_versions = []
        
        # Try to extract structured improvements from the response
        # This is a simplified parser - in practice, you might want more robust parsing
        sections = re.split(r'(?:VERSION|IMPROVED PROMPT)\s*\d+', llm_response, flags=re.IGNORECASE)
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            lines = section.strip().split('\n')
            improved_text = []
            improvements_made = []
            confidence_score = 7.0  # Default
            
            current_section = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if 'improved prompt:' in line.lower() or 'prompt:' in line.lower():
                    current_section = 'prompt'
                elif 'improvements:' in line.lower() or 'changes:' in line.lower():
                    current_section = 'improvements'
                elif 'confidence:' in line.lower() or 'score:' in line.lower():
                    # Extract confidence score
                    score_match = re.search(r'(\d+(?:\.\d+)?)', line)
                    if score_match:
                        confidence_score = float(score_match.group(1))
                        if confidence_score > 10:
                            confidence_score = confidence_score / 10  # Normalize if out of 10
                elif current_section == 'prompt':
                    improved_text.append(line)
                elif current_section == 'improvements':
                    if line.startswith(('-', '•', '*')):
                        improvements_made.append(line[1:].strip())
                    else:
                        improvements_made.append(line)
            
            if improved_text:
                improved_versions.append(ImprovedPrompt(
                    original=original_prompt,
                    improved='\n'.join(improved_text),
                    improvements_made=improvements_made[:5],  # Limit to 5 improvements
                    confidence_score=min(10.0, confidence_score)
                ))
            
            if len(improved_versions) >= 3:  # Limit to 3 versions
                break
        
        return improved_versions
    
    def _create_heuristic_improvements(self, prompt: str, suggestions: List[ImprovementSuggestion]) -> List[ImprovedPrompt]:
        """Create improved versions using heuristic methods."""
        improved_versions = []
        
        # Version 1: Add structure and examples
        structured_prompt = self._add_structure(prompt, suggestions)
        improvements_made = ["Added clear structure and organization", "Included specific examples"]
        
        improved_versions.append(ImprovedPrompt(
            original=prompt,
            improved=structured_prompt,
            improvements_made=improvements_made,
            confidence_score=7.0
        ))
        
        # Version 2: Focus on clarity and specificity
        clarified_prompt = self._improve_clarity(structured_prompt, suggestions)
        improvements_made = ["Replaced vague language with specific terms", "Added clear constraints"]
        
        improved_versions.append(ImprovedPrompt(
            original=prompt,
            improved=clarified_prompt,
            improvements_made=improvements_made,
            confidence_score=6.5
        ))
        
        return improved_versions
    
    def _add_structure(self, prompt: str, suggestions: List[ImprovementSuggestion]) -> str:
        """Add structure to prompt heuristically."""
        # Simple heuristic improvement by adding sections
        if not re.search(r'\n\s*[-*]\s+|\n\s*\d+\.\s+', prompt):
            # Add basic structure
            structured = f"""Task: {prompt}

Requirements:
- Provide a comprehensive response
- Use clear and specific language
- Include relevant examples where appropriate

Output Format:
- Structure your response clearly
- Use proper formatting
"""
            return structured
        return prompt
    
    def _improve_clarity(self, prompt: str, suggestions: List[ImprovementSuggestion]) -> str:
        """Improve clarity heuristically."""
        improved = prompt
        
        # Replace common vague terms
        vague_replacements = {
            r'\bsome\b': 'specific',
            r'\bmany\b': 'multiple',
            r'\bvarious\b': 'different',
            r'\bthing\b': 'element',
            r'\bstuff\b': 'content',
            r'\bgood\b': 'high-quality',
            r'\bnice\b': 'well-structured'
        }
        
        for pattern, replacement in vague_replacements.items():
            improved = re.sub(pattern, replacement, improved, flags=re.IGNORECASE)
        
        return improved
    
    def _generate_overall_assessment(self, analysis: Dict[str, Any], suggestions: List[ImprovementSuggestion]) -> str:
        """Generate overall assessment of the prompt."""
        scores = [
            analysis['structure']['score'],
            analysis['clarity']['score'],
            analysis['specificity']['score'],
            analysis['actionability']['score'],
            analysis['completeness']['score']
        ]
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 8.0:
            assessment = "Excellent prompt quality"
        elif avg_score >= 6.0:
            assessment = "Good prompt with minor improvements needed"
        elif avg_score >= 4.0:
            assessment = "Fair prompt requiring moderate improvements"
        else:
            assessment = "Poor prompt needing significant improvements"
        
        priority_issues = [s for s in suggestions if s.priority == "high"]
        if priority_issues:
            assessment += f" ({len(priority_issues)} high-priority issues identified)"
        
        return assessment
    
    def _generate_recommendations(self, suggestions: List[ImprovementSuggestion]) -> List[str]:
        """Generate high-level recommendations."""
        recommendations = []
        
        # Group suggestions by category
        by_category = {}
        for suggestion in suggestions:
            if suggestion.category not in by_category:
                by_category[suggestion.category] = []
            by_category[suggestion.category].append(suggestion)
        
        # Generate category-based recommendations
        for category, cat_suggestions in by_category.items():
            high_priority = [s for s in cat_suggestions if s.priority == "high"]
            if high_priority:
                recommendations.append(f"Address {category.lower()} issues: {high_priority[0].suggestion}")
        
        # Add general recommendations
        if len(suggestions) > 5:
            recommendations.append("Consider breaking complex prompts into simpler, focused tasks")
        
        if any("example" in s.suggestion.lower() for s in suggestions):
            recommendations.append("Add concrete examples to clarify expectations")
        
        return recommendations[:6]  # Limit to 6 recommendations
    
    def _display_improvement_report(self, report: ImprovementReport) -> None:
        """Display improvement report in rich format."""
        console.print("\n🔧 Prompt Improvement Analysis", style="bold blue")
        console.print("="*60)
        
        # Overall assessment
        assessment_color = "green" if "excellent" in report.overall_assessment.lower() else "yellow" if "good" in report.overall_assessment.lower() else "red"
        console.print(f"\n📊 {report.overall_assessment}", style=f"bold {assessment_color}")
        
        # Analysis summary
        analysis_table = Table(show_header=True, header_style="bold magenta")
        analysis_table.add_column("Dimension", style="cyan")
        analysis_table.add_column("Score", style="white")
        analysis_table.add_column("Quality", style="white")
        
        dimensions = ['structure', 'clarity', 'specificity', 'actionability', 'completeness']
        for dim in dimensions:
            if dim in report.analysis:
                dim_data = report.analysis[dim]
                score = dim_data.get('score', 0)
                quality = dim_data.get('quality', 'unknown')
                
                analysis_table.add_row(
                    dim.title(),
                    f"{score}/10",
                    quality.title()
                )
        
        console.print("\n", analysis_table)
        
        # Key suggestions
        if report.suggestions:
            console.print("\n🎯 Key Improvement Areas:", style="bold yellow")
            high_priority = [s for s in report.suggestions if s.priority == "high"]
            medium_priority = [s for s in report.suggestions if s.priority == "medium"]
            
            for suggestion in high_priority[:3]:  # Show top 3 high priority
                console.print(f"  🔴 {suggestion.category}: {suggestion.issue}")
                console.print(f"     → {suggestion.suggestion}")
                if suggestion.example:
                    console.print(f"     Example: {suggestion.example}", style="dim")
                console.print()
            
            for suggestion in medium_priority[:2]:  # Show top 2 medium priority
                console.print(f"  🟡 {suggestion.category}: {suggestion.issue}")
                console.print(f"     → {suggestion.suggestion}")
                console.print()
        
        # Improved versions
        if report.improved_versions:
            console.print("\n✨ Improved Versions:", style="bold green")
            
            for i, improved in enumerate(report.improved_versions, 1):
                # Create panels for original and improved
                original_panel = Panel(
                    report.original_prompt[:300] + "..." if len(report.original_prompt) > 300 else report.original_prompt,
                    title="Original",
                    border_style="red"
                )
                
                improved_panel = Panel(
                    improved.improved[:300] + "..." if len(improved.improved) > 300 else improved.improved,
                    title=f"Version {i} (Confidence: {improved.confidence_score}/10)",
                    border_style="green"
                )
                
                console.print(f"\n📝 Improved Version {i}:")
                console.print(Columns([original_panel, improved_panel]))
                
                if improved.improvements_made:
                    console.print("Improvements made:", style="bold")
                    for improvement in improved.improvements_made:
                        console.print(f"  • {improvement}")
                console.print()
        
        # Recommendations
        if report.recommendations:
            console.print("💡 Recommendations:", style="bold green")
            for i, rec in enumerate(report.recommendations, 1):
                console.print(f"  {i}. {rec}")
        
        console.print("\n" + "="*60)
    
    def export_improvement(self, report: ImprovementReport, output_path: str, format_type: str = "json") -> None:
        """
        Export improvement report to file.
        
        Args:
            report: Improvement report to export
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
                    f.write("PROMPT IMPROVEMENT REPORT\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Assessment: {report.overall_assessment}\n\n")
                    
                    f.write("ORIGINAL PROMPT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(report.original_prompt + "\n\n")
                    
                    if report.improved_versions:
                        for i, improved in enumerate(report.improved_versions, 1):
                            f.write(f"IMPROVED VERSION {i}:\n")
                            f.write("-" * 20 + "\n")
                            f.write(improved.improved + "\n\n")
                            f.write("Improvements made:\n")
                            for improvement in improved.improvements_made:
                                f.write(f"- {improvement}\n")
                            f.write(f"Confidence: {improved.confidence_score}/10\n\n")
                    
                    f.write("RECOMMENDATIONS:\n")
                    for i, rec in enumerate(report.recommendations, 1):
                        f.write(f"{i}. {rec}\n")
            
            console.print(f"✅ Improvement report exported to {output_path}", style="green")
            
        except Exception as e:
            console.print(f"❌ Failed to export improvement report: {e}", style="red")