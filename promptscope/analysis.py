"""
Prompt analysis module for extracting patterns, structure, and characteristics.

Analyzes prompts from JSONL logs to identify patterns, repetition, length distribution,
tone, structure, and other key metrics for prompt optimization.
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import jsonlines
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, track

from .apis import APIManager, LLMRequest

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

console = Console()

@dataclass
class PromptMetrics:
    """Individual prompt analysis metrics."""
    length_chars: int
    length_words: int
    length_sentences: int
    complexity_score: float
    sentiment_score: float
    readability_score: float
    structure_type: str
    patterns: List[str]
    entities: List[str]
    keywords: List[str]

@dataclass
class AnalysisReport:
    """Complete analysis report for a set of prompts."""
    total_prompts: int
    avg_length_chars: float
    avg_length_words: float
    avg_complexity: float
    sentiment_distribution: Dict[str, int]
    common_patterns: List[tuple]
    structure_types: Dict[str, int]
    top_keywords: List[tuple]
    recommendations: List[str]
    prompt_metrics: List[PromptMetrics]

class PromptAnalyzer:
    """
    Analyzes prompts to extract patterns, structure, and optimization opportunities.
    
    Provides comprehensive analysis of prompt characteristics including:
    - Length and complexity metrics
    - Sentiment and tone analysis
    - Structural pattern detection
    - Keyword and entity extraction
    - Optimization recommendations
    """
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        """
        Initialize prompt analyzer.
        
        Args:
            api_manager: Optional API manager for advanced analysis
        """
        self.api_manager = api_manager
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Common prompt patterns
        self.patterns = {
            'question': [r'\?', r'^what', r'^how', r'^why', r'^when', r'^where', r'^who'],
            'instruction': [r'^(please |)(?:write|create|generate|make|build)', 
                           r'^(you are|act as|assume the role)', r'^(do |)(not |)(?:include|exclude|avoid)'],
            'few_shot': [r'example\s*\d*:', r'input:|output:', r'q:|a:', r'question:|answer:'],
            'role_play': [r'you are (a|an)', r'act as', r'assume.*role', r'pretend'],
            'constraint': [r'must (not |)', r'should (not |)', r'do not', r'avoid', r'only', r'exactly'],
            'format': [r'format:', r'json', r'xml', r'csv', r'markdown', r'bullet', r'list'],
            'reasoning': [r'think step by step', r'reason through', r'explain.*reasoning', r'because'],
            'conditional': [r'if.*then', r'unless', r'when.*do', r'in case']
        }
    
    def analyze_file(self, file_path: str, format_type: str = "jsonl", **kwargs) -> AnalysisReport:
        """
        Analyze prompts from a file.
        
        Args:
            file_path: Path to file containing prompts
            format_type: File format ('jsonl', 'json', 'txt')
            **kwargs: Additional options
            
        Returns:
            Complete analysis report
        """
        prompts = self._load_prompts(file_path, format_type)
        
        if not prompts:
            console.print("❌ No prompts found in file", style="red")
            return self._empty_report()
        
        console.print(f"📊 Analyzing {len(prompts)} prompts...", style="blue")
        
        return self.analyze_prompts(prompts, **kwargs)
    
    def analyze_prompts(self, prompts: List[str], **kwargs) -> AnalysisReport:
        """
        Analyze a list of prompts.
        
        Args:
            prompts: List of prompt texts
            **kwargs: Additional analysis options
            
        Returns:
            Complete analysis report
        """
        prompt_metrics = []
        
        # Analyze each prompt individually
        for prompt in track(prompts, description="Analyzing prompts..."):
            metrics = self._analyze_single_prompt(prompt)
            prompt_metrics.append(metrics)
        
        # Generate aggregate statistics
        report = self._generate_report(prompt_metrics, prompts)
        
        # Display summary
        self._display_summary(report)
        
        return report
    
    def _load_prompts(self, file_path: str, format_type: str) -> List[str]:
        """Load prompts from file."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            console.print(f"❌ File not found: {file_path}", style="red")
            return []
        
        prompts = []
        
        try:
            if format_type == "jsonl":
                with jsonlines.open(file_path) as reader:
                    for item in reader:
                        if isinstance(item, dict):
                            # Extract prompt from common fields
                            prompt = item.get('prompt') or item.get('input') or item.get('text') or str(item)
                        else:
                            prompt = str(item)
                        prompts.append(prompt)
            
            elif format_type == "json":
                with open(file_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        prompts = [str(item) for item in data]
                    else:
                        prompts = [str(data)]
            
            elif format_type == "txt":
                with open(file_path) as f:
                    content = f.read().strip()
                    # Split by double newlines for multiple prompts, or use entire content
                    if '\n\n' in content:
                        prompts = [p.strip() for p in content.split('\n\n') if p.strip()]
                    else:
                        prompts = [content]
            
            else:
                console.print(f"❌ Unsupported format: {format_type}", style="red")
                return []
        
        except Exception as e:
            console.print(f"❌ Error loading file: {e}", style="red")
            return []
        
        return prompts
    
    def _analyze_single_prompt(self, prompt: str) -> PromptMetrics:
        """Analyze individual prompt for all metrics."""
        # Basic length metrics
        length_chars = len(prompt)
        words = word_tokenize(prompt.lower())
        length_words = len(words)
        sentences = sent_tokenize(prompt)
        length_sentences = len(sentences)
        
        # Complexity score (based on vocabulary diversity and sentence structure)
        complexity_score = self._calculate_complexity(prompt, words, sentences)
        
        # Sentiment analysis
        sentiment_score = self.sia.polarity_scores(prompt)['compound']
        
        # Readability score (simplified Flesch-Kincaid)
        readability_score = self._calculate_readability(words, sentences)
        
        # Structure type detection
        structure_type = self._detect_structure_type(prompt)
        
        # Pattern detection
        patterns = self._detect_patterns(prompt)
        
        # Entity extraction (simplified)
        entities = self._extract_entities(prompt)
        
        # Keyword extraction
        keywords = self._extract_keywords(prompt, words)
        
        return PromptMetrics(
            length_chars=length_chars,
            length_words=length_words,
            length_sentences=length_sentences,
            complexity_score=complexity_score,
            sentiment_score=sentiment_score,
            readability_score=readability_score,
            structure_type=structure_type,
            patterns=patterns,
            entities=entities,
            keywords=keywords
        )
    
    def _calculate_complexity(self, prompt: str, words: List[str], sentences: List[str]) -> float:
        """Calculate prompt complexity score (0-1)."""
        if not words:
            return 0.0
        
        # Vocabulary diversity
        unique_words = len(set(words))
        diversity_ratio = unique_words / len(words)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence length variation
        sentence_lengths = [len(word_tokenize(sent)) for sent in sentences]
        sentence_variation = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Combine metrics (normalized to 0-1)
        complexity = min(1.0, (
            diversity_ratio * 0.4 +
            min(1.0, avg_word_length / 10) * 0.3 +
            min(1.0, sentence_variation / 20) * 0.3
        ))
        
        return round(complexity, 3)
    
    def _calculate_readability(self, words: List[str], sentences: List[str]) -> float:
        """Calculate simplified readability score (0-100, higher = easier)."""
        if not words or not sentences:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simplified Flesch formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
        
        return max(0, min(100, round(score, 1)))
    
    def _detect_structure_type(self, prompt: str) -> str:
        """Detect the structural type of the prompt."""
        prompt_lower = prompt.lower()
        
        # Check for different structural patterns
        if any(pattern in prompt_lower for pattern in ['example:', 'input:', 'output:', 'q:', 'a:']):
            return "few_shot"
        elif re.search(r'^(you are|act as|assume)', prompt_lower):
            return "role_play"
        elif '?' in prompt:
            return "question"
        elif re.search(r'^(write|create|generate|make|build)', prompt_lower):
            return "instruction"
        elif re.search(r'(step by step|first.*then|1\.|2\.)', prompt_lower):
            return "structured"
        elif len(sent_tokenize(prompt)) == 1:
            return "simple"
        else:
            return "complex"
    
    def _detect_patterns(self, prompt: str) -> List[str]:
        """Detect common prompt patterns."""
        detected = []
        prompt_lower = prompt.lower()
        
        for pattern_type, regexes in self.patterns.items():
            for regex in regexes:
                if re.search(regex, prompt_lower):
                    detected.append(pattern_type)
                    break
        
        return detected
    
    def _extract_entities(self, prompt: str) -> List[str]:
        """Extract entities (simplified - proper nouns and capitalized words)."""
        # Simple entity extraction based on capitalization
        words = word_tokenize(prompt)
        entities = []
        
        for word in words:
            if word[0].isupper() and len(word) > 2 and word.isalpha():
                entities.append(word)
        
        return list(set(entities))[:10]  # Limit to top 10
    
    def _extract_keywords(self, prompt: str, words: List[str]) -> List[str]:
        """Extract key terms from prompt."""
        # Filter out stop words and short words
        keywords = [
            word for word in words 
            if word.lower() not in self.stop_words 
            and len(word) > 3 
            and word.isalpha()
        ]
        
        # Count frequency and return top keywords
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(10)]
    
    def _generate_report(self, prompt_metrics: List[PromptMetrics], prompts: List[str]) -> AnalysisReport:
        """Generate comprehensive analysis report."""
        if not prompt_metrics:
            return self._empty_report()
        
        total_prompts = len(prompt_metrics)
        
        # Aggregate metrics
        avg_length_chars = statistics.mean([m.length_chars for m in prompt_metrics])
        avg_length_words = statistics.mean([m.length_words for m in prompt_metrics])
        avg_complexity = statistics.mean([m.complexity_score for m in prompt_metrics])
        
        # Sentiment distribution
        sentiment_distribution = {
            "positive": sum(1 for m in prompt_metrics if m.sentiment_score > 0.1),
            "neutral": sum(1 for m in prompt_metrics if -0.1 <= m.sentiment_score <= 0.1),
            "negative": sum(1 for m in prompt_metrics if m.sentiment_score < -0.1)
        }
        
        # Common patterns
        all_patterns = []
        for m in prompt_metrics:
            all_patterns.extend(m.patterns)
        common_patterns = Counter(all_patterns).most_common(10)
        
        # Structure types
        structure_types = Counter([m.structure_type for m in prompt_metrics])
        
        # Top keywords
        all_keywords = []
        for m in prompt_metrics:
            all_keywords.extend(m.keywords)
        top_keywords = Counter(all_keywords).most_common(20)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(prompt_metrics, prompts)
        
        return AnalysisReport(
            total_prompts=total_prompts,
            avg_length_chars=round(avg_length_chars, 1),
            avg_length_words=round(avg_length_words, 1),
            avg_complexity=round(avg_complexity, 3),
            sentiment_distribution=sentiment_distribution,
            common_patterns=common_patterns,
            structure_types=dict(structure_types),
            top_keywords=top_keywords,
            recommendations=recommendations,
            prompt_metrics=prompt_metrics
        )
    
    def _generate_recommendations(self, prompt_metrics: List[PromptMetrics], prompts: List[str]) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Length recommendations
        avg_length = statistics.mean([m.length_words for m in prompt_metrics])
        if avg_length > 200:
            recommendations.append("Consider breaking down long prompts into smaller, focused instructions")
        elif avg_length < 10:
            recommendations.append("Prompts may be too brief - consider adding more context or examples")
        
        # Complexity recommendations
        avg_complexity = statistics.mean([m.complexity_score for m in prompt_metrics])
        if avg_complexity < 0.3:
            recommendations.append("Prompts may benefit from more varied vocabulary and structure")
        elif avg_complexity > 0.8:
            recommendations.append("Consider simplifying language for better model comprehension")
        
        # Structure recommendations
        structure_counts = Counter([m.structure_type for m in prompt_metrics])
        if structure_counts.get("simple", 0) > len(prompt_metrics) * 0.7:
            recommendations.append("Consider using more structured prompts with examples or step-by-step instructions")
        
        # Pattern recommendations
        all_patterns = []
        for m in prompt_metrics:
            all_patterns.extend(m.patterns)
        pattern_counts = Counter(all_patterns)
        
        if pattern_counts.get("few_shot", 0) == 0 and len(prompt_metrics) > 5:
            recommendations.append("Consider adding few-shot examples to improve response quality")
        
        if pattern_counts.get("reasoning", 0) < len(prompt_metrics) * 0.2:
            recommendations.append("Add explicit reasoning instructions (e.g., 'think step by step') for complex tasks")
        
        # Sentiment recommendations
        negative_count = sum(1 for m in prompt_metrics if m.sentiment_score < -0.1)
        if negative_count > len(prompt_metrics) * 0.3:
            recommendations.append("Consider using more positive, encouraging language in prompts")
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def _empty_report(self) -> AnalysisReport:
        """Return empty analysis report."""
        return AnalysisReport(
            total_prompts=0,
            avg_length_chars=0.0,
            avg_length_words=0.0,
            avg_complexity=0.0,
            sentiment_distribution={"positive": 0, "neutral": 0, "negative": 0},
            common_patterns=[],
            structure_types={},
            top_keywords=[],
            recommendations=[],
            prompt_metrics=[]
        )
    
    def _display_summary(self, report: AnalysisReport) -> None:
        """Display analysis summary in rich format."""
        console.print("\n📊 Prompt Analysis Summary", style="bold blue")
        console.print("="*50)
        
        # Basic metrics table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Total Prompts", str(report.total_prompts))
        table.add_row("Avg Length (chars)", f"{report.avg_length_chars:.1f}")
        table.add_row("Avg Length (words)", f"{report.avg_length_words:.1f}")
        table.add_row("Avg Complexity", f"{report.avg_complexity:.3f}")
        
        console.print(table)
        
        # Sentiment distribution
        console.print("\n🎭 Sentiment Distribution:", style="bold")
        for sentiment, count in report.sentiment_distribution.items():
            percentage = (count / report.total_prompts) * 100 if report.total_prompts > 0 else 0
            console.print(f"  {sentiment.title()}: {count} ({percentage:.1f}%)")
        
        # Top patterns
        if report.common_patterns:
            console.print("\n🔍 Common Patterns:", style="bold")
            for pattern, count in report.common_patterns[:5]:
                console.print(f"  {pattern}: {count}")
        
        # Structure types
        if report.structure_types:
            console.print("\n🏗️ Structure Types:", style="bold")
            for structure, count in report.structure_types.items():
                console.print(f"  {structure}: {count}")
        
        # Recommendations
        if report.recommendations:
            console.print("\n💡 Recommendations:", style="bold green")
            for i, rec in enumerate(report.recommendations, 1):
                console.print(f"  {i}. {rec}")
        
        console.print("\n" + "="*50)
    
    def export_report(self, report: AnalysisReport, output_path: str, format_type: str = "json") -> None:
        """
        Export analysis report to file.
        
        Args:
            report: Analysis report to export
            output_path: Output file path
            format_type: Export format ('json', 'csv', 'txt')
        """
        output_path = Path(output_path)
        
        try:
            if format_type == "json":
                with open(output_path, 'w') as f:
                    json.dump(asdict(report), f, indent=2, default=str)
            
            elif format_type == "csv":
                # Export basic metrics as CSV
                df_data = []
                for i, metrics in enumerate(report.prompt_metrics):
                    row = {
                        'prompt_id': i,
                        'length_chars': metrics.length_chars,
                        'length_words': metrics.length_words,
                        'complexity_score': metrics.complexity_score,
                        'sentiment_score': metrics.sentiment_score,
                        'structure_type': metrics.structure_type,
                        'patterns': ','.join(metrics.patterns),
                        'top_keywords': ','.join(metrics.keywords[:5])
                    }
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
            
            elif format_type == "txt":
                with open(output_path, 'w') as f:
                    f.write("PROMPT ANALYSIS REPORT\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Total Prompts: {report.total_prompts}\n")
                    f.write(f"Average Length: {report.avg_length_words:.1f} words\n")
                    f.write(f"Average Complexity: {report.avg_complexity:.3f}\n\n")
                    
                    f.write("RECOMMENDATIONS:\n")
                    for i, rec in enumerate(report.recommendations, 1):
                        f.write(f"{i}. {rec}\n")
            
            console.print(f"✅ Report exported to {output_path}", style="green")
            
        except Exception as e:
            console.print(f"❌ Failed to export report: {e}", style="red")