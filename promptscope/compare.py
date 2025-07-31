"""
Prompt comparison module for semantic and token-level analysis.

Provides comprehensive comparison between prompts including semantic similarity,
token-level differences, structural analysis, and deterministic scoring.
"""

import re
import json
import difflib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from rich.console import Console
from rich.table import Table
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text

from .apis import APIManager, LLMRequest

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

console = Console()

@dataclass
class TokenDiff:
    """Token-level difference information."""
    added: List[str]
    removed: List[str]
    modified: List[Tuple[str, str]]
    unchanged: List[str]

@dataclass
class SemanticSimilarity:
    """Semantic similarity analysis results."""
    similarity_score: float
    embedding_distance: float
    intent_similarity: float
    topic_overlap: float
    key_concepts: Dict[str, List[str]]

@dataclass
class StructuralComparison:
    """Structural comparison results."""
    length_diff: int
    sentence_count_diff: int
    complexity_diff: float
    structure_similarity: float
    format_similarity: float

@dataclass
class ComparisonResult:
    """Complete comparison result between two prompts."""
    prompt1: str
    prompt2: str
    token_diff: TokenDiff
    semantic_similarity: SemanticSimilarity
    structural_comparison: StructuralComparison
    overall_similarity: float
    summary: str
    tone_shift: str
    recommendations: List[str]

class PromptComparator:
    """
    Comprehensive prompt comparison tool.
    
    Provides token-level, semantic, and structural comparison between prompts
    with deterministic scoring and actionable insights.
    """
    
    def __init__(self, api_manager: Optional[APIManager] = None):
        """
        Initialize prompt comparator.
        
        Args:
            api_manager: Optional API manager for advanced semantic analysis
        """
        self.api_manager = api_manager
        
        # Initialize sentence transformer for semantic analysis
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            console.print("✅ Sentence transformer loaded", style="green")
        except Exception as e:
            console.print(f"⚠️ Sentence transformer not available: {e}", style="yellow")
            self.sentence_model = None
        
        # Initialize TF-IDF vectorizer
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
    
    def compare(self, prompt1: str, prompt2: str, **kwargs) -> ComparisonResult:
        """
        Compare two prompts comprehensively.
        
        Args:
            prompt1: First prompt to compare
            prompt2: Second prompt to compare
            **kwargs: Additional comparison options
            
        Returns:
            Complete comparison result
        """
        console.print("🔍 Comparing prompts...", style="blue")
        
        # Token-level comparison
        token_diff = self._compare_tokens(prompt1, prompt2)
        
        # Semantic similarity analysis
        semantic_similarity = self._analyze_semantic_similarity(prompt1, prompt2)
        
        # Structural comparison
        structural_comparison = self._compare_structure(prompt1, prompt2)
        
        # Calculate overall similarity
        overall_similarity = self._calculate_overall_similarity(
            token_diff, semantic_similarity, structural_comparison
        )
        
        # Generate summary and recommendations
        summary = self._generate_summary(token_diff, semantic_similarity, structural_comparison)
        tone_shift = self._analyze_tone_shift(prompt1, prompt2)
        recommendations = self._generate_recommendations(
            token_diff, semantic_similarity, structural_comparison
        )
        
        result = ComparisonResult(
            prompt1=prompt1,
            prompt2=prompt2,
            token_diff=token_diff,
            semantic_similarity=semantic_similarity,
            structural_comparison=structural_comparison,
            overall_similarity=overall_similarity,
            summary=summary,
            tone_shift=tone_shift,
            recommendations=recommendations
        )
        
        # Display results if requested
        if kwargs.get('display', True):
            self._display_comparison_results(result, **kwargs)
        
        return result
    
    def compare_files(self, file1: str, file2: str) -> ComparisonResult:
        """
        Compare prompts from two files.
        
        Args:
            file1: Path to first prompt file
            file2: Path to second prompt file
            
        Returns:
            Comparison result
        """
        try:
            prompt1 = Path(file1).read_text().strip()
            prompt2 = Path(file2).read_text().strip()
            return self.compare(prompt1, prompt2)
        except Exception as e:
            console.print(f"❌ Error reading files: {e}", style="red")
            raise
    
    def _compare_tokens(self, prompt1: str, prompt2: str) -> TokenDiff:
        """Compare prompts at token level."""
        tokens1 = word_tokenize(prompt1.lower())
        tokens2 = word_tokenize(prompt2.lower())
        
        # Use difflib for sequence comparison
        differ = difflib.SequenceMatcher(None, tokens1, tokens2)
        
        added = []
        removed = []
        modified = []
        unchanged = []
        
        for tag, i1, i2, j1, j2 in differ.get_opcodes():
            if tag == 'equal':
                unchanged.extend(tokens1[i1:i2])
            elif tag == 'delete':
                removed.extend(tokens1[i1:i2])
            elif tag == 'insert':
                added.extend(tokens2[j1:j2])
            elif tag == 'replace':
                # Treat as modification
                old_tokens = tokens1[i1:i2]
                new_tokens = tokens2[j1:j2]
                for old, new in zip(old_tokens, new_tokens):
                    modified.append((old, new))
                # Handle length differences
                if len(old_tokens) > len(new_tokens):
                    removed.extend(old_tokens[len(new_tokens):])
                elif len(new_tokens) > len(old_tokens):
                    added.extend(new_tokens[len(old_tokens):])
        
        return TokenDiff(
            added=list(set(added)),
            removed=list(set(removed)),
            modified=modified,
            unchanged=list(set(unchanged))
        )
    
    def _analyze_semantic_similarity(self, prompt1: str, prompt2: str) -> SemanticSimilarity:
        """Analyze semantic similarity between prompts."""
        # Calculate TF-IDF similarity
        tfidf_similarity = self._calculate_tfidf_similarity(prompt1, prompt2)
        
        # Calculate sentence embedding similarity if available
        embedding_similarity = 0.0
        embedding_distance = 1.0
        
        if self.sentence_model:
            embeddings = self.sentence_model.encode([prompt1, prompt2])
            embedding_similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            embedding_distance = 1.0 - embedding_similarity
        
        # Calculate intent similarity (based on question words, action verbs, etc.)
        intent_similarity = self._calculate_intent_similarity(prompt1, prompt2)
        
        # Calculate topic overlap
        topic_overlap = self._calculate_topic_overlap(prompt1, prompt2)
        
        # Extract key concepts
        key_concepts = self._extract_key_concepts(prompt1, prompt2)
        
        # Overall semantic similarity (weighted average)
        similarity_score = (
            tfidf_similarity * 0.3 +
            embedding_similarity * 0.4 +
            intent_similarity * 0.2 +
            topic_overlap * 0.1
        )
        
        return SemanticSimilarity(
            similarity_score=round(similarity_score, 3),
            embedding_distance=round(embedding_distance, 3),
            intent_similarity=round(intent_similarity, 3),
            topic_overlap=round(topic_overlap, 3),
            key_concepts=key_concepts
        )
    
    def _calculate_tfidf_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate TF-IDF cosine similarity."""
        try:
            tfidf_matrix = self.tfidf.fit_transform([prompt1, prompt2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            return 0.0
    
    def _calculate_intent_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity based on intent markers."""
        intent_patterns = {
            'question': [r'\?', r'^what', r'^how', r'^why', r'^when', r'^where', r'^who'],
            'instruction': [r'^(please |)(?:write|create|generate|make|build)', 
                           r'^(you are|act as)', r'^(do |)(not |)(?:include|exclude)'],
            'analysis': [r'analyze', r'examine', r'review', r'evaluate', r'assess'],
            'creative': [r'imagine', r'create', r'design', r'invent', r'brainstorm'],
            'explanation': [r'explain', r'describe', r'clarify', r'elaborate', r'detail']
        }
        
        def get_intent_vector(prompt):
            vector = []
            prompt_lower = prompt.lower()
            for intent_type, patterns in intent_patterns.items():
                score = sum(1 for pattern in patterns if re.search(pattern, prompt_lower))
                vector.append(score)
            return np.array(vector)
        
        vector1 = get_intent_vector(prompt1)
        vector2 = get_intent_vector(prompt2)
        
        # Cosine similarity of intent vectors
        if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
            return 0.0
        
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return float(similarity)
    
    def _calculate_topic_overlap(self, prompt1: str, prompt2: str) -> float:
        """Calculate topic overlap using key noun phrases."""
        def extract_noun_phrases(text):
            # Simple noun phrase extraction
            words = word_tokenize(text.lower())
            # Filter for nouns and adjectives (simplified)
            content_words = [word for word in words if len(word) > 3 and word.isalpha()]
            return set(content_words)
        
        phrases1 = extract_noun_phrases(prompt1)
        phrases2 = extract_noun_phrases(prompt2)
        
        if not phrases1 and not phrases2:
            return 1.0
        if not phrases1 or not phrases2:
            return 0.0
        
        intersection = len(phrases1.intersection(phrases2))
        union = len(phrases1.union(phrases2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_key_concepts(self, prompt1: str, prompt2: str) -> Dict[str, List[str]]:
        """Extract key concepts from both prompts."""
        def extract_concepts(text):
            words = word_tokenize(text.lower())
            # Extract meaningful words (simplified)
            concepts = [word for word in words if len(word) > 4 and word.isalpha()]
            return list(set(concepts))[:10]  # Top 10 concepts
        
        return {
            'prompt1_concepts': extract_concepts(prompt1),
            'prompt2_concepts': extract_concepts(prompt2),
            'shared_concepts': list(set(extract_concepts(prompt1)).intersection(set(extract_concepts(prompt2))))
        }
    
    def _compare_structure(self, prompt1: str, prompt2: str) -> StructuralComparison:
        """Compare structural aspects of prompts."""
        # Length differences
        length_diff = len(prompt2) - len(prompt1)
        
        # Sentence count differences
        sentences1 = nltk.sent_tokenize(prompt1)
        sentences2 = nltk.sent_tokenize(prompt2)
        sentence_count_diff = len(sentences2) - len(sentences1)
        
        # Complexity differences (vocabulary diversity)
        def calculate_complexity(text):
            words = word_tokenize(text.lower())
            if not words:
                return 0
            unique_words = len(set(words))
            return unique_words / len(words)
        
        complexity1 = calculate_complexity(prompt1)
        complexity2 = calculate_complexity(prompt2)
        complexity_diff = complexity2 - complexity1
        
        # Structure similarity (formatting patterns)
        structure_similarity = self._calculate_structure_similarity(prompt1, prompt2)
        
        # Format similarity (punctuation, capitalization)
        format_similarity = self._calculate_format_similarity(prompt1, prompt2)
        
        return StructuralComparison(
            length_diff=length_diff,
            sentence_count_diff=sentence_count_diff,
            complexity_diff=round(complexity_diff, 3),
            structure_similarity=round(structure_similarity, 3),
            format_similarity=round(format_similarity, 3)
        )
    
    def _calculate_structure_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate structural similarity based on formatting patterns."""
        patterns = [
            r'\n\n',  # Paragraph breaks
            r'\n-',   # List items
            r'\n\d+\.',  # Numbered lists
            r'[A-Z][^.!?]*[.!?]',  # Sentences
            r'[A-Z][^:]*:',  # Headers/labels
            r'"[^"]*"',  # Quoted text
            r'\([^)]*\)',  # Parenthetical
        ]
        
        def get_structure_vector(text):
            return [len(re.findall(pattern, text)) for pattern in patterns]
        
        vector1 = np.array(get_structure_vector(prompt1))
        vector2 = np.array(get_structure_vector(prompt2))
        
        # Normalize vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 and norm2 == 0:
            return 1.0
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vector1, vector2) / (norm1 * norm2)
        return float(similarity)
    
    def _calculate_format_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate formatting similarity."""
        def get_format_features(text):
            return {
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0,
                'whitespace_ratio': sum(1 for c in text if c.isspace()) / len(text) if text else 0,
                'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            }
        
        features1 = get_format_features(prompt1)
        features2 = get_format_features(prompt2)
        
        # Calculate similarity for each feature
        similarities = []
        for key in features1:
            val1, val2 = features1[key], features2[key]
            if val1 == 0 and val2 == 0:
                similarities.append(1.0)
            else:
                similarity = 1 - abs(val1 - val2) / max(val1, val2, 0.01)
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities)
    
    def _calculate_overall_similarity(self, token_diff: TokenDiff, 
                                      semantic_similarity: SemanticSimilarity,
                                      structural_comparison: StructuralComparison) -> float:
        """Calculate overall similarity score."""
        # Token-level similarity
        total_tokens = len(token_diff.added) + len(token_diff.removed) + len(token_diff.unchanged) + len(token_diff.modified)
        token_similarity = len(token_diff.unchanged) / total_tokens if total_tokens > 0 else 0
        
        # Structural similarity (inverse of normalized differences)
        struct_similarity = (structural_comparison.structure_similarity + structural_comparison.format_similarity) / 2
        
        # Weighted overall similarity
        overall = (
            semantic_similarity.similarity_score * 0.5 +
            token_similarity * 0.3 +
            struct_similarity * 0.2
        )
        
        return round(overall, 3)
    
    def _analyze_tone_shift(self, prompt1: str, prompt2: str) -> str:
        """Analyze tone shift between prompts."""
        tone_indicators = {
            'formal': [r'\b(?:please|kindly|would|could|may)\b', r'\b(?:furthermore|moreover|however)\b'],
            'casual': [r'\b(?:hey|hi|okay|cool|awesome)\b', r'[!]{2,}'],
            'imperative': [r'^(?:do|don\'t|make|create|write|generate)\b'],
            'questioning': [r'\?', r'\b(?:what|how|why|when|where|who)\b'],
            'encouraging': [r'\b(?:great|excellent|wonderful|amazing)\b'],
            'cautionary': [r'\b(?:careful|warning|note|important|critical)\b']
        }
        
        def get_tone_score(text):
            scores = {}
            text_lower = text.lower()
            for tone, patterns in tone_indicators.items():
                score = sum(len(re.findall(pattern, text_lower)) for pattern in patterns)
                scores[tone] = score
            return scores
        
        scores1 = get_tone_score(prompt1)
        scores2 = get_tone_score(prompt2)
        
        # Find dominant tones
        dominant1 = max(scores1.items(), key=lambda x: x[1])
        dominant2 = max(scores2.items(), key=lambda x: x[1])
        
        if dominant1[1] == 0 and dominant2[1] == 0:
            return "neutral → neutral"
        elif dominant1[0] == dominant2[0]:
            return f"consistent {dominant1[0]} tone"
        else:
            return f"{dominant1[0]} → {dominant2[0]}"
    
    def _generate_summary(self, token_diff: TokenDiff, 
                          semantic_similarity: SemanticSimilarity,
                          structural_comparison: StructuralComparison) -> str:
        """Generate comparison summary."""
        summary_parts = []
        
        # Semantic similarity summary
        if semantic_similarity.similarity_score > 0.8:
            summary_parts.append("Highly similar semantically")
        elif semantic_similarity.similarity_score > 0.6:
            summary_parts.append("Moderately similar semantically")
        elif semantic_similarity.similarity_score > 0.3:
            summary_parts.append("Somewhat similar semantically")
        else:
            summary_parts.append("Semantically different")
        
        # Token changes summary
        total_changes = len(token_diff.added) + len(token_diff.removed) + len(token_diff.modified)
        if total_changes == 0:
            summary_parts.append("identical token-wise")
        elif total_changes < 5:
            summary_parts.append("minor token changes")
        elif total_changes < 15:
            summary_parts.append("moderate token changes")
        else:
            summary_parts.append("significant token changes")
        
        # Length changes
        if abs(structural_comparison.length_diff) > 100:
            if structural_comparison.length_diff > 0:
                summary_parts.append("notably expanded")
            else:
                summary_parts.append("notably condensed")
        
        return ", ".join(summary_parts) + "."
    
    def _generate_recommendations(self, token_diff: TokenDiff,
                                  semantic_similarity: SemanticSimilarity,
                                  structural_comparison: StructuralComparison) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []
        
        # Semantic similarity recommendations
        if semantic_similarity.similarity_score < 0.5:
            recommendations.append("Consider aligning the core intent and purpose of both prompts")
        
        # Token-level recommendations
        if len(token_diff.removed) > len(token_diff.added):
            recommendations.append("The second prompt removes significant content - ensure important context isn't lost")
        elif len(token_diff.added) > len(token_diff.removed):
            recommendations.append("The second prompt adds significant content - verify all additions are necessary")
        
        # Structural recommendations
        if abs(structural_comparison.complexity_diff) > 0.2:
            if structural_comparison.complexity_diff > 0:
                recommendations.append("The second prompt is more complex - consider if simplification would help")
            else:
                recommendations.append("The second prompt is simpler - consider if more detail is needed")
        
        # Format recommendations
        if structural_comparison.format_similarity < 0.7:
            recommendations.append("Consider maintaining consistent formatting style between prompts")
        
        return recommendations[:5]  # Limit to top 5
    
    def _display_comparison_results(self, result: ComparisonResult, **kwargs) -> None:
        """Display comparison results in rich format."""
        console.print("\n🔍 Prompt Comparison Results", style="bold blue")
        console.print("="*60)
        
        # Overall similarity
        similarity_color = "green" if result.overall_similarity > 0.7 else "yellow" if result.overall_similarity > 0.4 else "red"
        console.print(f"\n📊 Overall Similarity: {result.overall_similarity:.3f}", style=f"bold {similarity_color}")
        console.print(f"📝 Summary: {result.summary}")
        console.print(f"🎭 Tone Shift: {result.tone_shift}")
        
        # Detailed metrics table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", style="white")
        table.add_column("Details", style="dim white")
        
        table.add_row(
            "Semantic Similarity", 
            f"{result.semantic_similarity.similarity_score:.3f}",
            f"Intent: {result.semantic_similarity.intent_similarity:.3f}, Topic: {result.semantic_similarity.topic_overlap:.3f}"
        )
        table.add_row(
            "Structural Similarity",
            f"{result.structural_comparison.structure_similarity:.3f}",
            f"Length diff: {result.structural_comparison.length_diff}, Sentences: {result.structural_comparison.sentence_count_diff}"
        )
        table.add_row(
            "Token Changes",
            f"{len(result.token_diff.added) + len(result.token_diff.removed)}",
            f"Added: {len(result.token_diff.added)}, Removed: {len(result.token_diff.removed)}"
        )
        
        console.print("\n", table)
        
        # Token diff details
        if kwargs.get('show_token_diff', True):
            self._display_token_diff(result.token_diff)
        
        # Recommendations
        if result.recommendations:
            console.print("\n💡 Recommendations:", style="bold green")
            for i, rec in enumerate(result.recommendations, 1):
                console.print(f"  {i}. {rec}")
        
        # Side-by-side display if requested
        if kwargs.get('side_by_side', False):
            self._display_side_by_side(result.prompt1, result.prompt2)
        
        console.print("\n" + "="*60)
    
    def _display_token_diff(self, token_diff: TokenDiff) -> None:
        """Display token-level differences."""
        if token_diff.added:
            added_text = Text(f"Added ({len(token_diff.added)}): ", style="green")
            added_text.append(" ".join(token_diff.added[:10]), style="green")
            if len(token_diff.added) > 10:
                added_text.append("...", style="dim green")
            console.print(added_text)
        
        if token_diff.removed:
            removed_text = Text(f"Removed ({len(token_diff.removed)}): ", style="red")
            removed_text.append(" ".join(token_diff.removed[:10]), style="red")
            if len(token_diff.removed) > 10:
                removed_text.append("...", style="dim red")
            console.print(removed_text)
        
        if token_diff.modified:
            console.print(f"Modified ({len(token_diff.modified)}):", style="yellow")
            for old, new in token_diff.modified[:5]:
                console.print(f"  {old} → {new}", style="yellow")
            if len(token_diff.modified) > 5:
                console.print("  ...", style="dim yellow")
    
    def _display_side_by_side(self, prompt1: str, prompt2: str) -> None:
        """Display prompts side by side."""
        console.print("\n📖 Side-by-Side Comparison:", style="bold")
        
        # Truncate if too long
        max_length = 200
        p1_display = prompt1[:max_length] + "..." if len(prompt1) > max_length else prompt1
        p2_display = prompt2[:max_length] + "..." if len(prompt2) > max_length else prompt2
        
        panel1 = Panel(p1_display, title="Prompt 1", border_style="blue")
        panel2 = Panel(p2_display, title="Prompt 2", border_style="green")
        
        console.print(Columns([panel1, panel2]))
    
    def export_comparison(self, result: ComparisonResult, output_path: str, format_type: str = "json") -> None:
        """
        Export comparison result to file.
        
        Args:
            result: Comparison result to export
            output_path: Output file path  
            format_type: Export format ('json', 'txt', 'html')
        """
        output_path = Path(output_path)
        
        try:
            if format_type == "json":
                with open(output_path, 'w') as f:
                    json.dump(asdict(result), f, indent=2, default=str)
            
            elif format_type == "txt":
                with open(output_path, 'w') as f:
                    f.write("PROMPT COMPARISON REPORT\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Overall Similarity: {result.overall_similarity:.3f}\n")
                    f.write(f"Summary: {result.summary}\n")
                    f.write(f"Tone Shift: {result.tone_shift}\n\n")
                    
                    f.write("RECOMMENDATIONS:\n")
                    for i, rec in enumerate(result.recommendations, 1):
                        f.write(f"{i}. {rec}\n")
            
            console.print(f"✅ Comparison exported to {output_path}", style="green")
            
        except Exception as e:
            console.print(f"❌ Failed to export comparison: {e}", style="red")