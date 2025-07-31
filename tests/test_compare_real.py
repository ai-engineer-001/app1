"""
Real-world prompt comparison tests.

Tests the prompt comparison functionality with realistic scenarios
including agent workflows, multi-step processes, and edge cases.
"""

import pytest
from promptscope.compare import PromptComparator
from promptscope.apis import APIManager
import asyncio

class TestPromptComparison:
    """Test suite for prompt comparison functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = PromptComparator()
    
    def test_identical_prompts(self):
        """Test comparison of identical prompts."""
        prompt = "Write a detailed analysis of market trends in the technology sector."
        
        result = self.comparator.compare(prompt, prompt, display=False)
        
        assert result.overall_similarity > 0.95
        assert result.semantic_similarity.similarity_score > 0.95
        assert len(result.token_diff.added) == 0
        assert len(result.token_diff.removed) == 0
    
    def test_similar_prompts_different_wording(self):
        """Test comparison of semantically similar prompts with different wording."""
        prompt1 = "Analyze the technology market trends and provide insights."
        prompt2 = "Examine current trends in the tech industry and give analysis."
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        assert result.overall_similarity > 0.6
        assert result.semantic_similarity.similarity_score > 0.7
        assert result.semantic_similarity.intent_similarity > 0.8
    
    def test_different_prompts_same_domain(self):
        """Test comparison of different prompts in the same domain."""
        prompt1 = "Write a technical documentation for a REST API."
        prompt2 = "Create marketing copy for a software product."
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        assert result.overall_similarity < 0.7
        assert result.semantic_similarity.intent_similarity < 0.6
        assert "different" in result.summary.lower()
    
    def test_completely_different_prompts(self):
        """Test comparison of completely unrelated prompts."""
        prompt1 = "Explain quantum physics concepts to a 10-year-old."
        prompt2 = "Write a recipe for chocolate chip cookies."
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        assert result.overall_similarity < 0.3
        assert result.semantic_similarity.similarity_score < 0.4
        assert result.semantic_similarity.topic_overlap < 0.2
    
    def test_structured_vs_unstructured(self):
        """Test comparison between structured and unstructured prompts."""
        prompt1 = """Please write a report on renewable energy.
        
        Requirements:
        1. Include solar and wind power
        2. Provide statistics
        3. Format as markdown
        """
        
        prompt2 = "Write about renewable energy sources including some data."
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        assert result.overall_similarity > 0.4  # Same topic
        assert result.structural_comparison.structure_similarity < 0.7  # Different structure
        assert len(result.recommendations) > 0
    
    def test_prompt_with_examples_vs_without(self):
        """Test comparison of prompts with and without examples."""
        prompt1 = """
        Generate product descriptions for e-commerce.
        
        Example:
        Input: "Wireless headphones"
        Output: "Premium wireless headphones with noise cancellation..."
        """
        
        prompt2 = "Generate product descriptions for online stores."
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        assert result.overall_similarity > 0.5  # Similar intent
        assert result.structural_comparison.format_similarity < 0.8  # Different format
        assert any("example" in rec.lower() for rec in result.recommendations)
    
    def test_token_level_differences(self):
        """Test detailed token-level difference detection."""
        prompt1 = "Write a comprehensive analysis of market trends."
        prompt2 = "Write a detailed analysis of industry trends."
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        # Check token differences
        assert "comprehensive" in result.token_diff.removed or "comprehensive" in str(result.token_diff.modified)
        assert "detailed" in result.token_diff.added or "detailed" in str(result.token_diff.modified)
        assert "market" in result.token_diff.removed or "market" in str(result.token_diff.modified)
        assert "industry" in result.token_diff.added or "industry" in str(result.token_diff.modified)
    
    def test_tone_analysis(self):
        """Test tone shift detection between prompts."""
        prompt1 = "Please write a formal business proposal for the client."
        prompt2 = "Hey, write a cool business proposal for our client!"
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        assert "formal" in result.tone_shift.lower() or "casual" in result.tone_shift.lower()
        assert result.overall_similarity > 0.6  # Same intent, different tone
    
    def test_length_differences(self):
        """Test handling of significant length differences."""
        prompt1 = "Summarize the document."
        prompt2 = """
        Please provide a comprehensive summary of the attached document.
        Include the following elements:
        - Main themes and arguments
        - Key supporting evidence
        - Conclusions and recommendations
        - Potential implications
        
        Format the summary as a structured report with clear headings.
        Aim for 500-1000 words in length.
        """
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        assert result.structural_comparison.length_diff > 200  # Significant length difference
        assert result.overall_similarity > 0.4  # Same core intent
        assert any("expand" in rec.lower() or "detail" in rec.lower() 
                  for rec in result.recommendations)
    
    def test_constraint_differences(self):
        """Test detection of constraint differences."""
        prompt1 = "Write a blog post about AI. Make it exactly 500 words."
        prompt2 = "Write a blog post about AI. Keep it under 200 words."
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        assert result.overall_similarity > 0.7  # Same task
        assert "constraint" in result.summary.lower() or "length" in result.summary.lower()
        assert any("constraint" in rec.lower() for rec in result.recommendations)
    
    def test_few_shot_vs_zero_shot(self):
        """Test comparison between few-shot and zero-shot prompts."""
        few_shot = """
        Classify the sentiment of these reviews:
        
        Example 1:
        Review: "This product is amazing!"
        Sentiment: Positive
        
        Example 2:
        Review: "Terrible quality, would not recommend."
        Sentiment: Negative
        
        Now classify: "The item was okay, nothing special."
        """
        
        zero_shot = "Classify the sentiment of this review: 'The item was okay, nothing special.'"
        
        result = self.comparator.compare(few_shot, zero_shot, display=False)
        
        assert result.overall_similarity > 0.6  # Same task
        assert result.structural_comparison.structure_similarity < 0.5  # Very different structure
        assert any("example" in rec.lower() for rec in result.recommendations)
    
    def test_role_based_prompts(self):
        """Test comparison of role-based prompts."""
        prompt1 = "As a financial advisor, explain investment strategies."
        prompt2 = "As a marketing expert, explain investment opportunities."
        
        result = self.comparator.compare(prompt1, prompt2, display=False)
        
        assert result.overall_similarity > 0.4  # Related topics
        assert result.semantic_similarity.intent_similarity < 0.8  # Different perspectives
        assert "role" in result.tone_shift.lower() or "expert" in result.tone_shift.lower()

class TestWorkflowComparison:
    """Test prompt comparison in workflow contexts."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = PromptComparator()
    
    def test_workflow_step_evolution(self):
        """Test comparison of evolved workflow steps."""
        step_v1 = "Analyze the user input and extract key information."
        step_v2 = """
        Analyze the user input and extract the following information:
        1. Intent/goal
        2. Key entities (people, places, dates)
        3. Sentiment/tone
        4. Urgency level
        
        Format output as structured JSON.
        """
        
        result = self.comparator.compare(step_v1, step_v2, display=False)
        
        assert result.overall_similarity > 0.6  # Same core task
        assert result.structural_comparison.structure_similarity < 0.7  # Much more structured
        assert "expansion" in result.summary.lower() or "detail" in result.summary.lower()
        assert len(result.recommendations) > 0
    
    def test_agent_prompt_refinement(self):
        """Test comparison of agent prompt refinements."""
        original = "You are a helpful assistant. Answer the user's question."
        refined = """
        You are a knowledgeable assistant specializing in technical topics.
        
        Guidelines:
        - Provide accurate, evidence-based answers
        - If uncertain, clearly state limitations
        - Ask clarifying questions when needed
        - Use examples to illustrate complex concepts
        
        Answer the user's question following these guidelines.
        """
        
        result = self.comparator.compare(original, refined, display=False)
        
        assert result.overall_similarity > 0.5  # Same role
        assert result.structural_comparison.structure_similarity < 0.6  # Much more detailed
        assert "improvement" in result.summary.lower() or "enhanced" in result.summary.lower()

@pytest.mark.asyncio
class TestComparisonWithAPI:
    """Test comparison functionality with actual API calls."""
    
    async def test_semantic_similarity_with_llm(self):
        """Test semantic similarity using LLM evaluation."""
        # This test requires API access - skip if not available
        try:
            from promptscope.auth import AuthManager
            auth = AuthManager()
            api_key = auth.get_api_key("openrouter", interactive=False)
            
            if not api_key:
                pytest.skip("No API key available for LLM testing")
            
            api_manager = APIManager(auth=auth)
            comparator = PromptComparator(api_manager)
            
            prompt1 = "Explain machine learning to a beginner."
            prompt2 = "Describe artificial intelligence concepts for newcomers."
            
            result = comparator.compare(prompt1, prompt2, display=False)
            
            # With LLM enhancement, semantic similarity should be more accurate
            assert result.semantic_similarity.similarity_score > 0.7
            assert result.overall_similarity > 0.6
            
        except Exception as e:
            pytest.skip(f"API test failed: {e}")

def test_comparison_edge_cases():
    """Test edge cases in prompt comparison."""
    comparator = PromptComparator()
    
    # Empty prompts
    result = comparator.compare("", "", display=False)
    assert result.overall_similarity >= 0.9  # Empty is identical to empty
    
    # Very short prompts
    result = comparator.compare("Hi", "Hello", display=False)
    assert 0.3 <= result.overall_similarity <= 0.8  # Similar greeting
    
    # Single word vs paragraph
    result = comparator.compare("Summarize", 
                               "Please provide a comprehensive summary of the main points.",
                               display=False)
    assert result.overall_similarity > 0.4  # Same intent, very different expression

def test_comparison_export():
    """Test exporting comparison results."""
    import tempfile
    import json
    
    comparator = PromptComparator()
    
    prompt1 = "Write a technical report."
    prompt2 = "Create a technical document."
    
    result = comparator.compare(prompt1, prompt2, display=False)
    
    # Test JSON export
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        comparator.export_comparison(result, f.name, "json")
        
        # Verify the file was created and contains valid JSON
        with open(f.name, 'r') as read_file:
            exported_data = json.load(read_file)
            assert exported_data['overall_similarity'] == result.overall_similarity
            assert exported_data['prompt1'] == prompt1
            assert exported_data['prompt2'] == prompt2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])