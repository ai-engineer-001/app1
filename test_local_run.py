"""
Local test runner for PromptScope development and validation.

Runs comprehensive tests without requiring external dependencies,
perfect for local development and CI/CD pipelines.
"""

import json
import tempfile
from pathlib import Path
import sys
import traceback

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from promptscope import PromptScope
from promptscope.analysis import PromptAnalyzer
from promptscope.compare import PromptComparator
from promptscope.evaluate import ResponseEvaluator
from promptscope.improve import PromptImprover
from promptscope.scoring import WorkflowScorer
from promptscope.auth import AuthManager

def create_test_data():
    """Create test data files for validation."""
    # Create test prompts JSONL
    test_prompts = [
        {"prompt": "Write a comprehensive analysis of renewable energy trends in 2024."},
        {"prompt": "Explain quantum computing concepts to a business audience."},
        {"prompt": "Create a marketing strategy for a new AI-powered app."},
        {"prompt": "Analyze the impact of remote work on team productivity."},
        {"prompt": "Write a technical documentation for a REST API endpoint."}
    ]
    
    # Create test responses JSONL
    test_responses = [
        {
            "prompt": "What is machine learning?",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "prompt": "Explain neural networks.",
            "response": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes that process information."
        },
        {
            "prompt": "What is deep learning?",
            "response": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data."
        }
    ]
    
    # Create test workflow
    test_workflow = {
        "name": "Content Generation Workflow",
        "steps": {
            "research": {
                "prompt": "Research the given topic and gather key facts and statistics.",
                "dependencies": [],
                "metadata": {"estimated_tokens": 200}
            },
            "outline": {
                "prompt": "Create a detailed outline based on the research findings. Structure it with main points and supporting details.",
                "dependencies": ["research"],
                "metadata": {"estimated_tokens": 150}
            },
            "draft": {
                "prompt": "Write a comprehensive first draft based on the outline. Include all key points with detailed explanations.",
                "dependencies": ["outline"],
                "metadata": {"estimated_tokens": 500}
            },
            "review": {
                "prompt": "Review and improve the draft for clarity, accuracy, and flow. Provide specific suggestions.",
                "dependencies": ["draft"],
                "metadata": {"estimated_tokens": 300}
            }
        }
    }
    
    return test_prompts, test_responses, test_workflow

def test_prompt_analysis():
    """Test prompt analysis functionality."""
    print("🔍 Testing Prompt Analysis...")
    
    try:
        # Create test data
        test_prompts, _, _ = create_test_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for prompt in test_prompts:
                f.write(json.dumps(prompt) + '\n')
            temp_file = f.name
        
        # Test analysis
        analyzer = PromptAnalyzer()
        report = analyzer.analyze_file(temp_file)
        
        # Validate results
        assert report.total_prompts == len(test_prompts)
        assert report.avg_length_words > 0
        assert report.avg_complexity >= 0
        assert len(report.recommendations) > 0
        
        print("✅ Prompt Analysis: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Prompt Analysis: FAILED - {e}")
        traceback.print_exc()
        return False

def test_prompt_comparison():
    """Test prompt comparison functionality."""
    print("🔍 Testing Prompt Comparison...")
    
    try:
        prompt1 = "Write a detailed analysis of market trends in technology."
        prompt2 = "Create a comprehensive analysis of tech industry trends."
        
        comparator = PromptComparator()
        result = comparator.compare(prompt1, prompt2, display=False)
        
        # Validate results
        assert 0 <= result.overall_similarity <= 1
        assert result.semantic_similarity.similarity_score >= 0
        assert result.structural_comparison is not None
        assert len(result.summary) > 0
        
        # Test identical prompts
        identical_result = comparator.compare(prompt1, prompt1, display=False)
        assert identical_result.overall_similarity > 0.95
        
        print("✅ Prompt Comparison: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Prompt Comparison: FAILED - {e}")
        traceback.print_exc()
        return False

def test_response_evaluation():
    """Test response evaluation functionality."""
    print("🔍 Testing Response Evaluation...")
    
    try:
        # Create test data
        _, test_responses, _ = create_test_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for response in test_responses:
                f.write(json.dumps(response) + '\n')
            temp_file = f.name
        
        # Test evaluation (without API)
        evaluator = ResponseEvaluator()
        report = evaluator.evaluate_file(temp_file, metrics=['toxicity', 'fluency', 'relevance'])
        
        # Validate results
        assert report.total_responses == len(test_responses)
        assert report.average_scores.toxicity >= 0
        assert report.average_scores.fluency >= 0
        assert len(report.recommendations) >= 0
        
        print("✅ Response Evaluation: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Response Evaluation: FAILED - {e}")
        traceback.print_exc()
        return False

def test_prompt_improvement():
    """Test prompt improvement functionality."""
    print("🔍 Testing Prompt Improvement...")
    
    try:
        test_prompt = "Write something about AI."
        
        # Test without API (heuristic improvements)
        improver = PromptImprover()
        report = improver.improve(test_prompt, display=False)
        
        # Validate results
        assert report.original_prompt == test_prompt
        assert len(report.suggestions) > 0
        assert len(report.recommendations) > 0
        assert report.overall_assessment is not None
        
        # Should identify vague language
        vague_issues = [s for s in report.suggestions if "vague" in s.issue.lower() or "something" in s.issue.lower()]
        assert len(vague_issues) > 0
        
        print("✅ Prompt Improvement: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Prompt Improvement: FAILED - {e}")
        traceback.print_exc()
        return False

def test_workflow_scoring():
    """Test workflow scoring functionality."""
    print("🔍 Testing Workflow Scoring...")
    
    try:
        # Create test data
        _, _, test_workflow = create_test_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_workflow, f, indent=2)
            temp_file = f.name
        
        # Test workflow scoring
        scorer = WorkflowScorer()
        report = scorer.score_file(temp_file, display=False)
        
        # Validate results
        assert report.total_steps == len(test_workflow['steps'])
        assert report.metrics.overall_score >= 0
        assert report.metrics.coherence_score >= 0
        assert len(report.dependency_graph) > 0
        assert len(report.optimization_recommendations) >= 0
        
        print("✅ Workflow Scoring: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Workflow Scoring: FAILED - {e}")
        traceback.print_exc()
        return False

def test_auth_manager():
    """Test authentication manager functionality."""
    print("🔍 Testing Authentication Manager...")
    
    try:
        auth_manager = AuthManager()
        
        # Test configuration loading
        config = auth_manager.load_config()
        assert isinstance(config, dict)
        
        # Test provider status
        status = auth_manager.list_configured_providers()
        assert isinstance(status, dict)
        assert len(status) > 0
        
        # Test supported providers
        assert "openrouter" in auth_manager.SUPPORTED_PROVIDERS
        assert "openai" in auth_manager.SUPPORTED_PROVIDERS
        
        print("✅ Authentication Manager: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Authentication Manager: FAILED - {e}")  
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between components."""
    print("🔍 Testing Integration...")
    
    try:
        # Test PromptScope main class
        ps = PromptScope()
        
        # Test that all components are initialized
        assert ps.analyzer is not None
        assert ps.comparator is not None
        assert ps.evaluator is not None
        assert ps.improver is not None
        assert ps.scorer is not None
        
        # Test a simple workflow
        test_prompt = "Analyze market trends."
        
        # This should work without API access
        comparison = ps.compare(test_prompt, test_prompt + " Include statistics.")
        assert comparison.overall_similarity > 0.7
        
        print("✅ Integration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Integration: FAILED - {e}")
        traceback.print_exc()
        return False

def test_cli_imports():
    """Test that CLI module imports work correctly."""
    print("🔍 Testing CLI Imports...")
    
    try:
        from promptscope.cli import app
        from promptscope import __version__
        
        assert app is not None
        assert __version__ is not None
        
        print("✅ CLI Imports: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ CLI Imports: FAILED - {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all local tests and return results."""
    print("🚀 Starting PromptScope Local Test Suite")
    print("="*60)
    
    tests = [
        ("Authentication Manager", test_auth_manager),
        ("Prompt Analysis", test_prompt_analysis),
        ("Prompt Comparison", test_prompt_comparison),
        ("Response Evaluation", test_response_evaluation),
        ("Prompt Improvement", test_prompt_improvement),
        ("Workflow Scoring", test_workflow_scoring),
        ("Integration", test_integration),
        ("CLI Imports", test_cli_imports),
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
            failed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("📊 Test Results Summary")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n📈 Total: {len(tests)} tests, {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! PromptScope is ready to use.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)