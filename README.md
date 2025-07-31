# PromptScope 🎯

**Production-grade Python module for prompt analysis, comparison, response evaluation, and improvement**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/promptscope.svg)](https://badge.fury.io/py/promptscope)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🧠 Philosophy

PromptScope prioritizes **output accuracy and interpretability** over UI flash. Every comparison, score, or suggestion is backed by clear logic and deterministic scoring. Modules function in offline mode for open-source models and support online mode via secure API key prompts.

## 🚀 Features

- **Prompt Analysis**: Extract patterns, repetition, length, tone, and structure from JSONL logs
- **Prompt Comparison**: Token-level + semantic diffs with deterministic scoring
- **Response Evaluation**: Score responses for relevance, hallucination, toxicity, and fluency
- **Workflow Scoring**: Analyze prompt chains for coherence, redundancy, and cost estimation
- **Prompt Improvement**: AI-powered suggestions to fix unclear or non-actionable prompts
- **Few-shot Optimizer**: Analyze and optimize few-shot example structure

## 📦 Installation

```bash
pip install promptscope
```

For development:
```bash
git clone https://github.com/promptscope/promptscope.git
cd promptscope
pip install -e ".[dev]"
```

## 🔑 Supported LLM Providers

- **OpenAI** (GPT-4, GPT-3.5)
- **Anthropic** (Claude-3, Claude-2)
- **Google Gemini** (Gemini Pro, Gemini Vision)
- **Mistral AI** (Mistral Large, Mistral Medium)
- **Cohere** (Command, Command-Light)
- **OpenRouter** (Access to 100+ models including Qwen, OpenChat)
- **Local Models** (via Transformers, Ollama, vLLM)

## 🛠️ CLI Usage

### Basic Commands

```bash
# Compare two prompts with semantic analysis
promptscope compare new.txt old.txt --semantic --no-color

# Analyze prompt logs for patterns
promptscope analyze logs.jsonl --json

# Evaluate responses for quality metrics
promptscope eval responses.jsonl --models openai,mistral --toxicity --relevance

# Improve unclear prompts
promptscope improve unclear_prompt.txt --model gemini-pro

# Score workflow chains
promptscope score-chain workflow.json
```

### Advanced Examples

```bash
# Side-by-side comparison with custom output
promptscope compare prompt1.txt prompt2.txt --side-by-side --output comparison_report.json

# Comprehensive analysis with visualization
promptscope analyze agent_logs.jsonl --visualize --save-plots analysis_plots/

# Multi-model evaluation with detailed scoring
promptscope eval responses.jsonl --models openai,anthropic,cohere --all-metrics --debug

# Few-shot optimization with embedding analysis
promptscope optimize-fewshot examples.jsonl --visualize --remove-outliers
```

## 🐍 Python API

```python
import promptscope

# Initialize with your preferred model
ps = promptscope.PromptScope(model="qwen2.5-vl-32b-instruct", provider="openrouter")

# Analyze prompts
analysis = ps.analyze("path/to/prompts.jsonl")
print(analysis.summary)

# Compare prompts
comparison = ps.compare("prompt1.txt", "prompt2.txt", semantic=True)
print(f"Similarity: {comparison.similarity_score}")

# Evaluate responses
evaluation = ps.evaluate("responses.jsonl", metrics=["relevance", "toxicity"])
print(evaluation.scores)

# Improve prompts
improved = ps.improve("unclear_prompt.txt")
print(improved.suggestions)
```

## 🔧 Configuration

PromptScope supports multiple configuration methods:

### 1. Environment Variables
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
export OPENROUTER_API_KEY="your-key-here"
```

### 2. Config File (`~/.promptscope/config.json`)
```json
{
  "default_model": "qwen2.5-vl-32b-instruct",
  "default_provider": "openrouter",
  "api_keys": {
    "openrouter": "your-key-here"
  },
  "offline_fallback": true
}
```

### 3. Interactive Setup
```bash
promptscope setup
```

## 📊 Output Formats

All commands support multiple output formats:

- **Console**: Rich, colored terminal output (default)
- **JSON**: Structured data for programmatic use
- **JSONL**: Line-delimited JSON logs with timestamps
- **CSV**: Tabular data for analysis
- **HTML**: Rich reports with embedded visualizations

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Local development testing:
```bash
python test_local_run.py
```

## 📈 Performance & Accuracy

PromptScope is designed for production use with:

- **Deterministic Scoring**: Reproducible results across runs
- **Batch Processing**: Efficient handling of large datasets
- **Caching**: Smart caching of embeddings and API responses
- **Fallback Support**: Graceful degradation to offline models
- **Memory Efficient**: Streaming processing for large files

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Typer](https://typer.tiangolo.com/) for the CLI interface
- Powered by [Hugging Face Transformers](https://huggingface.co/transformers/) for local model support
- Visualization powered by [Rich](https://rich.readthedocs.io/) and [Plotly](https://plotly.com/)

---

**Made with ❤️ by the PromptScope team**