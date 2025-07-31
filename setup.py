"""Setup script for promptscope package."""

from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="promptscope",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    author="PromptScope Team",
    author_email="team@promptscope.ai",
    description="Production-grade Python module for prompt analysis, comparison, response evaluation, and improvement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/promptscope/promptscope",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "typer>=0.9.0",
        "click>=8.1.0", 
        "rich>=13.5.0",
        "pydantic>=2.4.0",
        "openai>=1.12.0",
        "anthropic>=0.18.0",
        "google-generativeai>=0.4.0",
        "cohere>=4.47.0",
        "mistralai>=0.1.0",
        "requests>=2.31.0",
        "nltk>=3.8.1",
        "spacy>=3.7.0",
        "transformers>=4.36.0",
        "sentence-transformers>=2.2.2",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "jsonlines>=4.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "ruff>=0.0.292",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "promptscope=promptscope.cli:app",
        ],
    },
)