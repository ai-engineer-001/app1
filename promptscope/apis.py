"""
API integration layer for multiple LLM providers.

Provides unified interface for calling different LLM APIs with consistent
error handling, retry logic, and response formatting.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import openai
from anthropic import Anthropic
import google.generativeai as genai
import cohere
from mistralai.client import MistralClient
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .auth import AuthManager, AuthConfig

console = Console()

class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    LOCAL = "local"

@dataclass
class LLMRequest:
    """Standardized LLM request format."""
    prompt: str
    system_prompt: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False
    model: Optional[str] = None

@dataclass  
class LLMResponse:
    """Standardized LLM response format."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    latency: float
    metadata: Dict[str, Any]

class APIManager:
    """
    Unified API manager for multiple LLM providers.
    
    Handles authentication, request formatting, response parsing,
    and error handling across different provider APIs.
    """
    
    def __init__(self, model: str = "qwen/qwen-2.5-vl-32b-instruct", 
                 provider: str = "openrouter", auth: Optional[AuthManager] = None):
        """
        Initialize API manager.
        
        Args:
            model: Default model to use
            provider: Default provider to use  
            auth: Authentication manager instance
        """
        self.model = model
        self.provider = provider
        self.auth = auth or AuthManager()
        self._clients: Dict[str, Any] = {}
        self._auth_configs: Dict[str, AuthConfig] = {}
        
        # Initialize default provider
        self._init_provider(provider, model)
    
    def _init_provider(self, provider: str, model: Optional[str] = None) -> bool:
        """
        Initialize API client for specified provider.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            True if initialization successful, False otherwise
        """
        if provider in self._clients:
            return True
        
        auth_config = self.auth.get_auth_config(provider, model)
        if not auth_config:
            console.print(f"❌ Failed to get auth config for {provider}", style="red")
            return False
        
        try:
            if provider == "openrouter":
                # OpenRouter uses OpenAI-compatible API
                client = openai.OpenAI(
                    api_key=auth_config.api_key,
                    base_url=auth_config.base_url
                )
                
            elif provider == "openai":
                client = openai.OpenAI(api_key=auth_config.api_key)
                
            elif provider == "anthropic":
                client = Anthropic(api_key=auth_config.api_key)
                
            elif provider == "google":
                genai.configure(api_key=auth_config.api_key)
                client = genai.GenerativeModel(auth_config.model)
                
            elif provider == "mistral":
                client = MistralClient(api_key=auth_config.api_key)
                
            elif provider == "cohere":
                client = cohere.Client(auth_config.api_key)
                
            else:
                console.print(f"❌ Unsupported provider: {provider}", style="red")
                return False
            
            self._clients[provider] = client
            self._auth_configs[provider] = auth_config
            console.print(f"✅ Initialized {provider} client", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Failed to initialize {provider}: {e}", style="red")
            return False
    
    async def call_llm(self, request: LLMRequest, provider: Optional[str] = None) -> LLMResponse:
        """
        Make LLM API call with standardized request/response format.
        
        Args:
            request: Standardized LLM request
            provider: Override provider for this request
            
        Returns:
            Standardized LLM response
            
        Raises:
            Exception: If API call fails after retries
        """
        provider = provider or self.provider
        model = request.model or self.model
        
        if not self._init_provider(provider, model):
            raise Exception(f"Failed to initialize provider: {provider}")
        
        client = self._clients[provider]
        auth_config = self._auth_configs[provider]
        
        start_time = time.time()
        
        try:
            if provider in ["openrouter", "openai"]:
                response = await self._call_openai_compatible(client, request, auth_config)
            elif provider == "anthropic":
                response = await self._call_anthropic(client, request, auth_config)
            elif provider == "google":
                response = await self._call_google(client, request, auth_config)
            elif provider == "mistral":
                response = await self._call_mistral(client, request, auth_config)
            elif provider == "cohere":
                response = await self._call_cohere(client, request, auth_config)
            else:
                raise Exception(f"Provider not implemented: {provider}")
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=response["content"],
                model=response["model"],
                provider=provider,
                usage=response.get("usage", {}),
                latency=latency,
                metadata=response.get("metadata", {})
            )
            
        except Exception as e:
            console.print(f"❌ LLM API call failed: {e}", style="red")
            raise
    
    async def _call_openai_compatible(self, client: openai.OpenAI, 
                                      request: LLMRequest, 
                                      auth_config: AuthConfig) -> Dict[str, Any]:
        """Call OpenAI-compatible API (OpenAI, OpenRouter)."""
        messages = []
        
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        messages.append({"role": "user", "content": request.prompt})
        
        response = client.chat.completions.create(
            model=auth_config.model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=request.stream
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens, 
                "total_tokens": response.usage.total_tokens
            }
        }
    
    async def _call_anthropic(self, client: Anthropic, 
                              request: LLMRequest,
                              auth_config: AuthConfig) -> Dict[str, Any]:
        """Call Anthropic Claude API."""
        messages = [{"role": "user", "content": request.prompt}]
        
        response = client.messages.create(
            model=auth_config.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            messages=messages,
            system=request.system_prompt or ""
        )
        
        return {
            "content": response.content[0].text,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }
    
    async def _call_google(self, client, request: LLMRequest, 
                           auth_config: AuthConfig) -> Dict[str, Any]:
        """Call Google Gemini API."""
        prompt = request.prompt
        if request.system_prompt:
            prompt = f"{request.system_prompt}\n\n{prompt}"
        
        response = client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
        )
        
        return {
            "content": response.text,
            "model": auth_config.model,
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count,
                "completion_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": response.usage_metadata.total_token_count
            }
        }
    
    async def _call_mistral(self, client: MistralClient, 
                            request: LLMRequest,
                            auth_config: AuthConfig) -> Dict[str, Any]:
        """Call Mistral AI API."""
        messages = []
        
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        
        messages.append({"role": "user", "content": request.prompt})
        
        response = client.chat(
            model=auth_config.model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    
    async def _call_cohere(self, client: cohere.Client,
                           request: LLMRequest, 
                           auth_config: AuthConfig) -> Dict[str, Any]:
        """Call Cohere API."""
        response = client.generate(
            model=auth_config.model,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            p=request.top_p
        )
        
        return {
            "content": response.generations[0].text,
            "model": auth_config.model,
            "usage": {
                "prompt_tokens": 0,  # Cohere doesn't provide token counts
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    
    def batch_call(self, requests: List[LLMRequest], 
                   provider: Optional[str] = None,
                   max_concurrent: int = 5) -> List[LLMResponse]:
        """
        Make multiple LLM calls concurrently.
        
        Args:
            requests: List of LLM requests
            provider: Override provider for all requests
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of LLM responses in same order as requests
        """
        async def _batch_call():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def call_with_semaphore(req):
                async with semaphore:
                    return await self.call_llm(req, provider)
            
            tasks = [call_with_semaphore(req) for req in requests]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Processing {len(requests)} requests...", total=len(requests))
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                responses = loop.run_until_complete(_batch_call())
                progress.update(task, completed=len(requests))
                return responses
            finally:
                loop.close()
    
    def test_connection(self, provider: Optional[str] = None) -> bool:
        """
        Test API connection with a simple request.
        
        Args:
            provider: Provider to test (defaults to current provider)
            
        Returns:
            True if connection successful, False otherwise
        """
        provider = provider or self.provider
        
        test_request = LLMRequest(
            prompt="Hello! Please respond with 'Connection test successful.'",
            max_tokens=50,
            temperature=0.1
        )
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(self.call_llm(test_request, provider))
                success = "successful" in response.content.lower()
                
                if success:
                    console.print(f"✅ {provider} connection test passed", style="green")
                else:
                    console.print(f"⚠️ {provider} connection test unclear response", style="yellow")
                
                return success
            finally:
                loop.close()
                
        except Exception as e:
            console.print(f"❌ {provider} connection test failed: {e}", style="red")
            return False
    
    def get_available_models(self, provider: Optional[str] = None) -> List[str]:
        """
        Get list of available models for provider.
        
        Args:
            provider: Provider to query (defaults to current provider)
            
        Returns:
            List of available model names
        """
        provider = provider or self.provider
        
        # Return hardcoded lists for now - could be made dynamic
        model_lists = {
            "openrouter": [
                "qwen/qwen-2.5-vl-32b-instruct",
                "anthropic/claude-3-opus",
                "openai/gpt-4-turbo-preview",
                "meta-llama/llama-3.2-90b-vision-instruct",
                "google/gemini-pro-1.5",
                "mistralai/mistral-large-2407"
            ],
            "openai": [
                "gpt-4-turbo-preview",
                "gpt-4",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k"
            ],
            "anthropic": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229", 
                "claude-3-haiku-20240307"
            ],
            "google": [
                "gemini-pro",
                "gemini-pro-vision"
            ],
            "mistral": [
                "mistral-large-latest",
                "mistral-medium-latest",
                "mistral-small-latest"
            ],
            "cohere": [
                "command",
                "command-light"
            ]
        }
        
        return model_lists.get(provider, [])