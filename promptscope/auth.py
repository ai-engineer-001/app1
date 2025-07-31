"""
Authentication and API key management for PromptScope.

Handles secure API key storage, validation, and provider authentication
across multiple LLM providers.
"""

import os
import json
import getpass
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass
import keyring
from rich.console import Console
from rich.prompt import Prompt

console = Console()

@dataclass
class AuthConfig:
    """Configuration for authentication settings."""
    provider: str
    api_key: str
    base_url: Optional[str] = None
    model: Optional[str] = None
    max_retries: int = 3
    timeout: int = 30

class AuthManager:
    """
    Manages authentication for multiple LLM providers.
    
    Supports secure storage via keyring, environment variables,
    and config files with fallback mechanisms.
    """
    
    SUPPORTED_PROVIDERS = {
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": "qwen/qwen-2.5-vl-32b-instruct",
            "env_key": "OPENROUTER_API_KEY"
        },
        "openai": {
            "base_url": "https://api.openai.com/v1", 
            "default_model": "gpt-4-turbo-preview",
            "env_key": "OPENAI_API_KEY"
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com",
            "default_model": "claude-3-opus-20240229",
            "env_key": "ANTHROPIC_API_KEY"
        },
        "google": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "default_model": "gemini-pro",
            "env_key": "GOOGLE_API_KEY"
        },
        "mistral": {
            "base_url": "https://api.mistral.ai/v1",
            "default_model": "mistral-large-latest",
            "env_key": "MISTRAL_API_KEY"
        },
        "cohere": {
            "base_url": "https://api.cohere.ai/v1",
            "default_model": "command",
            "env_key": "COHERE_API_KEY"
        }
    }
    
    def __init__(self):
        """Initialize authentication manager."""
        self.config_dir = Path.home() / ".promptscope"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(exist_ok=True)
        self._config_cache: Optional[Dict[str, Any]] = None
    
    def get_api_key(self, provider: str, interactive: bool = False) -> Optional[str]:
        """
        Get API key for specified provider using multiple fallback methods.
        
        Args:
            provider: Name of the LLM provider
            interactive: Whether to prompt user for key if not found
            
        Returns:
            API key string or None if not found/provided
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            console.print(f"❌ Unsupported provider: {provider}", style="red")
            return None
        
        provider_info = self.SUPPORTED_PROVIDERS[provider]
        
        # Method 1: Environment variable
        env_key = provider_info["env_key"]
        api_key = os.getenv(env_key)
        if api_key:
            console.print(f"✅ Found {provider} API key in environment", style="green")
            return api_key
        
        # Method 2: Keyring (secure storage)
        try:
            api_key = keyring.get_password("promptscope", f"{provider}_api_key")
            if api_key:
                console.print(f"✅ Found {provider} API key in keyring", style="green")
                return api_key
        except Exception as e:
            console.print(f"⚠️ Keyring access failed: {e}", style="yellow")
        
        # Method 3: Config file
        config = self.load_config()
        if config.get("api_keys", {}).get(provider):
            api_key = config["api_keys"][provider]
            console.print(f"✅ Found {provider} API key in config file", style="green")
            return api_key
        
        # Method 4: Interactive prompt
        if interactive:
            return self._prompt_for_api_key(provider)
        
        return None
    
    def _prompt_for_api_key(self, provider: str) -> Optional[str]:
        """
        Interactively prompt user for API key with secure input.
        
        Args:
            provider: Name of the LLM provider
            
        Returns:
            API key string or None if cancelled
        """
        console.print(f"\n🔑 API key required for {provider.upper()}", style="bold blue")
        
        # Show where to get the API key
        key_urls = {
            "openrouter": "https://openrouter.ai/keys",
            "openai": "https://platform.openai.com/api-keys",
            "anthropic": "https://console.anthropic.com/keys",
            "google": "https://aistudio.google.com/app/apikey", 
            "mistral": "https://console.mistral.ai/api-keys/",
            "cohere": "https://dashboard.cohere.com/api-keys"
        }
        
        if provider in key_urls:
            console.print(f"📝 Get your API key from: {key_urls[provider]}")
        
        try:
            api_key = getpass.getpass(f"Enter {provider} API key (hidden input): ").strip()
            
            if not api_key:
                console.print("❌ No API key provided", style="red")
                return None
            
            # Ask if user wants to save the key
            save_choice = Prompt.ask(
                "💾 Save this API key securely for future use?",
                choices=["keyring", "config", "no"],
                default="keyring"
            )
            
            if save_choice == "keyring":
                try:
                    keyring.set_password("promptscope", f"{provider}_api_key", api_key)
                    console.print("✅ API key saved to system keyring", style="green")
                except Exception as e:
                    console.print(f"⚠️ Failed to save to keyring: {e}", style="yellow")
                    self._save_to_config(provider, api_key)
            
            elif save_choice == "config":
                self._save_to_config(provider, api_key)
            
            return api_key
            
        except KeyboardInterrupt:
            console.print("\n❌ Operation cancelled", style="red")
            return None
    
    def _save_to_config(self, provider: str, api_key: str) -> None:
        """Save API key to config file."""
        try:
            config = self.load_config()
            if "api_keys" not in config:
                config["api_keys"] = {}
            
            config["api_keys"][provider] = api_key
            self.save_config(config)
            console.print("✅ API key saved to config file", style="green")
        except Exception as e:
            console.print(f"❌ Failed to save to config: {e}", style="red")
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self._config_cache is not None:
            return self._config_cache
        
        if not self.config_file.exists():
            default_config = {
                "default_provider": "openrouter",
                "default_model": "qwen/qwen-2.5-vl-32b-instruct", 
                "offline_fallback": True,
                "api_keys": {}
            }
            self.save_config(default_config)
            return default_config
        
        try:
            with open(self.config_file, "r") as f:
                config = json.load(f)
                self._config_cache = config
                return config
        except Exception as e:
            console.print(f"❌ Failed to load config: {e}", style="red")
            return {}
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            self._config_cache = config
        except Exception as e:
            console.print(f"❌ Failed to save config: {e}", style="red")
    
    def get_auth_config(self, provider: str, model: Optional[str] = None) -> Optional[AuthConfig]:
        """
        Get complete authentication configuration for provider.
        
        Args:
            provider: Name of the LLM provider
            model: Optional model override
            
        Returns:
            AuthConfig object or None if authentication fails
        """
        api_key = self.get_api_key(provider, interactive=False)
        if not api_key:
            return None
        
        provider_info = self.SUPPORTED_PROVIDERS[provider]
        
        return AuthConfig(
            provider=provider,
            api_key=api_key,
            base_url=provider_info["base_url"],
            model=model or provider_info["default_model"]
        )
    
    def validate_api_key(self, provider: str, api_key: str) -> bool:
        """
        Validate API key by making a test request.
        
        Args:
            provider: Name of the LLM provider
            api_key: API key to validate
            
        Returns:
            True if key is valid, False otherwise
        """
        # This will be implemented when we create the API client
        # For now, just check if key looks valid
        return bool(api_key and len(api_key.strip()) > 10)
    
    def list_configured_providers(self) -> Dict[str, bool]:
        """
        List all providers and their authentication status.
        
        Returns:
            Dict mapping provider names to authentication status
        """
        status = {}
        for provider in self.SUPPORTED_PROVIDERS:
            api_key = self.get_api_key(provider, interactive=False)
            status[provider] = bool(api_key)
        return status
    
    def setup_interactive(self) -> None:
        """Interactive setup wizard for configuring authentication."""
        console.print("\n🚀 PromptScope Authentication Setup", style="bold blue")
        console.print("Configure API keys for your preferred LLM providers\n")
        
        # Show current status
        status = self.list_configured_providers()
        for provider, configured in status.items():
            emoji = "✅" if configured else "❌"
            console.print(f"{emoji} {provider.upper()}: {'Configured' if configured else 'Not configured'}")
        
        console.print("\n" + "="*50)
        
        # Ask which providers to configure
        providers_to_setup = []
        for provider in self.SUPPORTED_PROVIDERS:
            if not status[provider]:
                setup = Prompt.ask(f"Configure {provider.upper()}?", choices=["y", "n"], default="n")
                if setup == "y":
                    providers_to_setup.append(provider)
        
        # Configure each provider
        for provider in providers_to_setup:
            console.print(f"\n🔧 Setting up {provider.upper()}")
            api_key = self._prompt_for_api_key(provider)
            if api_key:
                console.print(f"✅ {provider.upper()} configured successfully")
            else:
                console.print(f"❌ {provider.upper()} setup cancelled")
        
        console.print("\n🎉 Setup complete! You can now use PromptScope.", style="green")