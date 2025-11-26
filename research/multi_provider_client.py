"""
Multi-provider LLM client supporting AWS Bedrock, OpenAI, and Google Gemini.

Environment variables:
- AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (for Bedrock)
- OPENAI_KEY (for OpenAI)
- GEMINI_KEY (for Google Gemini)

Can also load from .env file in the research directory.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional
import time
from pathlib import Path

# Load .env file if it exists
def load_dotenv():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)

load_dotenv()

# Import providers
try:
    import boto3
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""
    model_id: str
    name: str
    provider: str  # "bedrock", "openai", "gemini"
    max_tokens: int = 2048
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    tier: str = "mid"  # budget, mid, premium, ultra


# All available models across providers
MODELS = {
    # === AWS Bedrock Models ===
    "nova-micro": ModelConfig(
        model_id="amazon.nova-micro-v1:0",
        name="Amazon Nova Micro",
        provider="bedrock",
        input_cost_per_1k=0.000035,
        output_cost_per_1k=0.00014,
        tier="budget",
    ),
    "nova-lite": ModelConfig(
        model_id="amazon.nova-lite-v1:0",
        name="Amazon Nova Lite",
        provider="bedrock",
        input_cost_per_1k=0.00006,
        output_cost_per_1k=0.00024,
        tier="budget",
    ),
    "nova-pro": ModelConfig(
        model_id="amazon.nova-pro-v1:0",
        name="Amazon Nova Pro",
        provider="bedrock",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.0032,
        tier="mid",
    ),
    "claude-haiku": ModelConfig(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        name="Claude Haiku 4.5",
        provider="bedrock",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.004,
        tier="budget",
    ),
    "claude-sonnet": ModelConfig(
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        name="Claude Sonnet 4.5",
        provider="bedrock",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        tier="premium",
    ),
    "claude-opus": ModelConfig(
        model_id="global.anthropic.claude-opus-4-5-20251101-v1:0",
        name="Claude Opus 4.5",
        provider="bedrock",
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        tier="ultra",
    ),
    "mistral-7b": ModelConfig(
        model_id="mistral.mistral-7b-instruct-v0:2",
        name="Mistral 7B Instruct",
        provider="bedrock",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0002,
        tier="budget",
    ),
    "mistral-large": ModelConfig(
        model_id="mistral.mistral-large-2402-v1:0",
        name="Mistral Large",
        provider="bedrock",
        input_cost_per_1k=0.004,
        output_cost_per_1k=0.012,
        tier="premium",
    ),
    "llama3-70b": ModelConfig(
        model_id="meta.llama3-3-70b-instruct-v1:0",
        name="Llama 3.3 70B",
        provider="bedrock",
        input_cost_per_1k=0.00099,
        output_cost_per_1k=0.00099,
        tier="mid",
    ),

    # === OpenAI Models ===
    "gpt-4o": ModelConfig(
        model_id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        input_cost_per_1k=0.0025,
        output_cost_per_1k=0.01,
        tier="premium",
    ),
    "gpt-4o-mini": ModelConfig(
        model_id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        tier="budget",
    ),
    "gpt-4-turbo": ModelConfig(
        model_id="gpt-4-turbo",
        name="GPT-4 Turbo",
        provider="openai",
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        tier="premium",
    ),
    "gpt-3.5-turbo": ModelConfig(
        model_id="gpt-3.5-turbo",
        name="GPT-3.5 Turbo",
        provider="openai",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        tier="budget",
    ),
    "o1-mini": ModelConfig(
        model_id="o1-mini",
        name="OpenAI o1-mini",
        provider="openai",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.012,
        tier="premium",
    ),

    # === Google Gemini Models ===
    "gemini-2.0-flash": ModelConfig(
        model_id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="gemini",
        input_cost_per_1k=0.0,  # Free tier available
        output_cost_per_1k=0.0,
        tier="budget",
    ),
    "gemini-1.5-flash": ModelConfig(
        model_id="gemini-1.5-flash",
        name="Gemini 1.5 Flash",
        provider="gemini",
        input_cost_per_1k=0.000075,
        output_cost_per_1k=0.0003,
        tier="budget",
    ),
    "gemini-1.5-pro": ModelConfig(
        model_id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        provider="gemini",
        input_cost_per_1k=0.00125,
        output_cost_per_1k=0.005,
        tier="premium",
    ),
    "gemini-2.0-flash-thinking": ModelConfig(
        model_id="gemini-2.0-flash-thinking-exp",
        name="Gemini 2.0 Flash Thinking",
        provider="gemini",
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        tier="premium",
    ),
}


class MultiProviderClient:
    """Unified client for multiple LLM providers."""

    def __init__(self):
        self._bedrock_client = None
        self._openai_client = None
        self._gemini_configured = False

    def _get_bedrock_client(self):
        """Lazy initialization of Bedrock client."""
        if self._bedrock_client is None:
            if not BEDROCK_AVAILABLE:
                raise ImportError("boto3 not installed. Run: pip install boto3")
            self._bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name="us-east-1",
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
        return self._bedrock_client

    def _get_openai_client(self):
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed. Run: pip install openai")
            api_key = os.environ.get("OPENAI_KEY")
            if not api_key:
                raise ValueError("OPENAI_KEY environment variable not set")
            self._openai_client = openai.OpenAI(api_key=api_key)
        return self._openai_client

    def _configure_gemini(self):
        """Configure Gemini API."""
        if not self._gemini_configured:
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
            api_key = os.environ.get("GEMINI_KEY")
            if not api_key:
                raise ValueError("GEMINI_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self._gemini_configured = True

    def invoke(
        self,
        prompt: str,
        model_config: ModelConfig,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> dict:
        """
        Invoke a model with a prompt.

        Returns dict with:
        - response: The model's text response
        - input_tokens: Number of input tokens
        - output_tokens: Number of output tokens
        - model: Model ID used
        - latency_ms: Response time in milliseconds
        """
        max_tokens = max_tokens or model_config.max_tokens
        start_time = time.time()

        if model_config.provider == "bedrock":
            result = self._invoke_bedrock(prompt, model_config, temperature, max_tokens)
        elif model_config.provider == "openai":
            result = self._invoke_openai(prompt, model_config, temperature, max_tokens)
        elif model_config.provider == "gemini":
            result = self._invoke_gemini(prompt, model_config, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {model_config.provider}")

        result["latency_ms"] = (time.time() - start_time) * 1000
        return result

    def invoke_multi_turn(
        self,
        messages: list[dict],
        model_config: ModelConfig,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> dict:
        """
        Invoke a model with multi-turn conversation history.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            model_config: Model configuration
            temperature: Sampling temperature
            max_tokens: Max output tokens

        Returns dict with:
        - response: The model's text response
        - input_tokens: Number of input tokens
        - output_tokens: Number of output tokens
        - model: Model ID used
        - latency_ms: Response time in milliseconds
        """
        max_tokens = max_tokens or model_config.max_tokens
        start_time = time.time()

        if model_config.provider == "bedrock":
            result = self._invoke_bedrock_multi(messages, model_config, temperature, max_tokens)
        elif model_config.provider == "openai":
            result = self._invoke_openai_multi(messages, model_config, temperature, max_tokens)
        elif model_config.provider == "gemini":
            result = self._invoke_gemini_multi(messages, model_config, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {model_config.provider}")

        result["latency_ms"] = (time.time() - start_time) * 1000
        return result

    def _invoke_bedrock(self, prompt: str, model_config: ModelConfig, temperature: float, max_tokens: int) -> dict:
        """Invoke AWS Bedrock model."""
        client = self._get_bedrock_client()

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        inference_config = {"temperature": temperature, "maxTokens": max_tokens}

        response = client.converse(
            modelId=model_config.model_id,
            messages=messages,
            inferenceConfig=inference_config
        )

        output_message = response.get("output", {}).get("message", {})
        content = output_message.get("content", [])
        response_text = content[0].get("text", "") if content else ""
        usage = response.get("usage", {})

        return {
            "response": response_text,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "model": model_config.model_id,
        }

    def _invoke_bedrock_multi(self, messages: list[dict], model_config: ModelConfig, temperature: float, max_tokens: int) -> dict:
        """Invoke AWS Bedrock model with multi-turn conversation."""
        client = self._get_bedrock_client()

        # Convert to Bedrock format
        bedrock_messages = []
        for msg in messages:
            bedrock_messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })

        inference_config = {"temperature": temperature, "maxTokens": max_tokens}

        response = client.converse(
            modelId=model_config.model_id,
            messages=bedrock_messages,
            inferenceConfig=inference_config
        )

        output_message = response.get("output", {}).get("message", {})
        content = output_message.get("content", [])
        response_text = content[0].get("text", "") if content else ""
        usage = response.get("usage", {})

        return {
            "response": response_text,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "model": model_config.model_id,
        }

    def _invoke_openai(self, prompt: str, model_config: ModelConfig, temperature: float, max_tokens: int) -> dict:
        """Invoke OpenAI model."""
        client = self._get_openai_client()

        # o1 models don't support temperature
        if model_config.model_id.startswith("o1"):
            response = client.chat.completions.create(
                model=model_config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
        else:
            response = client.chat.completions.create(
                model=model_config.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return {
            "response": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "model": model_config.model_id,
        }

    def _invoke_openai_multi(self, messages: list[dict], model_config: ModelConfig, temperature: float, max_tokens: int) -> dict:
        """Invoke OpenAI model with multi-turn conversation."""
        client = self._get_openai_client()

        # OpenAI already uses the same format
        if model_config.model_id.startswith("o1"):
            response = client.chat.completions.create(
                model=model_config.model_id,
                messages=messages,
                max_completion_tokens=max_tokens,
            )
        else:
            response = client.chat.completions.create(
                model=model_config.model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return {
            "response": response.choices[0].message.content or "",
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "model": model_config.model_id,
        }

    def _invoke_gemini(self, prompt: str, model_config: ModelConfig, temperature: float, max_tokens: int) -> dict:
        """Invoke Google Gemini model with retry logic for rate limits."""
        self._configure_gemini()

        model = genai.GenerativeModel(model_config.model_id)

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Retry logic for rate limiting
        max_retries = 3
        retry_delay = 5  # seconds

        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt, generation_config=generation_config)

                # Extract token counts from usage metadata
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, 'usage_metadata'):
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

                return {
                    "response": response.text if response.text else "",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model": model_config.model_id,
                }
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        continue
                raise e

        raise Exception("Max retries exceeded for Gemini API")

    def _invoke_gemini_multi(self, messages: list[dict], model_config: ModelConfig, temperature: float, max_tokens: int) -> dict:
        """Invoke Google Gemini model with multi-turn conversation."""
        self._configure_gemini()

        model = genai.GenerativeModel(model_config.model_id)

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Convert messages to Gemini format (user/model roles)
        gemini_history = []
        for msg in messages[:-1]:  # All but last message
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        # Start chat with history
        chat = model.start_chat(history=gemini_history)

        # Send the last message
        last_msg = messages[-1]["content"]

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = chat.send_message(last_msg, generation_config=generation_config)

                input_tokens = 0
                output_tokens = 0
                if hasattr(response, 'usage_metadata'):
                    input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                    output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

                return {
                    "response": response.text if response.text else "",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "model": model_config.model_id,
                }
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "quota" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                raise e

        raise Exception("Max retries exceeded for Gemini API")


def list_models(provider: Optional[str] = None):
    """List available models, optionally filtered by provider."""
    print("\n=== Available Models ===\n")

    providers = [provider] if provider else ["bedrock", "openai", "gemini"]

    for p in providers:
        print(f"--- {p.upper()} ---")
        for key, model in MODELS.items():
            if model.provider == p:
                available = ""
                if p == "bedrock" and not BEDROCK_AVAILABLE:
                    available = " (boto3 not installed)"
                elif p == "openai" and not OPENAI_AVAILABLE:
                    available = " (openai not installed)"
                elif p == "gemini" and not GEMINI_AVAILABLE:
                    available = " (google-generativeai not installed)"
                print(f"  {key:20} {model.name:25} [{model.tier}]{available}")
        print()


def test_connection(model_key: str):
    """Test connection to a specific model."""
    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available models: {list(MODELS.keys())}")
        return False

    model = MODELS[model_key]
    client = MultiProviderClient()

    print(f"Testing connection to {model.name} ({model.provider})...")

    try:
        result = client.invoke(
            prompt="Say hello in exactly 5 words.",
            model_config=model,
            max_tokens=50
        )
        print(f"[OK] Connection successful!")
        print(f"  Model: {result['model']}")
        print(f"  Response: {result['response']}")
        print(f"  Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
        print(f"  Latency: {result['latency_ms']:.0f}ms")
        return True
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            provider = sys.argv[2] if len(sys.argv) > 2 else None
            list_models(provider)
        else:
            test_connection(sys.argv[1])
    else:
        list_models()
