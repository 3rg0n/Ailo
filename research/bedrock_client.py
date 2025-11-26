"""
AWS Bedrock client for testing prompt formats.
Uses environment variables for credentials:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
"""

import boto3
import json
import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a Bedrock model."""
    model_id: str
    name: str
    provider: str
    region: str = "us-east-1"
    max_tokens: int = 2048
    # Pricing per 1K tokens (USD) - approximate for cost analysis
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    # Model tier for analysis (budget, mid, premium)
    tier: str = "mid"


# Selected models for benchmark - diverse providers for solid statistical sampling
# Using global inference profiles where available for better reliability/throughput
# Pricing as of Nov 2025 (USD per 1K tokens) - verify current rates
MODELS = {
    # Amazon - flagship model (regional - no global profile)
    "nova-pro": ModelConfig(
        model_id="amazon.nova-pro-v1:0",
        name="Amazon Nova Pro",
        provider="Amazon",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.0032,
        tier="mid",
    ),

    # Anthropic - using global inference profile for better throughput
    "claude-sonnet": ModelConfig(
        model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        name="Claude Sonnet 4.5 (Global)",
        provider="Anthropic",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        tier="premium",
    ),

    # Anthropic - ultra premium flagship model
    "claude-opus": ModelConfig(
        model_id="global.anthropic.claude-opus-4-5-20251101-v1:0",
        name="Claude Opus 4.5 (Global)",
        provider="Anthropic",
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        tier="ultra",
    ),

    # Anthropic - fast/cheap option with global profile
    "claude-haiku": ModelConfig(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        name="Claude Haiku 4.5 (Global)",
        provider="Anthropic",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.004,
        tier="budget",
    ),

    # Meta - open source leader
    "llama3-70b": ModelConfig(
        model_id="meta.llama3-3-70b-instruct-v1:0",
        name="Llama 3.3 70B Instruct",
        provider="Meta",
        input_cost_per_1k=0.00099,
        output_cost_per_1k=0.00099,
        tier="mid",
    ),

    # Mistral - European alternative
    "mistral-large": ModelConfig(
        model_id="mistral.mistral-large-2402-v1:0",
        name="Mistral Large",
        provider="Mistral AI",
        input_cost_per_1k=0.004,
        output_cost_per_1k=0.012,
        tier="premium",
    ),

    # Cohere - enterprise focused
    "command-r-plus": ModelConfig(
        model_id="cohere.command-r-plus-v1:0",
        name="Command R+",
        provider="Cohere",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        tier="premium",
    ),

    # AI21 - different architecture (Mamba-based)
    "jamba-large": ModelConfig(
        model_id="ai21.jamba-1-5-large-v1:0",
        name="Jamba 1.5 Large",
        provider="AI21 Labs",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.008,
        tier="mid",
    ),
}

# Quick/cheap models for testing setup
QUICK_MODELS = {
    "nova-micro": ModelConfig(
        model_id="amazon.nova-micro-v1:0",
        name="Amazon Nova Micro",
        provider="Amazon",
        input_cost_per_1k=0.000035,
        output_cost_per_1k=0.00014,
        tier="budget",
    ),
    "nova-lite": ModelConfig(
        model_id="amazon.nova-lite-v1:0",
        name="Amazon Nova Lite",
        provider="Amazon",
        input_cost_per_1k=0.00006,
        output_cost_per_1k=0.00024,
        tier="budget",
    ),
    # Additional budget models
    "mistral-7b": ModelConfig(
        model_id="mistral.mistral-7b-instruct-v0:2",
        name="Mistral 7B Instruct",
        provider="Mistral AI",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0002,
        tier="budget",
    ),
    "mistral-small": ModelConfig(
        model_id="mistral.mistral-small-2402-v1:0",
        name="Mistral Small",
        provider="Mistral AI",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.003,
        tier="budget",
    ),
    "mixtral-8x7b": ModelConfig(
        model_id="mistral.mixtral-8x7b-instruct-v0:1",
        name="Mixtral 8x7B Instruct",
        provider="Mistral AI",
        input_cost_per_1k=0.00045,
        output_cost_per_1k=0.0007,
        tier="mid",
    ),
    "llama3-8b": ModelConfig(
        model_id="meta.llama3-8b-instruct-v1:0",
        name="Llama 3 8B Instruct",
        provider="Meta",
        input_cost_per_1k=0.0003,
        output_cost_per_1k=0.0006,
        tier="budget",
    ),
    # Smaller Llama models
    "llama3-2-1b": ModelConfig(
        model_id="meta.llama3-2-1b-instruct-v1:0",
        name="Llama 3.2 1B Instruct",
        provider="Meta",
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0001,
        tier="budget",
    ),
    "llama3-2-3b": ModelConfig(
        model_id="meta.llama3-2-3b-instruct-v1:0",
        name="Llama 3.2 3B Instruct",
        provider="Meta",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.00015,
        tier="budget",
    ),
    # Amazon Titan
    "titan-text": ModelConfig(
        model_id="amazon.titan-tg1-large",
        name="Amazon Titan Text Large",
        provider="Amazon",
        input_cost_per_1k=0.0003,
        output_cost_per_1k=0.0004,
        tier="budget",
    ),
}

# Combine all models
ALL_MODELS = {**MODELS, **QUICK_MODELS}


class BedrockClient:
    """Client for invoking AWS Bedrock models using the Converse API."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

    def invoke(
        self,
        prompt: str,
        model_config: ModelConfig,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> dict:
        """
        Invoke a Bedrock model with a prompt using the unified Converse API.

        The Converse API works across all major model providers, simplifying
        our code significantly.

        Returns dict with:
        - response: The model's text response
        - input_tokens: Number of input tokens
        - output_tokens: Number of output tokens
        - model: Model ID used
        """
        max_tokens = max_tokens or model_config.max_tokens

        # Use Converse API - works for most models
        try:
            return self._invoke_converse(prompt, model_config, temperature, max_tokens)
        except Exception as e:
            # Fallback to model-specific invoke if converse fails
            if "ai21.jamba" in model_config.model_id:
                return self._invoke_jamba(prompt, model_config, temperature, max_tokens)
            raise e

    def _invoke_converse(
        self,
        prompt: str,
        model_config: ModelConfig,
        temperature: float,
        max_tokens: int
    ) -> dict:
        """
        Invoke using Bedrock Converse API.
        Works with: Amazon Nova, Anthropic Claude, Meta Llama, Mistral, Cohere
        """
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]

        inference_config = {
            "temperature": temperature,
            "maxTokens": max_tokens,
        }

        response = self.client.converse(
            modelId=model_config.model_id,
            messages=messages,
            inferenceConfig=inference_config
        )

        # Extract response text
        output_message = response.get("output", {}).get("message", {})
        content = output_message.get("content", [])
        response_text = content[0].get("text", "") if content else ""

        # Extract token usage
        usage = response.get("usage", {})

        return {
            "response": response_text,
            "input_tokens": usage.get("inputTokens", 0),
            "output_tokens": usage.get("outputTokens", 0),
            "model": model_config.model_id,
        }

    def _invoke_jamba(
        self,
        prompt: str,
        model_config: ModelConfig,
        temperature: float,
        max_tokens: int
    ) -> dict:
        """Fallback for AI21 Jamba models if Converse doesn't work."""
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self.client.invoke_model(
            modelId=model_config.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read())

        choices = response_body.get("choices", [])
        response_text = ""
        if choices:
            message = choices[0].get("message", {})
            response_text = message.get("content", "")

        usage = response_body.get("usage", {})

        return {
            "response": response_text,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "model": model_config.model_id,
        }


def test_connection(model_key: str = "nova-micro"):
    """Test that we can connect to Bedrock."""
    client = BedrockClient()

    # Use quick model for testing
    if model_key in ALL_MODELS:
        model = ALL_MODELS[model_key]
    else:
        model = QUICK_MODELS["nova-micro"]

    print(f"Testing connection with {model.name}...")

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
        return True
    except Exception as e:
        print(f"[FAIL] Connection failed: {e}")
        return False


def list_models():
    """List all available models."""
    print("\n=== Benchmark Models ===")
    for key, model in MODELS.items():
        print(f"  {key:20} {model.name:30} ({model.provider})")

    print("\n=== Quick Test Models ===")
    for key, model in QUICK_MODELS.items():
        print(f"  {key:20} {model.name:30} ({model.provider})")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_models()
        else:
            test_connection(sys.argv[1])
    else:
        test_connection()
