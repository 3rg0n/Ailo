"""
Unified benchmark runner for all providers (Bedrock, OpenAI, Gemini).

Usage:
    python unified_benchmark.py --model gpt-4o-mini
    python unified_benchmark.py --model gemini-2.0-flash
    python unified_benchmark.py --model claude-haiku
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from multi_provider_client import MultiProviderClient, MODELS, ModelConfig
from test_prompts_v3 import MULTI_STYLE_PROMPTS, PromptStyle

# Convert to simpler format
TEST_PROMPTS = []
for prompt in MULTI_STYLE_PROMPTS:
    test = {
        "id": prompt.id,
        "name": prompt.name,
        "category": prompt.category.value,
        "prompts": {},
        "criteria": {},
    }
    # Add prompts for each style
    for style in PromptStyle:
        style_prompt = getattr(prompt, style.value, None)
        if style_prompt:
            test["prompts"][style.value] = style_prompt
    # Add criteria checks
    for criterion in prompt.criteria:
        test["criteria"][criterion.name] = criterion.check
    TEST_PROMPTS.append(test)

STYLES = [s.value for s in PromptStyle]


def evaluate_criteria(response: str, criteria: dict) -> dict:
    """Evaluate response against criteria."""
    results = {}

    for criterion_name, check_func in criteria.items():
        try:
            # If it's a callable (from test_prompts_v3), use it directly
            if callable(check_func):
                results[criterion_name] = check_func(response)
            else:
                results[criterion_name] = True  # Default pass
        except Exception as e:
            results[criterion_name] = False  # Failed check

    return results


def run_benchmark(model_key: str, styles_to_test: list = None):
    """Run comprehensive benchmark on a model."""

    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {list(MODELS.keys())}")
        return None

    model_config = MODELS[model_key]
    client = MultiProviderClient()

    if styles_to_test is None:
        styles_to_test = STYLES

    print(f"\n{'='*70}")
    print(f"Unified Benchmark: {model_config.name} ({model_config.provider})")
    print(f"Styles: {len(styles_to_test)} | Tests: {len(TEST_PROMPTS)}")
    print(f"{'='*70}\n")

    results = []

    for test in TEST_PROMPTS:
        test_results = {
            "test_id": test["id"],
            "test_name": test["name"],
            "category": test["category"],
            "styles": {}
        }

        print(f"  {test['name'][:30]:30}", end=" ")

        for style in styles_to_test:
            if style not in test["prompts"]:
                continue

            prompt = test["prompts"][style]

            try:
                response = client.invoke(prompt, model_config)
                criteria_results = evaluate_criteria(response["response"], test["criteria"])

                # Add small delay for rate-limited APIs (Gemini)
                if model_config.provider == "gemini":
                    time.sleep(1)

                passed = sum(1 for v in criteria_results.values() if v)
                total = len(criteria_results)

                test_results["styles"][style] = {
                    "response": response["response"],
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                    "total_tokens": response["input_tokens"] + response["output_tokens"],
                    "latency_ms": response.get("latency_ms", 0),
                    "criteria_results": criteria_results,
                    "criteria_passed": passed,
                    "criteria_total": total,
                }

                print(f"{style[:4]}:{passed}/{total}", end=" | ")

            except Exception as e:
                print(f"{style[:4]}:ERR", end=" | ")
                test_results["styles"][style] = {"error": str(e)}

        print()
        results.append(test_results)

    # Calculate summary statistics
    summary = calculate_summary(results, styles_to_test)

    # Print summary
    print_summary(summary, model_config)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"unified_{model_key}_{timestamp}.json"

    output_data = {
        "metadata": {
            "model": model_key,
            "model_name": model_config.name,
            "provider": model_config.provider,
            "tier": model_config.tier,
            "timestamp": timestamp,
            "styles_tested": styles_to_test,
        },
        "results": results,
        "summary": summary,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output_data


def calculate_summary(results: list, styles: list) -> dict:
    """Calculate summary statistics by style."""
    summary = {}

    for style in styles:
        passed_total = 0
        criteria_total = 0
        tokens_total = 0
        count = 0

        for test in results:
            if style in test["styles"] and "error" not in test["styles"][style]:
                style_data = test["styles"][style]
                passed_total += style_data["criteria_passed"]
                criteria_total += style_data["criteria_total"]
                tokens_total += style_data["total_tokens"]
                count += 1

        if count > 0:
            summary[style] = {
                "pass_rate": round(passed_total / criteria_total * 100, 1) if criteria_total > 0 else 0,
                "avg_tokens": round(tokens_total / count),
                "tests_completed": count,
            }

    return summary


def print_summary(summary: dict, model_config: ModelConfig):
    """Print formatted summary."""
    print(f"\n{'='*70}")
    print(f"Summary: {model_config.name}")
    print(f"{'='*70}")

    # Get zero-shot baseline
    baseline_rate = summary.get("zero_shot", {}).get("pass_rate", 0)
    baseline_tokens = summary.get("zero_shot", {}).get("avg_tokens", 0)

    print(f"{'Style':<20} {'Pass Rate':>12} {'vs Zero':>10} {'Tokens':>10} {'Token Diff':>12}")
    print("-" * 66)

    for style, data in sorted(summary.items(), key=lambda x: -x[1]["pass_rate"]):
        rate = data["pass_rate"]
        tokens = data["avg_tokens"]

        if style == "zero_shot":
            vs_zero = "baseline"
            token_diff = "baseline"
        else:
            diff = rate - baseline_rate
            vs_zero = f"{'+' if diff >= 0 else ''}{diff:.1f}%"
            if baseline_tokens > 0:
                tdiff = (tokens - baseline_tokens) / baseline_tokens * 100
                token_diff = f"{'+' if tdiff >= 0 else ''}{tdiff:.1f}%"
            else:
                token_diff = "N/A"

        print(f"{style:<20} {rate:>11.1f}% {vs_zero:>10} {tokens:>10} {token_diff:>12}")


def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Provider Benchmark")
    parser.add_argument("--model", required=True, help="Model to test (e.g., gpt-4o-mini, gemini-2.0-flash)")
    parser.add_argument("--styles", nargs="+", default=None, help="Styles to test")
    args = parser.parse_args()

    run_benchmark(args.model, args.styles)


if __name__ == "__main__":
    main()
