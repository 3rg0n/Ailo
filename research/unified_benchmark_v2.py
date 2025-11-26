"""
Unified Benchmark v2 - With Improved Evaluation

Uses the new evaluation module with:
- Fuzzy matching for text answers
- Numeric matching with tolerance for math
- F1 scoring for partial credit
- Keywords matching for subjective tasks

Usage:
    python unified_benchmark_v2.py --model gpt-4o-mini
    python unified_benchmark_v2.py --model gemini-2.0-flash
    python unified_benchmark_v2.py --model claude-haiku
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from multi_provider_client import MultiProviderClient, MODELS, ModelConfig
from test_prompts_v4 import MULTI_STYLE_PROMPTS, PromptStyle, MultiStylePrompt
from evaluation import evaluate, EvalResult, evaluate_with_llm_judge


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_benchmark(
    model_key: str,
    styles_to_test: list[str] = None,
    prompts_to_test: list[str] = None,
    use_llm_judge: bool = False,
):
    """Run comprehensive benchmark on a model."""

    if model_key not in MODELS:
        print(f"Unknown model: {model_key}")
        print(f"Available: {list(MODELS.keys())}")
        return None

    model_config = MODELS[model_key]
    client = MultiProviderClient()

    if styles_to_test is None:
        styles_to_test = [s.value for s in PromptStyle]

    # Filter prompts if specified
    prompts = MULTI_STYLE_PROMPTS
    if prompts_to_test:
        prompts = [p for p in MULTI_STYLE_PROMPTS if p.id in prompts_to_test]

    print(f"\n{'='*70}")
    print(f"Unified Benchmark v2: {model_config.name} ({model_config.provider})")
    print(f"Styles: {len(styles_to_test)} | Tests: {len(prompts)}")
    print(f"LLM Judge (Opus 4.5): {'ENABLED' if use_llm_judge else 'disabled'}")
    print(f"{'='*70}\n")

    results = []

    for prompt_def in prompts:
        test_case = prompt_def.to_test_case()

        test_results = {
            "test_id": prompt_def.id,
            "test_name": prompt_def.name,
            "category": prompt_def.category.value,
            "eval_type": prompt_def.eval_type.value,
            "styles": {}
        }

        print(f"  {prompt_def.name[:35]:35}", end=" ")

        for style in styles_to_test:
            # Get prompt for this style
            style_prompt = getattr(prompt_def, style, None)
            if not style_prompt:
                continue

            try:
                # Call the model
                response = client.invoke(style_prompt, model_config)

                # Evaluate the response (deterministic)
                eval_result = evaluate(response["response"], test_case)

                # Add rate limiting for Gemini
                if model_config.provider == "gemini":
                    time.sleep(1)

                test_results["styles"][style] = {
                    "response": response["response"][:500],
                    "input_tokens": response["input_tokens"],
                    "output_tokens": response["output_tokens"],
                    "total_tokens": response["input_tokens"] + response["output_tokens"],
                    "latency_ms": response.get("latency_ms", 0),
                    # Deterministic evaluation fields
                    "correct": eval_result.correct,
                    "score": eval_result.score,
                    "eval_method": eval_result.method,
                    "eval_reason": eval_result.reason,
                    "eval_details": eval_result.details,
                }

                # Optional LLM Judge evaluation
                if use_llm_judge:
                    judge_eval = evaluate_with_llm_judge(
                        response["response"],
                        test_case,
                        eval_result
                    )
                    test_results["styles"][style]["llm_judge"] = judge_eval["llm_judge"]
                    test_results["styles"][style]["combined_score"] = judge_eval["combined"]["score"]
                    test_results["styles"][style]["combined_correct"] = judge_eval["combined"]["correct"]
                    # Use combined score for display when judge is enabled
                    display_score = judge_eval["combined"]["score"]
                    display_correct = judge_eval["combined"]["correct"]
                else:
                    display_score = eval_result.score
                    display_correct = eval_result.correct

                status = "+" if display_correct else "-"
                print(f"{style[:4]}:{status}{display_score:.1f}", end=" | ")

            except Exception as e:
                print(f"{style[:4]}:ERR", end=" | ")
                test_results["styles"][style] = {"error": str(e)}

        print()
        results.append(test_results)

    # Calculate summary statistics
    summary = calculate_summary(results, styles_to_test, use_llm_judge)

    # Print summary
    print_summary(summary, model_config, use_llm_judge)

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"unified_v2_{model_key}_{timestamp}.json"

    output_data = {
        "metadata": {
            "model": model_key,
            "model_name": model_config.name,
            "provider": model_config.provider,
            "tier": model_config.tier,
            "timestamp": timestamp,
            "styles_tested": styles_to_test,
            "version": "v2_improved_eval",
            "llm_judge_enabled": use_llm_judge,
        },
        "results": results,
        "summary": summary,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output_data


def calculate_summary(results: list, styles: list, use_llm_judge: bool = False) -> dict:
    """Calculate summary statistics by style."""
    summary = {}

    for style in styles:
        correct_count = 0
        total_score = 0.0
        tokens_total = 0
        count = 0

        # LLM judge specific metrics
        judge_scores = {"correctness": [], "completeness": [], "clarity": [], "relevance": []}
        combined_scores = []

        for test in results:
            if style in test["styles"] and "error" not in test["styles"][style]:
                style_data = test["styles"][style]

                # Use combined score if LLM judge was used
                if use_llm_judge and "combined_correct" in style_data:
                    correct_count += 1 if style_data.get("combined_correct", False) else 0
                    total_score += style_data.get("combined_score", 0.0)
                    combined_scores.append(style_data.get("combined_score", 0.0))

                    # Collect judge dimension scores
                    if "llm_judge" in style_data:
                        for dim in judge_scores.keys():
                            if dim in style_data["llm_judge"]:
                                judge_scores[dim].append(style_data["llm_judge"][dim])
                else:
                    correct_count += 1 if style_data.get("correct", False) else 0
                    total_score += style_data.get("score", 0.0)

                tokens_total += style_data.get("total_tokens", 0)
                count += 1

        if count > 0:
            summary[style] = {
                "accuracy": round(correct_count / count * 100, 1),
                "avg_score": round(total_score / count, 3),
                "avg_tokens": round(tokens_total / count),
                "tests_completed": count,
                "correct": correct_count,
            }

            # Add LLM judge dimension averages if available
            if use_llm_judge and combined_scores:
                summary[style]["avg_combined_score"] = round(sum(combined_scores) / len(combined_scores), 3)
                for dim, scores in judge_scores.items():
                    if scores:
                        summary[style][f"avg_{dim}"] = round(sum(scores) / len(scores), 3)

    return summary


def print_summary(summary: dict, model_config: ModelConfig, use_llm_judge: bool = False):
    """Print formatted summary."""
    print(f"\n{'='*80}")
    print(f"Summary: {model_config.name}")
    print(f"{'='*80}")

    # Get zero-shot baseline
    baseline_acc = summary.get("zero_shot", {}).get("accuracy", 0)
    baseline_score = summary.get("zero_shot", {}).get("avg_score", 0)
    baseline_tokens = summary.get("zero_shot", {}).get("avg_tokens", 0)

    print(f"{'Style':<20} {'Accuracy':>10} {'vs Zero':>10} {'Avg Score':>10} {'Tokens':>10} {'vs Zero':>10}")
    print("-" * 72)

    for style, data in sorted(summary.items(), key=lambda x: -x[1]["accuracy"]):
        acc = data["accuracy"]
        score = data["avg_score"]
        tokens = data["avg_tokens"]

        if style == "zero_shot":
            acc_diff = "baseline"
            token_diff = "baseline"
        else:
            acc_diff_val = acc - baseline_acc
            acc_diff = f"{'+' if acc_diff_val >= 0 else ''}{acc_diff_val:.1f}%"
            if baseline_tokens > 0:
                token_diff_val = (tokens - baseline_tokens) / baseline_tokens * 100
                token_diff = f"{'+' if token_diff_val >= 0 else ''}{token_diff_val:.1f}%"
            else:
                token_diff = "N/A"

        print(f"{style:<20} {acc:>9.1f}% {acc_diff:>10} {score:>10.2f} {tokens:>10.0f} {token_diff:>10}")

    # Print LLM Judge dimension scores if available
    has_judge_data = any("avg_correctness" in data for data in summary.values())
    if has_judge_data:
        print(f"\n{'='*80}")
        print("LLM Judge Dimension Scores (Opus 4.5)")
        print("-" * 80)
        print(f"{'Style':<20} {'Correctness':>12} {'Completeness':>12} {'Clarity':>12} {'Relevance':>12}")
        print("-" * 80)

        for style, data in sorted(summary.items(), key=lambda x: -x[1]["accuracy"]):
            corr = data.get("avg_correctness", 0)
            comp = data.get("avg_completeness", 0)
            clar = data.get("avg_clarity", 0)
            rel = data.get("avg_relevance", 0)
            print(f"{style:<20} {corr:>11.2f} {comp:>12.2f} {clar:>12.2f} {rel:>12.2f}")

    # ROI analysis
    print(f"\n{'='*80}")
    print("ROI Analysis (Accuracy Gain per 100 extra tokens)")
    print("-" * 40)

    for style, data in sorted(summary.items(), key=lambda x: -x[1]["accuracy"]):
        if style == "zero_shot":
            continue

        acc_gain = data["accuracy"] - baseline_acc
        token_increase = data["avg_tokens"] - baseline_tokens

        if token_increase > 0:
            roi = acc_gain / (token_increase / 100)
            print(f"{style:<20} {roi:>+.2f}% acc per 100 tokens")
        elif token_increase < 0:
            print(f"{style:<20} {acc_gain:>+.1f}% acc, {token_increase:.0f} fewer tokens [EFFICIENT]")
        else:
            print(f"{style:<20} {acc_gain:>+.1f}% acc, same tokens")


def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Provider Benchmark v2")
    parser.add_argument("--model", required=True, help="Model to test (e.g., gpt-4o-mini, gemini-2.0-flash)")
    parser.add_argument("--styles", nargs="+", default=None, help="Styles to test")
    parser.add_argument("--prompts", nargs="+", default=None, help="Specific prompt IDs to test")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Enable LLM-as-judge evaluation using Opus 4.5 (adds cost but more accurate)")
    args = parser.parse_args()

    run_benchmark(args.model, args.styles, args.prompts, args.llm_judge)


if __name__ == "__main__":
    main()
