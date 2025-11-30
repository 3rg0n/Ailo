"""
Unified Benchmark v2.1 - With Improved Evaluation + Verbalized Sampling

Uses the new evaluation module with:
- Fuzzy matching for text answers
- Numeric matching with tolerance for math
- F1 scoring for partial credit
- Keywords matching for subjective tasks
- TRUE multi-turn conversations for ToT and Self-Consistency
- Verbalized Sampling (VS) from arXiv:2510.01171 with diversity metrics

Usage:
    python unified_benchmark_v2.py --model gpt-4o-mini
    python unified_benchmark_v2.py --model gemini-2.0-flash
    python unified_benchmark_v2.py --model claude-haiku

    # Test only specific styles including new VS
    python unified_benchmark_v2.py --model claude-haiku --styles zero_shot cot verbalized_sampling
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from multi_provider_client import MultiProviderClient, MODELS, ModelConfig
from test_prompts_v4 import MULTI_STYLE_PROMPTS, PromptStyle, MultiStylePrompt
from evaluation import (
    evaluate, EvalResult, evaluate_with_llm_judge,
    evaluate_vs_response, evaluate_vs_diversity, parse_verbalized_sampling
)


# =============================================================================
# Multi-Turn Implementations for ToT and Self-Consistency
# =============================================================================

def run_multi_turn_tot(client: MultiProviderClient, model_config: ModelConfig, task: str) -> dict:
    """
    Run TRUE multi-turn Tree of Thoughts:
    1. Explore Path A
    2. Explore Path B
    3. Explore Path C
    4. Synthesize and select best answer

    Returns dict with response, token counts, and turn details.
    """
    total_input = 0
    total_output = 0
    turns = []

    # Turn 1: Explore Path A
    messages = [{"role": "user", "content": f"""I need you to solve this problem. First, let's explore Path A.

TASK: {task}

PATH A: Solve this using the most straightforward, direct approach. Show your work."""}]

    result_a = client.invoke_multi_turn(messages, model_config)
    total_input += result_a["input_tokens"]
    total_output += result_a["output_tokens"]
    turns.append({"path": "A", "response": result_a["response"]})
    messages.append({"role": "assistant", "content": result_a["response"]})

    # Turn 2: Explore Path B
    messages.append({"role": "user", "content": """Good. Now let's explore Path B.

PATH B: Solve this using an alternative method or perspective. Show your work differently than Path A."""})

    result_b = client.invoke_multi_turn(messages, model_config)
    total_input += result_b["input_tokens"]
    total_output += result_b["output_tokens"]
    turns.append({"path": "B", "response": result_b["response"]})
    messages.append({"role": "assistant", "content": result_b["response"]})

    # Turn 3: Explore Path C
    messages.append({"role": "user", "content": """Good. Now let's explore Path C.

PATH C: Verify using a third approach - this could be checking the logic, using estimation, or working backwards. Show your verification."""})

    result_c = client.invoke_multi_turn(messages, model_config)
    total_input += result_c["input_tokens"]
    total_output += result_c["output_tokens"]
    turns.append({"path": "C", "response": result_c["response"]})
    messages.append({"role": "assistant", "content": result_c["response"]})

    # Turn 4: Synthesize
    messages.append({"role": "user", "content": """Now evaluate all three paths:

1. Which path(s) arrived at the correct answer?
2. Which approach was most reliable?
3. What is the FINAL ANSWER?

Synthesize your findings and provide the definitive answer."""})

    result_synth = client.invoke_multi_turn(messages, model_config)
    total_input += result_synth["input_tokens"]
    total_output += result_synth["output_tokens"]
    turns.append({"path": "synthesis", "response": result_synth["response"]})

    return {
        "response": result_synth["response"],
        "input_tokens": total_input,
        "output_tokens": total_output,
        "turns": turns,
        "num_calls": 4,
    }


def run_multi_turn_self_consistency(client: MultiProviderClient, model_config: ModelConfig, task: str) -> dict:
    """
    Run TRUE multi-turn Self-Consistency:
    1. Solve with Method 1
    2. Solve with Method 2
    3. Solve with Method 3
    4. Compare and reconcile answers

    Returns dict with response, token counts, and turn details.
    """
    total_input = 0
    total_output = 0
    turns = []

    # Turn 1: Method 1
    messages = [{"role": "user", "content": f"""I need you to solve this problem using Method 1: Standard/Direct Calculation.

TASK: {task}

Solve using the most standard approach. Show your work and state your answer clearly."""}]

    result_1 = client.invoke_multi_turn(messages, model_config)
    total_input += result_1["input_tokens"]
    total_output += result_1["output_tokens"]
    turns.append({"method": "1", "response": result_1["response"]})
    messages.append({"role": "assistant", "content": result_1["response"]})

    # Turn 2: Method 2
    messages.append({"role": "user", "content": """Now solve the SAME problem using Method 2: Alternative Approach.

Use a different technique or formula. Don't just repeat what you did before - approach it fresh. Show your work and state your answer."""})

    result_2 = client.invoke_multi_turn(messages, model_config)
    total_input += result_2["input_tokens"]
    total_output += result_2["output_tokens"]
    turns.append({"method": "2", "response": result_2["response"]})
    messages.append({"role": "assistant", "content": result_2["response"]})

    # Turn 3: Method 3
    messages.append({"role": "user", "content": """Now solve using Method 3: Verification/Cross-check.

This could be: working backwards from an expected answer, using estimation, or applying a completely different framework. Show your work."""})

    result_3 = client.invoke_multi_turn(messages, model_config)
    total_input += result_3["input_tokens"]
    total_output += result_3["output_tokens"]
    turns.append({"method": "3", "response": result_3["response"]})
    messages.append({"role": "assistant", "content": result_3["response"]})

    # Turn 4: Reconcile
    messages.append({"role": "user", "content": """Now compare all three methods:

1. Method 1 answer: [state it]
2. Method 2 answer: [state it]
3. Method 3 answer: [state it]

Do all methods agree? If not, which is most reliable and why?

State your FINAL ANSWER with confidence."""})

    result_reconcile = client.invoke_multi_turn(messages, model_config)
    total_input += result_reconcile["input_tokens"]
    total_output += result_reconcile["output_tokens"]
    turns.append({"method": "reconcile", "response": result_reconcile["response"]})

    return {
        "response": result_reconcile["response"],
        "input_tokens": total_input,
        "output_tokens": total_output,
        "turns": turns,
        "num_calls": 4,
    }


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
                # Use multi-turn for ToT and Self-Consistency, single-call for others
                if style == "tot":
                    # TRUE multi-turn Tree of Thoughts
                    response = run_multi_turn_tot(client, model_config, prompt_def.zero_shot)
                    response["latency_ms"] = 0  # Multi-turn doesn't track latency easily
                elif style == "self_consistency":
                    # TRUE multi-turn Self-Consistency
                    response = run_multi_turn_self_consistency(client, model_config, prompt_def.zero_shot)
                    response["latency_ms"] = 0
                else:
                    # Standard single-call for other styles
                    response = client.invoke(style_prompt, model_config)

                # Special evaluation for Verbalized Sampling
                if style == "verbalized_sampling":
                    eval_result, vs_result = evaluate_vs_response(response["response"], test_case)
                    # Also calculate diversity metrics for VS outputs
                    diversity_result = evaluate_vs_diversity(vs_result)
                else:
                    # Standard evaluation for other styles
                    eval_result = evaluate(response["response"], test_case)
                    vs_result = None
                    diversity_result = None

                # Add rate limiting for Gemini
                if model_config.provider == "gemini":
                    time.sleep(1)

                test_results["styles"][style] = {
                    "response": response["response"],  # Store full response
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

                # Add multi-turn metadata if applicable
                if style in ["tot", "self_consistency"]:
                    test_results["styles"][style]["multi_turn"] = True
                    test_results["styles"][style]["num_calls"] = response.get("num_calls", 1)
                    test_results["styles"][style]["turns"] = response.get("turns", [])

                # Add Verbalized Sampling metadata if applicable
                if style == "verbalized_sampling" and vs_result and diversity_result:
                    test_results["styles"][style]["verbalized_sampling"] = True
                    test_results["styles"][style]["vs_parse_success"] = vs_result.parse_success
                    test_results["styles"][style]["vs_total_responses"] = vs_result.total_responses
                    test_results["styles"][style]["vs_highest_prob"] = vs_result.highest_prob
                    test_results["styles"][style]["vs_any_correct"] = eval_result.details.get("vs_any_correct", False)
                    test_results["styles"][style]["vs_correct_count"] = eval_result.details.get("vs_correct_count", 0)
                    # Diversity metrics (now includes embedding-based from OpenAI)
                    test_results["styles"][style]["diversity"] = {
                        "lexical": diversity_result.lexical_diversity,
                        "semantic": diversity_result.semantic_diversity,
                        "embedding": diversity_result.embedding_diversity,  # OpenAI embeddings (paper methodology)
                        "bigram": diversity_result.bigram_diversity,
                        "trigram": diversity_result.trigram_diversity,
                        "combined": diversity_result.combined_diversity,
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
            "version": "v2.1_with_verbalized_sampling",
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

        # Verbalized Sampling specific metrics
        vs_diversity_scores = []
        vs_embedding_diversity_scores = []
        vs_any_correct_count = 0
        vs_parse_success_count = 0

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

                # Collect Verbalized Sampling metrics
                if style == "verbalized_sampling":
                    if style_data.get("vs_any_correct", False):
                        vs_any_correct_count += 1
                    if style_data.get("vs_parse_success", False):
                        vs_parse_success_count += 1
                    if "diversity" in style_data:
                        vs_diversity_scores.append(style_data["diversity"]["combined"])
                        if style_data["diversity"].get("embedding") is not None:
                            vs_embedding_diversity_scores.append(style_data["diversity"]["embedding"])

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

            # Add Verbalized Sampling specific metrics
            if style == "verbalized_sampling" and vs_diversity_scores:
                summary[style]["avg_diversity"] = round(sum(vs_diversity_scores) / len(vs_diversity_scores), 3)
                summary[style]["any_correct_rate"] = round(vs_any_correct_count / count * 100, 1)
                summary[style]["parse_success_rate"] = round(vs_parse_success_count / count * 100, 1)
                if vs_embedding_diversity_scores:
                    summary[style]["avg_embedding_diversity"] = round(sum(vs_embedding_diversity_scores) / len(vs_embedding_diversity_scores), 3)

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

    # Print Verbalized Sampling metrics if available
    vs_data = summary.get("verbalized_sampling", {})
    if "avg_diversity" in vs_data:
        print(f"\n{'='*80}")
        print("Verbalized Sampling (VS) Metrics - arXiv:2510.01171")
        print("-" * 60)
        if "avg_embedding_diversity" in vs_data:
            print(f"  Embedding Diversity: {vs_data.get('avg_embedding_diversity', 0):.3f}  (OpenAI text-embedding-3-small)")
        print(f"  Combined Diversity:  {vs_data.get('avg_diversity', 0):.3f}  (0=identical, 1=max diverse)")
        print(f"  Parse Success Rate:  {vs_data.get('parse_success_rate', 0):.1f}%")
        print(f"  Any Correct Rate:    {vs_data.get('any_correct_rate', 0):.1f}%  (at least 1 of 5 correct)")
        print(f"  Top-1 Accuracy:      {vs_data.get('accuracy', 0):.1f}%  (highest prob answer)")

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
