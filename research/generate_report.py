"""
Report Generator for Prompting Style Research

Aggregates benchmark results and generates:
- Summary statistics with ASCII bar charts
- Style-by-style comparison against Zero-shot baseline
- Model-by-model breakdown
- Recommendations by model tier
"""

import json
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class StyleStats:
    """Statistics for a single prompting style."""
    style: str
    total_passed: int
    total_criteria: int
    pass_rate: float
    avg_tokens: float
    vs_baseline_diff: float  # Difference from zero-shot


@dataclass
class ModelResults:
    """Results for a single model across all styles."""
    model_name: str
    model_key: str
    tier: str
    styles: dict[str, StyleStats]


def load_results(results_dir: str = "results") -> list[dict]:
    """Load all multi-style benchmark results."""
    results_path = Path(results_dir)
    all_results = []

    for file in results_path.glob("multi_style_*.json"):
        with open(file) as f:
            data = json.load(f)
            all_results.append(data)

    return all_results


def parse_results(raw_results: list[dict]) -> list[ModelResults]:
    """Parse raw JSON results into structured data."""
    model_results = []

    # Model tier mapping
    tier_map = {
        "nova-micro": "budget",
        "nova-lite": "budget",
        "mistral-7b": "budget",
        "llama3-8b": "budget",
        "mistral-small": "budget",
        "nova-pro": "mid",
        "mixtral-8x7b": "mid",
        "llama3-70b": "mid",
        "claude-haiku": "mid",
        "mistral-large": "premium",
        "claude-sonnet": "premium",
        "command-r-plus": "premium",
    }

    for data in raw_results:
        model_key = data["metadata"]["model"]

        # Aggregate stats by style
        style_stats = defaultdict(lambda: {"passed": 0, "total": 0, "tokens": 0, "count": 0})

        for result in data["results"]:
            for style, style_data in result["styles"].items():
                style_stats[style]["passed"] += style_data["criteria_passed"]
                style_stats[style]["total"] += style_data["criteria_total"]
                style_stats[style]["tokens"] += style_data["total_tokens"]
                style_stats[style]["count"] += 1

        # Calculate rates and create StyleStats
        styles = {}
        zero_shot_rate = 0

        for style, stats in style_stats.items():
            pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            avg_tokens = stats["tokens"] / stats["count"] if stats["count"] > 0 else 0

            if style == "zero_shot":
                zero_shot_rate = pass_rate

            styles[style] = StyleStats(
                style=style,
                total_passed=stats["passed"],
                total_criteria=stats["total"],
                pass_rate=pass_rate,
                avg_tokens=avg_tokens,
                vs_baseline_diff=0  # Will be set after all styles processed
            )

        # Set baseline differences
        for style_name, style_stat in styles.items():
            style_stat.vs_baseline_diff = style_stat.pass_rate - zero_shot_rate

        model_results.append(ModelResults(
            model_name=model_key,
            model_key=model_key,
            tier=tier_map.get(model_key, "unknown"),
            styles=styles
        ))

    return model_results


def ascii_bar(value: float, max_value: float = 100, width: int = 30) -> str:
    """Create an ASCII bar chart."""
    filled = int((value / max_value) * width)
    empty = width - filled
    return "#" * filled + "-" * empty


def generate_report(model_results: list[ModelResults]) -> str:
    """Generate the full research report."""
    lines = []

    lines.append("=" * 80)
    lines.append("PROMPTING STYLE RESEARCH REPORT")
    lines.append("Comparing Zero-shot vs Schema vs Chain-of-Thought")
    lines.append("=" * 80)
    lines.append("")

    # Executive Summary
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 40)

    # Aggregate across all models
    all_zero_shot = {"passed": 0, "total": 0, "tokens": 0}
    all_schema = {"passed": 0, "total": 0, "tokens": 0}
    all_cot = {"passed": 0, "total": 0, "tokens": 0}

    for mr in model_results:
        if "zero_shot" in mr.styles:
            all_zero_shot["passed"] += mr.styles["zero_shot"].total_passed
            all_zero_shot["total"] += mr.styles["zero_shot"].total_criteria
            all_zero_shot["tokens"] += mr.styles["zero_shot"].avg_tokens
        if "schema" in mr.styles:
            all_schema["passed"] += mr.styles["schema"].total_passed
            all_schema["total"] += mr.styles["schema"].total_criteria
            all_schema["tokens"] += mr.styles["schema"].avg_tokens
        if "cot" in mr.styles:
            all_cot["passed"] += mr.styles["cot"].total_passed
            all_cot["total"] += mr.styles["cot"].total_criteria
            all_cot["tokens"] += mr.styles["cot"].avg_tokens

    n_models = len(model_results)

    zs_rate = (all_zero_shot["passed"] / all_zero_shot["total"] * 100) if all_zero_shot["total"] > 0 else 0
    schema_rate = (all_schema["passed"] / all_schema["total"] * 100) if all_schema["total"] > 0 else 0
    cot_rate = (all_cot["passed"] / all_cot["total"] * 100) if all_cot["total"] > 0 else 0

    zs_tokens = all_zero_shot["tokens"] / n_models if n_models > 0 else 0
    schema_tokens = all_schema["tokens"] / n_models if n_models > 0 else 0
    cot_tokens = all_cot["tokens"] / n_models if n_models > 0 else 0

    lines.append(f"Models tested: {n_models}")
    lines.append(f"Test cases per model: 12")
    lines.append(f"Criteria evaluated: {all_zero_shot['total']} (across all models)")
    lines.append("")

    lines.append("OVERALL PASS RATES (all models combined):")
    lines.append("")
    lines.append(f"  Zero-shot (baseline): {zs_rate:5.1f}% [{ascii_bar(zs_rate)}]")
    lines.append(f"  Schema:               {schema_rate:5.1f}% [{ascii_bar(schema_rate)}]  ({schema_rate - zs_rate:+.1f}% vs baseline)")
    lines.append(f"  Chain-of-Thought:     {cot_rate:5.1f}% [{ascii_bar(cot_rate)}]  ({cot_rate - zs_rate:+.1f}% vs baseline)")
    lines.append("")

    lines.append("AVERAGE TOKEN USAGE:")
    lines.append("")
    max_tokens = max(zs_tokens, schema_tokens, cot_tokens)
    lines.append(f"  Zero-shot:        {zs_tokens:6.0f} [{ascii_bar(zs_tokens, max_tokens, 20)}]")
    schema_diff = ((schema_tokens - zs_tokens) / zs_tokens * 100) if zs_tokens > 0 else 0
    cot_diff = ((cot_tokens - zs_tokens) / zs_tokens * 100) if zs_tokens > 0 else 0
    lines.append(f"  Schema:           {schema_tokens:6.0f} [{ascii_bar(schema_tokens, max_tokens, 20)}]  ({schema_diff:+.1f}% vs baseline)")
    lines.append(f"  Chain-of-Thought: {cot_tokens:6.0f} [{ascii_bar(cot_tokens, max_tokens, 20)}]  ({cot_diff:+.1f}% vs baseline)")
    lines.append("")

    # Model-by-Model Breakdown
    lines.append("=" * 80)
    lines.append("MODEL-BY-MODEL BREAKDOWN")
    lines.append("=" * 80)
    lines.append("")

    # Sort by tier
    tier_order = {"budget": 0, "mid": 1, "premium": 2, "unknown": 3}
    sorted_results = sorted(model_results, key=lambda x: (tier_order.get(x.tier, 3), x.model_name))

    for mr in sorted_results:
        lines.append(f"  {mr.model_name.upper()} ({mr.tier})")
        lines.append("  " + "-" * 50)

        zs = mr.styles.get("zero_shot")
        schema = mr.styles.get("schema")
        cot = mr.styles.get("cot")

        if zs:
            lines.append(f"    Zero-shot:        {zs.pass_rate:5.1f}% | {zs.avg_tokens:5.0f} tokens")
        if schema:
            diff_str = f"({schema.vs_baseline_diff:+.1f}%)" if schema.vs_baseline_diff != 0 else "(same)"
            lines.append(f"    Schema:           {schema.pass_rate:5.1f}% | {schema.avg_tokens:5.0f} tokens  {diff_str}")
        if cot:
            diff_str = f"({cot.vs_baseline_diff:+.1f}%)" if cot.vs_baseline_diff != 0 else "(same)"
            lines.append(f"    Chain-of-Thought: {cot.pass_rate:5.1f}% | {cot.avg_tokens:5.0f} tokens  {diff_str}")

        # Determine best style
        best_style = "zero_shot"
        best_rate = zs.pass_rate if zs else 0
        if schema and schema.pass_rate > best_rate:
            best_style = "schema"
            best_rate = schema.pass_rate
        if cot and cot.pass_rate > best_rate:
            best_style = "cot"
            best_rate = cot.pass_rate

        lines.append(f"    --> Best: {best_style.replace('_', '-').title()}")
        lines.append("")

    # Analysis by Tier
    lines.append("=" * 80)
    lines.append("ANALYSIS BY MODEL TIER")
    lines.append("=" * 80)
    lines.append("")

    tier_stats = defaultdict(lambda: {"zero_shot": [], "schema": [], "cot": []})

    for mr in model_results:
        for style_name, style_stat in mr.styles.items():
            tier_stats[mr.tier][style_name].append(style_stat.pass_rate)

    for tier in ["budget", "mid", "premium"]:
        if tier not in tier_stats:
            continue

        stats = tier_stats[tier]
        lines.append(f"  {tier.upper()} TIER MODELS:")
        lines.append("  " + "-" * 40)

        for style in ["zero_shot", "schema", "cot"]:
            if style in stats and stats[style]:
                avg = sum(stats[style]) / len(stats[style])
                lines.append(f"    {style.replace('_', '-').title():20} Avg: {avg:5.1f}%")

        # Show improvement potential
        zs_avg = sum(stats["zero_shot"]) / len(stats["zero_shot"]) if stats["zero_shot"] else 0
        schema_avg = sum(stats["schema"]) / len(stats["schema"]) if stats["schema"] else 0
        cot_avg = sum(stats["cot"]) / len(stats["cot"]) if stats["cot"] else 0

        best = max(schema_avg - zs_avg, cot_avg - zs_avg)
        if best > 0:
            lines.append(f"    --> Potential improvement: +{best:.1f}%")
        else:
            lines.append(f"    --> Zero-shot performs best or equal")
        lines.append("")

    # Key Findings
    lines.append("=" * 80)
    lines.append("KEY FINDINGS")
    lines.append("=" * 80)
    lines.append("")

    # Calculate key insights
    schema_wins = sum(1 for mr in model_results
                      if mr.styles.get("schema") and
                      mr.styles["schema"].pass_rate > mr.styles.get("zero_shot", StyleStats("", 0, 0, 0, 0, 0)).pass_rate)
    cot_wins = sum(1 for mr in model_results
                   if mr.styles.get("cot") and
                   mr.styles["cot"].pass_rate > mr.styles.get("zero_shot", StyleStats("", 0, 0, 0, 0, 0)).pass_rate)

    lines.append(f"1. Schema prompting outperformed Zero-shot in {schema_wins}/{n_models} models")
    lines.append(f"2. Chain-of-Thought outperformed Zero-shot in {cot_wins}/{n_models} models")
    lines.append(f"3. Schema adds ~{schema_diff:.0f}% token overhead vs Zero-shot")
    lines.append(f"4. CoT adds ~{cot_diff:.0f}% token overhead vs Zero-shot")
    lines.append("")

    # Recommendations
    lines.append("=" * 80)
    lines.append("RECOMMENDATIONS BY USE CASE")
    lines.append("=" * 80)
    lines.append("")

    lines.append("When to use each style:")
    lines.append("")
    lines.append("  ZERO-SHOT (Plain English):")
    lines.append("    - Premium/advanced models (Claude, GPT-4)")
    lines.append("    - Simple, straightforward tasks")
    lines.append("    - When token cost is a concern")
    lines.append("    - When speed is critical")
    lines.append("")
    lines.append("  SCHEMA (Structured Prompts):")
    lines.append("    - Budget/smaller models")
    lines.append("    - Tasks requiring specific output format")
    lines.append("    - When persona/context matters")
    lines.append("    - Complex multi-constraint tasks")
    lines.append("")
    lines.append("  CHAIN-OF-THOUGHT:")
    lines.append("    - Mathematical/logical reasoning")
    lines.append("    - Multi-step problem solving")
    lines.append("    - When accuracy > token efficiency")
    lines.append("    - Complex analysis tasks")
    lines.append("")

    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    """Generate report from benchmark results."""
    print("Loading benchmark results...")
    raw_results = load_results()

    if not raw_results:
        print("No multi-style benchmark results found in results/ directory")
        print("Run multi_style_benchmark.py first to generate results")
        return

    print(f"Found {len(raw_results)} benchmark result files")

    model_results = parse_results(raw_results)
    report = generate_report(model_results)

    # Print to console
    print("\n")
    print(report)

    # Save to file
    output_file = Path("results") / "prompting_style_report.txt"
    with open(output_file, "w") as f:
        f.write(report)

    print(f"\nReport saved to: {output_file}")


if __name__ == "__main__":
    main()
