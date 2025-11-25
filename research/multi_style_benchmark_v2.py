"""
Multi-Style Prompting Benchmark Runner v2

Compares comprehensive prompting techniques:
- Zero-shot: Plain natural language
- Few-shot: With examples
- Chain-of-Thought: Step-by-step reasoning
- Schema: Structured format (Ailo)
- Meta: LLM designs its own approach
- Generate Knowledge: Generate facts first
- Directional Stimulus: Hints/keywords guide
- Tree of Thoughts: Explore multiple paths (single-call simulation)
- Self-Consistency: Multiple approaches, reconcile (single-call simulation)

Note on complex techniques:
- True Self-Consistency requires multiple API calls + voting
- True ToT requires branching + backtracking
- We simulate these within single prompts for fair comparison
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from bedrock_client import BedrockClient, ALL_MODELS, ModelConfig
from test_prompts_v3 import MULTI_STYLE_PROMPTS, MultiStylePrompt, PromptStyle, TaskCategory

console = Console()

# All testable styles
ALL_STYLES = [
    "zero_shot",
    "few_shot",
    "cot",
    "schema",
    "meta",
    "gen_knowledge",
    "directional",
    "tot",
    "self_consistency"
]


@dataclass
class StyleResult:
    """Result from a single prompt style execution."""
    prompt_id: str
    style: str
    model_id: str
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    criteria_results: dict[str, bool]
    criteria_passed: int
    criteria_total: int
    timestamp: str


@dataclass
class MultiStyleComparison:
    """Comparison of all styles for a single prompt."""
    prompt_id: str
    prompt_name: str
    category: str
    model_id: str
    results: dict[str, StyleResult]


class MultiStyleBenchmarkRunner:
    """Runs benchmarks comparing multiple prompting styles."""

    def __init__(self, output_dir: str = "results"):
        self.client = BedrockClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_single_style(
        self,
        prompt_text: str,
        prompt_id: str,
        style: str,
        model_config: ModelConfig,
        test_prompt: MultiStylePrompt,
    ) -> StyleResult:
        """Run a single prompt style and evaluate results."""
        start_time = time.time()

        result = self.client.invoke(
            prompt=prompt_text,
            model_config=model_config,
            temperature=0.7,
        )

        latency_ms = (time.time() - start_time) * 1000

        # Evaluate criteria
        criteria_results = {}
        for criteria in test_prompt.criteria:
            try:
                passed = criteria.check(result["response"])
                criteria_results[criteria.name] = passed
            except Exception as e:
                criteria_results[criteria.name] = False

        criteria_passed = sum(1 for v in criteria_results.values() if v)
        criteria_total = len(criteria_results)

        return StyleResult(
            prompt_id=prompt_id,
            style=style,
            model_id=result["model"],
            response=result["response"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_tokens=result["input_tokens"] + result["output_tokens"],
            latency_ms=latency_ms,
            criteria_results=criteria_results,
            criteria_passed=criteria_passed,
            criteria_total=criteria_total,
            timestamp=datetime.now().isoformat(),
        )

    def run_multi_style_comparison(
        self,
        test_prompt: MultiStylePrompt,
        model_config: ModelConfig,
        styles: list[str] = None,
    ) -> MultiStyleComparison:
        """Run all prompt styles for a single test case."""
        if styles is None:
            styles = ALL_STYLES

        results = {}

        for style in styles:
            prompt_text = getattr(test_prompt, style, None)
            if prompt_text is None:
                continue

            result = self.run_single_style(
                prompt_text=prompt_text,
                prompt_id=test_prompt.id,
                style=style,
                model_config=model_config,
                test_prompt=test_prompt,
            )
            results[style] = result
            time.sleep(0.5)  # Rate limiting

        return MultiStyleComparison(
            prompt_id=test_prompt.id,
            prompt_name=test_prompt.name,
            category=test_prompt.category.value,
            model_id=model_config.model_id,
            results=results,
        )

    def run_benchmark(
        self,
        model_key: str = "nova-micro",
        prompt_ids: Optional[list[str]] = None,
        styles: list[str] = None,
    ) -> list[MultiStyleComparison]:
        """Run full multi-style benchmark suite."""
        model_config = ALL_MODELS[model_key]
        comparisons = []

        if styles is None:
            styles = ALL_STYLES

        prompts = MULTI_STYLE_PROMPTS
        if prompt_ids:
            prompts = [p for p in MULTI_STYLE_PROMPTS if p.id in prompt_ids]

        console.print(Panel(
            f"[bold]Comprehensive Prompting Style Benchmark[/bold]\n\n"
            f"Model: {model_config.name} ({model_config.model_id})\n"
            f"Prompts: {len(prompts)}\n"
            f"Styles: {len(styles)} ({', '.join(styles[:4])}...)",
            title="Benchmark Configuration"
        ))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running tests...", total=len(prompts))

            for test_prompt in prompts:
                progress.update(task, description=f"Testing: {test_prompt.name}")

                try:
                    comparison = self.run_multi_style_comparison(
                        test_prompt, model_config, styles
                    )
                    comparisons.append(comparison)

                    # Show quick result
                    style_scores = []
                    for style, result in comparison.results.items():
                        style_scores.append(f"{style[:4]}:{result.criteria_passed}/{result.criteria_total}")
                    console.print(f"  [dim]{test_prompt.id}[/dim] {' | '.join(style_scores[:5])}")

                except Exception as e:
                    console.print(f"[red]Error on {test_prompt.id}: {e}[/red]")

                progress.advance(task)
                time.sleep(1)

        return comparisons

    def save_results(self, comparisons: list[MultiStyleComparison], model_key: str):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"comprehensive_{model_key}_{timestamp}.json"

        data = {
            "metadata": {
                "model": model_key,
                "timestamp": timestamp,
                "total_tests": len(comparisons),
                "styles_tested": ALL_STYLES,
            },
            "results": [],
        }

        for comp in comparisons:
            result_entry = {
                "prompt_id": comp.prompt_id,
                "prompt_name": comp.prompt_name,
                "category": comp.category,
                "model_id": comp.model_id,
                "styles": {},
            }
            for style, result in comp.results.items():
                result_entry["styles"][style] = {
                    "response": result.response,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                    "latency_ms": result.latency_ms,
                    "criteria_results": result.criteria_results,
                    "criteria_passed": result.criteria_passed,
                    "criteria_total": result.criteria_total,
                }
            data["results"].append(result_entry)

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        console.print(f"\n[green]Results saved to: {filename}[/green]")
        return filename

    def print_summary(self, comparisons: list[MultiStyleComparison]):
        """Print a summary table of results."""
        console.print("\n")

        # Collect all styles that have results
        all_tested_styles = set()
        for comp in comparisons:
            all_tested_styles.update(comp.results.keys())

        tested_styles = [s for s in ALL_STYLES if s in all_tested_styles]

        # Summary table
        table = Table(title="Comprehensive Prompting Style Results")
        table.add_column("Test", style="cyan", max_width=20)
        table.add_column("Cat", style="dim", max_width=8)

        # Add column for each style (abbreviated)
        style_abbrevs = {
            "zero_shot": "Zero",
            "few_shot": "Few",
            "cot": "CoT",
            "schema": "Schema",
            "meta": "Meta",
            "gen_knowledge": "GenK",
            "directional": "Dir",
            "tot": "ToT",
            "self_consistency": "Self"
        }

        for style in tested_styles:
            table.add_column(style_abbrevs.get(style, style[:4]), justify="center", max_width=6)

        table.add_column("Best", justify="center", max_width=8)

        style_totals = {s: {"passed": 0, "total": 0, "tokens": 0, "count": 0} for s in tested_styles}

        for comp in comparisons:
            row = [comp.prompt_name[:18], comp.category[:7]]

            best_style = None
            best_score = -1

            for style in tested_styles:
                if style in comp.results:
                    result = comp.results[style]
                    score_str = f"{result.criteria_passed}/{result.criteria_total}"
                    row.append(score_str)

                    style_totals[style]["passed"] += result.criteria_passed
                    style_totals[style]["total"] += result.criteria_total
                    style_totals[style]["tokens"] += result.total_tokens
                    style_totals[style]["count"] += 1

                    if result.criteria_passed > best_score:
                        best_score = result.criteria_passed
                        best_style = style
                else:
                    row.append("-")

            # Best style
            if best_style:
                row.append(f"[green]{style_abbrevs.get(best_style, best_style[:4])}[/green]")
            else:
                row.append("-")

            table.add_row(*row)

        console.print(table)

        # Overall statistics
        console.print("\n[bold]Overall Statistics (vs Zero-shot baseline):[/bold]\n")

        # Get zero-shot baseline
        zs = style_totals.get("zero_shot", {"passed": 0, "total": 1, "tokens": 0, "count": 1})
        zs_rate = zs["passed"] / zs["total"] * 100 if zs["total"] > 0 else 0
        zs_tokens = zs["tokens"] / zs["count"] if zs["count"] > 0 else 0

        console.print(f"  {'Style':<20} {'Pass Rate':>10} {'vs Zero':>10} {'Avg Tokens':>12} {'Token Diff':>12}")
        console.print("  " + "-" * 66)

        for style in tested_styles:
            totals = style_totals[style]
            if totals["total"] > 0 and totals["count"] > 0:
                pct = totals["passed"] / totals["total"] * 100
                avg_tokens = totals["tokens"] / totals["count"]

                vs_zero = pct - zs_rate
                token_diff = ((avg_tokens - zs_tokens) / zs_tokens * 100) if zs_tokens > 0 else 0

                vs_str = f"{vs_zero:+.1f}%" if style != "zero_shot" else "baseline"
                token_str = f"{token_diff:+.1f}%" if style != "zero_shot" else "baseline"

                style_name = style.replace("_", "-").title()
                console.print(f"  {style_name:<20} {pct:>9.1f}% {vs_str:>10} {avg_tokens:>11.0f} {token_str:>12}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run comprehensive prompting style benchmark")
    parser.add_argument(
        "--model",
        default="nova-micro",
        choices=list(ALL_MODELS.keys()),
        help="Model to use for testing"
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Specific prompt IDs to test (default: all)"
    )
    parser.add_argument(
        "--styles",
        nargs="*",
        choices=ALL_STYLES,
        help="Specific styles to test (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run first 2 prompts"
    )

    args = parser.parse_args()

    runner = MultiStyleBenchmarkRunner()

    prompt_ids = args.prompts
    if args.dry_run:
        prompt_ids = [MULTI_STYLE_PROMPTS[0].id, MULTI_STYLE_PROMPTS[1].id]
        console.print("[yellow]Dry run mode: testing only 2 prompts[/yellow]\n")

    styles = args.styles if args.styles else ALL_STYLES

    comparisons = runner.run_benchmark(
        model_key=args.model,
        prompt_ids=prompt_ids,
        styles=styles
    )

    if comparisons:
        runner.print_summary(comparisons)
        runner.save_results(comparisons, args.model)


if __name__ == "__main__":
    main()
