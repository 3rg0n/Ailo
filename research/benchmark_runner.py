"""
Benchmark runner for Ailo vs Plain English prompts.
Runs prompts through AWS Bedrock and evaluates results.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from bedrock_client import BedrockClient, MODELS, ALL_MODELS, QUICK_MODELS, ModelConfig
from test_prompts import TEST_PROMPTS, TestPrompt, TaskCategory


console = Console()


@dataclass
class PromptResult:
    """Result from a single prompt execution."""
    prompt_id: str
    prompt_type: str  # "plain" or "ailo"
    model_id: str
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    criteria_results: dict[str, bool]  # criteria_name -> passed
    timestamp: str


@dataclass
class ComparisonResult:
    """Comparison between plain and Ailo prompts."""
    prompt_id: str
    prompt_name: str
    category: str
    model_id: str

    plain_result: PromptResult
    ailo_result: PromptResult

    # Computed metrics
    token_diff: int  # ailo_tokens - plain_tokens (negative = ailo used fewer)
    token_diff_percent: float
    plain_criteria_passed: int
    ailo_criteria_passed: int
    criteria_improvement: int  # ailo_passed - plain_passed


class BenchmarkRunner:
    """Runs benchmarks comparing plain vs Ailo prompts."""

    def __init__(self, output_dir: str = "results"):
        self.client = BedrockClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_single_prompt(
        self,
        prompt_text: str,
        prompt_id: str,
        prompt_type: str,
        model_config: ModelConfig,
        test_prompt: TestPrompt,
    ) -> PromptResult:
        """Run a single prompt and evaluate results."""
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
                console.print(f"[yellow]Warning: Criteria '{criteria.name}' failed: {e}[/yellow]")
                criteria_results[criteria.name] = False

        return PromptResult(
            prompt_id=prompt_id,
            prompt_type=prompt_type,
            model_id=result["model"],
            response=result["response"],
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            total_tokens=result["input_tokens"] + result["output_tokens"],
            latency_ms=latency_ms,
            criteria_results=criteria_results,
            timestamp=datetime.now().isoformat(),
        )

    def run_comparison(
        self,
        test_prompt: TestPrompt,
        model_config: ModelConfig,
    ) -> ComparisonResult:
        """Run both plain and Ailo prompts and compare."""
        # Run plain prompt
        plain_result = self.run_single_prompt(
            prompt_text=test_prompt.plain_prompt,
            prompt_id=test_prompt.id,
            prompt_type="plain",
            model_config=model_config,
            test_prompt=test_prompt,
        )

        # Small delay to avoid rate limiting
        time.sleep(0.5)

        # Run Ailo prompt
        ailo_result = self.run_single_prompt(
            prompt_text=test_prompt.ailo_prompt,
            prompt_id=test_prompt.id,
            prompt_type="ailo",
            model_config=model_config,
            test_prompt=test_prompt,
        )

        # Calculate metrics
        plain_total = plain_result.total_tokens
        ailo_total = ailo_result.total_tokens
        token_diff = ailo_total - plain_total
        token_diff_percent = ((ailo_total - plain_total) / plain_total * 100) if plain_total > 0 else 0

        plain_passed = sum(1 for v in plain_result.criteria_results.values() if v)
        ailo_passed = sum(1 for v in ailo_result.criteria_results.values() if v)

        return ComparisonResult(
            prompt_id=test_prompt.id,
            prompt_name=test_prompt.name,
            category=test_prompt.category.value,
            model_id=model_config.model_id,
            plain_result=plain_result,
            ailo_result=ailo_result,
            token_diff=token_diff,
            token_diff_percent=token_diff_percent,
            plain_criteria_passed=plain_passed,
            ailo_criteria_passed=ailo_passed,
            criteria_improvement=ailo_passed - plain_passed,
        )

    def run_benchmark(
        self,
        model_key: str = "nova-micro",
        prompt_ids: Optional[list[str]] = None,
    ) -> list[ComparisonResult]:
        """Run full benchmark suite."""
        model_config = ALL_MODELS[model_key]
        results = []

        # Filter prompts if specific IDs provided
        prompts = TEST_PROMPTS
        if prompt_ids:
            prompts = [p for p in TEST_PROMPTS if p.id in prompt_ids]

        console.print(Panel(
            f"[bold]Running Ailo Benchmark[/bold]\n\n"
            f"Model: {model_config.name} ({model_config.model_id})\n"
            f"Prompts: {len(prompts)}\n"
            f"Region: {model_config.region}",
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
                    comparison = self.run_comparison(test_prompt, model_config)
                    results.append(comparison)

                    # Show quick result
                    criteria_change = comparison.criteria_improvement
                    change_symbol = "+" if criteria_change > 0 else ("-" if criteria_change < 0 else "=")
                    token_symbol = "-" if comparison.token_diff < 0 else ("+" if comparison.token_diff > 0 else "=")

                    console.print(
                        f"  [dim]{test_prompt.id}[/dim] "
                        f"Criteria: {comparison.plain_criteria_passed}->{comparison.ailo_criteria_passed} {change_symbol}  "
                        f"Tokens: {comparison.plain_result.total_tokens}->{comparison.ailo_result.total_tokens} {token_symbol}"
                    )

                except Exception as e:
                    console.print(f"[red]Error on {test_prompt.id}: {e}[/red]")

                progress.advance(task)
                time.sleep(1)  # Rate limiting

        return results

    def save_results(self, results: list[ComparisonResult], model_key: str):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"benchmark_{model_key}_{timestamp}.json"

        # Convert to serializable format
        data = {
            "metadata": {
                "model": model_key,
                "timestamp": timestamp,
                "total_tests": len(results),
            },
            "results": [],
        }

        for r in results:
            data["results"].append({
                "prompt_id": r.prompt_id,
                "prompt_name": r.prompt_name,
                "category": r.category,
                "model_id": r.model_id,
                "plain": {
                    "response": r.plain_result.response,
                    "input_tokens": r.plain_result.input_tokens,
                    "output_tokens": r.plain_result.output_tokens,
                    "total_tokens": r.plain_result.total_tokens,
                    "latency_ms": r.plain_result.latency_ms,
                    "criteria_results": r.plain_result.criteria_results,
                },
                "ailo": {
                    "response": r.ailo_result.response,
                    "input_tokens": r.ailo_result.input_tokens,
                    "output_tokens": r.ailo_result.output_tokens,
                    "total_tokens": r.ailo_result.total_tokens,
                    "latency_ms": r.ailo_result.latency_ms,
                    "criteria_results": r.ailo_result.criteria_results,
                },
                "comparison": {
                    "token_diff": r.token_diff,
                    "token_diff_percent": r.token_diff_percent,
                    "plain_criteria_passed": r.plain_criteria_passed,
                    "ailo_criteria_passed": r.ailo_criteria_passed,
                    "criteria_improvement": r.criteria_improvement,
                },
            })

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        console.print(f"\n[green]Results saved to: {filename}[/green]")
        return filename

    def print_summary(self, results: list[ComparisonResult]):
        """Print a summary table of results."""
        console.print("\n")

        # Summary table
        table = Table(title="Benchmark Results Summary")
        table.add_column("Test", style="cyan")
        table.add_column("Category", style="dim")
        table.add_column("Plain Criteria", justify="center")
        table.add_column("Ailo Criteria", justify="center")
        table.add_column("Diff", justify="center")
        table.add_column("Plain Tokens", justify="right")
        table.add_column("Ailo Tokens", justify="right")
        table.add_column("Token Diff", justify="right")

        total_plain_criteria = 0
        total_ailo_criteria = 0
        total_plain_tokens = 0
        total_ailo_tokens = 0
        total_criteria_count = 0

        for r in results:
            criteria_count = len(r.plain_result.criteria_results)
            total_criteria_count += criteria_count

            # Determine style for criteria change
            if r.criteria_improvement > 0:
                criteria_style = "[green]+{}[/green]".format(r.criteria_improvement)
            elif r.criteria_improvement < 0:
                criteria_style = "[red]{}[/red]".format(r.criteria_improvement)
            else:
                criteria_style = "[dim]0[/dim]"

            # Determine style for token change
            if r.token_diff < 0:
                token_style = "[green]{}[/green]".format(r.token_diff)
            elif r.token_diff > 0:
                token_style = "[red]+{}[/red]".format(r.token_diff)
            else:
                token_style = "[dim]0[/dim]"

            table.add_row(
                r.prompt_name,
                r.category,
                f"{r.plain_criteria_passed}/{criteria_count}",
                f"{r.ailo_criteria_passed}/{criteria_count}",
                criteria_style,
                str(r.plain_result.total_tokens),
                str(r.ailo_result.total_tokens),
                token_style,
            )

            total_plain_criteria += r.plain_criteria_passed
            total_ailo_criteria += r.ailo_criteria_passed
            total_plain_tokens += r.plain_result.total_tokens
            total_ailo_tokens += r.ailo_result.total_tokens

        console.print(table)

        # Overall statistics
        console.print("\n[bold]Overall Statistics:[/bold]")

        criteria_improvement = total_ailo_criteria - total_plain_criteria
        criteria_pct = (total_ailo_criteria / total_criteria_count * 100) if total_criteria_count > 0 else 0
        plain_criteria_pct = (total_plain_criteria / total_criteria_count * 100) if total_criteria_count > 0 else 0

        token_diff = total_ailo_tokens - total_plain_tokens
        token_diff_pct = (token_diff / total_plain_tokens * 100) if total_plain_tokens > 0 else 0

        console.print(f"  Criteria Pass Rate:")
        console.print(f"    Plain: {total_plain_criteria}/{total_criteria_count} ({plain_criteria_pct:.1f}%)")
        console.print(f"    Ailo:  {total_ailo_criteria}/{total_criteria_count} ({criteria_pct:.1f}%)")

        if criteria_improvement > 0:
            console.print(f"    [green]Ailo improved by {criteria_improvement} criteria (+{criteria_improvement/total_criteria_count*100:.1f}%)[/green]")
        elif criteria_improvement < 0:
            console.print(f"    [red]Ailo worse by {abs(criteria_improvement)} criteria ({criteria_improvement/total_criteria_count*100:.1f}%)[/red]")
        else:
            console.print(f"    [dim]No difference[/dim]")

        console.print(f"\n  Token Usage:")
        console.print(f"    Plain: {total_plain_tokens} tokens")
        console.print(f"    Ailo:  {total_ailo_tokens} tokens")

        if token_diff > 0:
            console.print(f"    [yellow]Ailo used {token_diff} more tokens (+{token_diff_pct:.1f}%)[/yellow]")
        elif token_diff < 0:
            console.print(f"    [green]Ailo saved {abs(token_diff)} tokens ({token_diff_pct:.1f}%)[/green]")
        else:
            console.print(f"    [dim]No difference[/dim]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Ailo benchmark")
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
        "--dry-run",
        action="store_true",
        help="Only run first 2 prompts"
    )

    args = parser.parse_args()

    runner = BenchmarkRunner()

    prompt_ids = args.prompts
    if args.dry_run:
        prompt_ids = [TEST_PROMPTS[0].id, TEST_PROMPTS[1].id]
        console.print("[yellow]Dry run mode: testing only 2 prompts[/yellow]\n")

    results = runner.run_benchmark(model_key=args.model, prompt_ids=prompt_ids)

    if results:
        runner.print_summary(results)
        runner.save_results(results, args.model)


if __name__ == "__main__":
    main()
