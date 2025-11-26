"""
Code Generation Benchmark v2 - With Execution-Based Evaluation

Improvements over v1:
- Actually executes generated code against test cases
- Linting with pylint/node for syntax validation
- Combines execution results (70%) with lint quality (30%)
- Supports both Python and JavaScript

Usage:
    python code_benchmark_v2.py --model nova-micro
    python code_benchmark_v2.py --model claude-haiku --language python
"""

import json
import re
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from pathlib import Path

try:
    from multi_provider_client import MultiProviderClient, MODELS as ALL_MODELS
    USE_MULTI_PROVIDER = True
except ImportError:
    from bedrock_client import BedrockClient, ALL_MODELS
    USE_MULTI_PROVIDER = False

from evaluation import (
    extract_code,
    lint_python_code,
    lint_javascript_code,
    execute_python_code,
    EvalResult,
)


# =============================================================================
# Algorithm Definitions with Test Cases
# =============================================================================

@dataclass
class CodeChallenge:
    """Code generation challenge with test cases."""
    id: str
    name: str
    description: str
    language: str
    function_name: str
    params: str
    examples: str

    # Test cases for execution
    test_cases: list[dict] = field(default_factory=list)

    # Prompts for each style
    prompts: dict = field(default_factory=dict)


PYTHON_CHALLENGES = [
    CodeChallenge(
        id="py-factorial",
        name="Factorial",
        description="Calculate the factorial of a number (n! = n * (n-1) * ... * 1)",
        language="python",
        function_name="factorial",
        params="n",
        examples="factorial(5) = 120, factorial(0) = 1",
        test_cases=[
            {"input": 0, "expected": 1, "function": "factorial"},
            {"input": 1, "expected": 1, "function": "factorial"},
            {"input": 5, "expected": 120, "function": "factorial"},
            {"input": 10, "expected": 3628800, "function": "factorial"},
        ],
    ),
    CodeChallenge(
        id="py-fibonacci",
        name="Fibonacci Sequence",
        description="Return the first n numbers in the Fibonacci sequence as a list",
        language="python",
        function_name="fibonacci",
        params="n",
        examples="fibonacci(5) = [1, 1, 2, 3, 5], fibonacci(1) = [1]",
        test_cases=[
            {"input": 1, "expected": [1], "function": "fibonacci"},
            {"input": 5, "expected": [1, 1, 2, 3, 5], "function": "fibonacci"},
            {"input": 8, "expected": [1, 1, 2, 3, 5, 8, 13, 21], "function": "fibonacci"},
        ],
    ),
    CodeChallenge(
        id="py-gcd",
        name="Greatest Common Divisor",
        description="Find the greatest common divisor of two numbers using the Euclidean algorithm",
        language="python",
        function_name="gcd",
        params="a, b",
        examples="gcd(12, 8) = 4, gcd(17, 13) = 1",
        test_cases=[
            {"input": (12, 8), "expected": 4, "function": "gcd"},
            {"input": (17, 13), "expected": 1, "function": "gcd"},
            {"input": (100, 25), "expected": 25, "function": "gcd"},
            {"input": (48, 18), "expected": 6, "function": "gcd"},
        ],
    ),
    CodeChallenge(
        id="py-isprime",
        name="Primality Test",
        description="Check if a number is prime",
        language="python",
        function_name="is_prime",
        params="n",
        examples="is_prime(17) = True, is_prime(100) = False",
        test_cases=[
            {"input": 1, "expected": False, "function": "is_prime"},
            {"input": 2, "expected": True, "function": "is_prime"},
            {"input": 17, "expected": True, "function": "is_prime"},
            {"input": 100, "expected": False, "function": "is_prime"},
            {"input": 97, "expected": True, "function": "is_prime"},
        ],
    ),
    CodeChallenge(
        id="py-reverse",
        name="Reverse String",
        description="Reverse a string without using built-in reverse functions",
        language="python",
        function_name="reverse_string",
        params="s",
        examples="reverse_string('hello') = 'olleh'",
        test_cases=[
            {"input": "hello", "expected": "olleh", "function": "reverse_string"},
            {"input": "Python", "expected": "nohtyP", "function": "reverse_string"},
            {"input": "", "expected": "", "function": "reverse_string"},
            {"input": "a", "expected": "a", "function": "reverse_string"},
        ],
    ),
    CodeChallenge(
        id="py-twosum",
        name="Two Sum",
        description="Find indices of two numbers in a list that add up to target",
        language="python",
        function_name="two_sum",
        params="nums, target",
        examples="two_sum([2, 7, 11, 15], 9) = [0, 1]",
        test_cases=[
            {"input": ([2, 7, 11, 15], 9), "expected": [0, 1], "function": "two_sum"},
            {"input": ([3, 2, 4], 6), "expected": [1, 2], "function": "two_sum"},
            {"input": ([3, 3], 6), "expected": [0, 1], "function": "two_sum"},
        ],
    ),
]


def generate_prompts(challenge: CodeChallenge) -> dict:
    """Generate prompts for each style for a given challenge."""
    spec = challenge

    return {
        "zero_shot": f"""Write a Python function that implements {spec.name}.

{spec.description}.

Function signature: def {spec.function_name}({spec.params}):

Examples: {spec.examples}

Return only the Python code, no explanations.""",

        "few_shot": f"""Here's an example of a simple Python algorithm:

Example - Sum of list:
```python
def sum_list(arr):
    total = 0
    for num in arr:
        total += num
    return total
```

Now write a Python function that implements {spec.name}.

{spec.description}.

Function signature: def {spec.function_name}({spec.params}):
Examples: {spec.examples}

Return only the Python code.""",

        "cot": f"""Write a Python function that implements {spec.name}.

{spec.description}.

Function signature: def {spec.function_name}({spec.params}):
Examples: {spec.examples}

Think through this step by step:
1. What are the edge cases to handle?
2. What's the core algorithm logic?
3. What variables do I need?
4. Write the implementation

After your reasoning, provide the final Python code.""",

        "schema": f"""ACT=CodeGeneration
LANG=Python
ALGO={spec.name}
DESC={spec.description}
FUNC=def {spec.function_name}({spec.params}):
EXAMPLES={spec.examples}
OUTPUT=Python code only
STYLE=Clean, readable, handle edge cases""",

        "meta": f"""You are tasked with writing a Python function for {spec.name}.

First, decide on your approach:
- What algorithm pattern fits best?
- What's the time/space complexity tradeoff?
- What edge cases need handling?

Then implement: def {spec.function_name}({spec.params}):
Description: {spec.description}
Examples: {spec.examples}

Provide your approach reasoning, then the final Python code.""",

        "gen_knowledge": f"""First, recall key facts about {spec.name}:
- Mathematical definition
- Common implementation patterns
- Edge cases to consider

Now using this knowledge, write a Python function:
Function: def {spec.function_name}({spec.params}):
Description: {spec.description}
Examples: {spec.examples}

Return the Python code.""",

        "directional": f"""Write a Python function for {spec.name}.

HINTS:
- Handle edge cases first (empty input, base cases)
- Use descriptive variable names
- Consider iterative vs recursive approach
- Keep it simple and readable

Function: def {spec.function_name}({spec.params}):
Description: {spec.description}
Examples: {spec.examples}

Return Python code only.""",

        "tot": f"""Write a Python function for {spec.name}.

Explore multiple implementation approaches:

APPROACH A - Iterative:
Consider using a for/while loop

APPROACH B - Recursive:
Consider a recursive solution

APPROACH C - Functional:
Consider using built-in functions or reduce

Evaluate each approach for:
- Readability
- Performance
- Edge case handling

Choose the best approach and implement:
Function: def {spec.function_name}({spec.params}):
Description: {spec.description}
Examples: {spec.examples}

Return the best Python implementation.""",

        "self_consistency": f"""Write a Python function for {spec.name}.

Generate the solution using three different perspectives:
1. As a beginner - focus on clarity
2. As an expert - focus on elegance
3. As a code reviewer - focus on correctness

Function: def {spec.function_name}({spec.params}):
Description: {spec.description}
Examples: {spec.examples}

After considering all three perspectives, provide the final balanced Python implementation.""",
    }


# =============================================================================
# Evaluation
# =============================================================================

@dataclass
class CodeBenchmarkResult:
    """Result of a single code generation benchmark."""
    challenge_id: str
    style: str
    model: str

    # Generated output
    generated_code: str
    raw_response: str

    # Evaluation results
    lint_result: dict
    execution_result: Optional[dict]

    # Scores
    lint_score: float
    execution_score: float
    combined_score: float

    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Pass/fail
    passed: bool


def evaluate_generated_code(
    response: str,
    challenge: CodeChallenge,
) -> tuple[str, dict, Optional[dict], float]:
    """
    Evaluate generated code.

    Returns: (extracted_code, lint_result, exec_result, combined_score)
    """
    # Extract code from response
    code = extract_code(response, challenge.language)

    # Lint the code
    if challenge.language == "python":
        lint_result = lint_python_code(code)

        # Execute if syntax is valid
        if lint_result["syntax_valid"]:
            exec_result = execute_python_code(code, challenge.test_cases)
        else:
            exec_result = {"passed": 0, "total": len(challenge.test_cases), "score": 0.0}
    else:
        lint_result = lint_javascript_code(code)
        exec_result = None  # JS execution not implemented

    # Calculate combined score
    # Execution weighted 70%, lint 30%
    lint_score = lint_result.get("score", 0.0)
    exec_score = exec_result["score"] if exec_result else 0.0

    if exec_result:
        combined_score = (lint_score * 0.3) + (exec_score * 0.7)
    else:
        combined_score = lint_score

    return code, lint_result, exec_result, combined_score


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_code_benchmark(
    model_key: str,
    challenges: Optional[list[CodeChallenge]] = None,
    styles: Optional[list[str]] = None,
) -> list[CodeBenchmarkResult]:
    """Run code generation benchmark with execution-based evaluation."""

    if challenges is None:
        challenges = PYTHON_CHALLENGES

    if styles is None:
        styles = ["zero_shot", "few_shot", "cot", "schema", "meta",
                  "gen_knowledge", "directional", "tot", "self_consistency"]

    if model_key not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(ALL_MODELS.keys())}")

    model_config = ALL_MODELS[model_key]

    if USE_MULTI_PROVIDER:
        client = MultiProviderClient()
    else:
        client = BedrockClient()

    results = []

    print(f"\n{'='*70}")
    print(f"Code Benchmark v2 (Execution-Based)")
    print(f"Model: {model_config.name}")
    print(f"Challenges: {len(challenges)} | Styles: {len(styles)}")
    print(f"{'='*70}\n")

    for challenge in challenges:
        prompts = generate_prompts(challenge)

        print(f"\n{'-'*60}")
        print(f"Challenge: {challenge.name} ({challenge.id})")
        print(f"Test cases: {len(challenge.test_cases)}")
        print(f"{'-'*60}")

        for style in styles:
            if style not in prompts:
                continue

            prompt = prompts[style]
            print(f"\n  Style: {style}...", end=" ", flush=True)

            try:
                response = client.invoke(prompt, model_config)
                raw_response = response["response"]

                # Evaluate the generated code
                code, lint_result, exec_result, combined_score = evaluate_generated_code(
                    raw_response, challenge
                )

                # Determine pass/fail
                passed = False
                if exec_result:
                    passed = exec_result["passed"] == exec_result["total"]

                result = CodeBenchmarkResult(
                    challenge_id=challenge.id,
                    style=style,
                    model=model_key,
                    generated_code=code,
                    raw_response=raw_response[:500],
                    lint_result=lint_result,
                    execution_result=exec_result,
                    lint_score=lint_result.get("score", 0.0),
                    execution_score=exec_result["score"] if exec_result else 0.0,
                    combined_score=combined_score,
                    input_tokens=response["input_tokens"],
                    output_tokens=response["output_tokens"],
                    total_tokens=response["input_tokens"] + response["output_tokens"],
                    passed=passed,
                )
                results.append(result)

                # Print result
                exec_str = f"{exec_result['passed']}/{exec_result['total']}" if exec_result else "N/A"
                status = "[PASS]" if passed else "[FAIL]"
                print(f"{status} Exec: {exec_str} | Lint: {lint_result['syntax_valid']} | Score: {combined_score:.2f} | Tokens: {result.total_tokens}")

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    return results


def summarize_results(results: list[CodeBenchmarkResult]) -> dict:
    """Generate summary statistics by style."""
    by_style = {}

    for r in results:
        if r.style not in by_style:
            by_style[r.style] = {
                "passed": 0,
                "total": 0,
                "combined_scores": [],
                "execution_scores": [],
                "lint_scores": [],
                "tokens": [],
            }

        by_style[r.style]["total"] += 1
        by_style[r.style]["passed"] += 1 if r.passed else 0
        by_style[r.style]["combined_scores"].append(r.combined_score)
        by_style[r.style]["execution_scores"].append(r.execution_score)
        by_style[r.style]["lint_scores"].append(r.lint_score)
        by_style[r.style]["tokens"].append(r.total_tokens)

    summary = {}
    for style, data in by_style.items():
        summary[style] = {
            "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 0,
            "avg_combined_score": sum(data["combined_scores"]) / len(data["combined_scores"]) if data["combined_scores"] else 0,
            "avg_execution_score": sum(data["execution_scores"]) / len(data["execution_scores"]) if data["execution_scores"] else 0,
            "avg_lint_score": sum(data["lint_scores"]) / len(data["lint_scores"]) if data["lint_scores"] else 0,
            "avg_tokens": sum(data["tokens"]) / len(data["tokens"]) if data["tokens"] else 0,
            "total_tests": data["total"],
            "passed": data["passed"],
        }

    return summary


def print_summary(summary: dict, model_name: str):
    """Print formatted summary."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*80}")
    print(f"{'Style':<20} {'Pass Rate':>12} {'Exec Score':>12} {'Lint Score':>12} {'Tokens':>10}")
    print("-" * 70)

    # Sort by pass rate
    for style, data in sorted(summary.items(), key=lambda x: -x[1]["pass_rate"]):
        print(f"{style:<20} {data['pass_rate']:>11.1%} {data['avg_execution_score']:>11.2f} {data['avg_lint_score']:>11.2f} {data['avg_tokens']:>10.0f}")


def save_results(results: list[CodeBenchmarkResult], model_key: str, output_dir: str = "results"):
    """Save results to JSON."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"code_v2_{model_key}_{timestamp}.json"

    output_data = {
        "metadata": {
            "model": model_key,
            "timestamp": timestamp,
            "total_tests": len(results),
            "version": "v2_execution_based",
        },
        "results": [
            {
                "challenge_id": r.challenge_id,
                "style": r.style,
                "passed": r.passed,
                "combined_score": r.combined_score,
                "execution_score": r.execution_score,
                "lint_score": r.lint_score,
                "total_tokens": r.total_tokens,
                "execution_result": r.execution_result,
                "lint_result": {
                    "valid": r.lint_result.get("valid"),
                    "syntax_valid": r.lint_result.get("syntax_valid"),
                    "errors": r.lint_result.get("errors", [])[:3],
                },
                "generated_code": r.generated_code[:500],
            }
            for r in results
        ],
        "summary": summarize_results(results),
    }

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {filename}")
    return filename


def main():
    parser = argparse.ArgumentParser(description="Code Generation Benchmark v2 (Execution-Based)")
    parser.add_argument("--model", default="nova-micro",
                        help="Model to test (nova-micro, claude-haiku, etc.)")
    parser.add_argument("--challenges", nargs="+", default=None,
                        help="Specific challenges to test (by ID)")
    parser.add_argument("--styles", nargs="+", default=None,
                        help="Styles to test")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    args = parser.parse_args()

    # Filter challenges if specified
    challenges = PYTHON_CHALLENGES
    if args.challenges:
        challenges = [c for c in PYTHON_CHALLENGES if c.id in args.challenges]

    print(f"\nCode Generation Benchmark v2")
    print(f"Model: {args.model}")
    print(f"Challenges: {[c.id for c in challenges]}")
    print(f"Styles: {args.styles or 'all'}")

    results = run_code_benchmark(
        model_key=args.model,
        challenges=challenges,
        styles=args.styles,
    )

    if results:
        summary = summarize_results(results)
        print_summary(summary, args.model)
        save_results(results, args.model, args.output_dir)


if __name__ == "__main__":
    main()
