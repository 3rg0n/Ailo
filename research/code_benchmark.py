"""
Code Generation Benchmark - Comparing prompting styles for code generation tasks.

Tests each prompting style's ability to generate algorithm implementations
that match reference implementations from javascript-algorithms repository.
"""

import json
import re
import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
try:
    from multi_provider_client import MultiProviderClient, MODELS as ALL_MODELS
    USE_MULTI_PROVIDER = True
except ImportError:
    from bedrock_client import BedrockClient, ALL_MODELS
    USE_MULTI_PROVIDER = False

# Reference implementations (JavaScript)
REFERENCE_CODE = {
    "factorial": '''export default function factorial(number) {
  let result = 1;

  for (let i = 2; i <= number; i += 1) {
    result *= i;
  }

  return result;
}''',

    "fibonacci": '''export default function fibonacci(n) {
  const fibSequence = [1];

  let currentValue = 1;
  let previousValue = 0;

  if (n === 1) {
    return fibSequence;
  }

  let iterationsCounter = n - 1;

  while (iterationsCounter) {
    currentValue += previousValue;
    previousValue = currentValue - previousValue;

    fibSequence.push(currentValue);

    iterationsCounter -= 1;
  }

  return fibSequence;
}''',

    "euclidean_gcd": '''export default function euclideanAlgorithm(originalA, originalB) {
  const a = Math.abs(originalA);
  const b = Math.abs(originalB);

  return (b === 0) ? a : euclideanAlgorithm(b, a % b);
}''',

    "is_prime": '''export default function trialDivision(number) {
  if (number % 1 !== 0) {
    return false;
  }

  if (number <= 1) {
    return false;
  }

  if (number <= 3) {
    return true;
  }

  if (number % 2 === 0) {
    return false;
  }

  const dividerLimit = Math.sqrt(number);
  for (let divider = 3; divider <= dividerLimit; divider += 2) {
    if (number % divider === 0) {
      return false;
    }
  }

  return true;
}'''
}

# Test cases for functional correctness
TEST_CASES = {
    "factorial": [
        {"input": 0, "expected": 1},
        {"input": 1, "expected": 1},
        {"input": 5, "expected": 120},
        {"input": 10, "expected": 3628800},
    ],
    "fibonacci": [
        {"input": 1, "expected": [1]},
        {"input": 5, "expected": [1, 1, 2, 3, 5]},
        {"input": 8, "expected": [1, 1, 2, 3, 5, 8, 13, 21]},
    ],
    "euclidean_gcd": [
        {"input": (12, 8), "expected": 4},
        {"input": (17, 13), "expected": 1},
        {"input": (100, 25), "expected": 25},
        {"input": (-12, 8), "expected": 4},
    ],
    "is_prime": [
        {"input": 1, "expected": False},
        {"input": 2, "expected": True},
        {"input": 17, "expected": True},
        {"input": 100, "expected": False},
        {"input": 97, "expected": True},
    ],
}

# Algorithm descriptions for prompts
ALGORITHM_SPECS = {
    "factorial": {
        "name": "Factorial",
        "description": "Calculate the factorial of a number (n! = n * (n-1) * ... * 1)",
        "function_name": "factorial",
        "params": "number",
        "examples": "factorial(5) = 120, factorial(0) = 1",
    },
    "fibonacci": {
        "name": "Fibonacci Sequence",
        "description": "Return the first n numbers in the Fibonacci sequence as an array",
        "function_name": "fibonacci",
        "params": "n",
        "examples": "fibonacci(5) = [1, 1, 2, 3, 5]",
    },
    "euclidean_gcd": {
        "name": "Euclidean Algorithm (GCD)",
        "description": "Find the greatest common divisor of two numbers using the Euclidean algorithm",
        "function_name": "euclideanAlgorithm",
        "params": "originalA, originalB",
        "examples": "gcd(12, 8) = 4, gcd(17, 13) = 1",
    },
    "is_prime": {
        "name": "Primality Test (Trial Division)",
        "description": "Check if a number is prime using trial division",
        "function_name": "trialDivision",
        "params": "number",
        "examples": "isPrime(17) = true, isPrime(100) = false",
    },
}


def create_prompts(algo_key: str) -> Dict[str, str]:
    """Generate prompts for each style for a given algorithm."""
    spec = ALGORITHM_SPECS[algo_key]

    return {
        "zero_shot": f"""Write a JavaScript function that implements {spec['name']}.

{spec['description']}.

Function signature: {spec['function_name']}({spec['params']})

Examples: {spec['examples']}

Return only the JavaScript code with export default.""",

        "few_shot": f"""Here's an example of a simple JavaScript algorithm:

Example - Sum of array:
```javascript
export default function sumArray(arr) {{
  let total = 0;
  for (let i = 0; i < arr.length; i += 1) {{
    total += arr[i];
  }}
  return total;
}}
```

Now write a JavaScript function that implements {spec['name']}.

{spec['description']}.

Function signature: {spec['function_name']}({spec['params']})
Examples: {spec['examples']}

Return only the JavaScript code with export default.""",

        "cot": f"""Write a JavaScript function that implements {spec['name']}.

{spec['description']}.

Function signature: {spec['function_name']}({spec['params']})
Examples: {spec['examples']}

Think through this step by step:
1. What are the edge cases to handle?
2. What's the core algorithm logic?
3. What variables do I need?
4. Write the implementation

After your reasoning, provide the final JavaScript code with export default.""",

        "schema": f"""ACT=CodeGeneration
LANG=JavaScript
ALGO={spec['name']}
DESC={spec['description']}
FUNC={spec['function_name']}({spec['params']})
EXAMPLES={spec['examples']}
OUTPUT=Code only with export default
STYLE=Clean, readable, use descriptive variable names""",

        "meta": f"""You are tasked with writing a JavaScript function for {spec['name']}.

First, decide on your approach:
- What algorithm pattern fits best?
- What's the time/space complexity tradeoff?
- What edge cases need handling?

Then implement: {spec['function_name']}({spec['params']})
Description: {spec['description']}
Examples: {spec['examples']}

Provide your approach reasoning, then the final JavaScript code with export default.""",

        "gen_knowledge": f"""First, recall key facts about {spec['name']}:
- Mathematical definition
- Common implementation patterns
- Edge cases to consider
- Optimization opportunities

Now using this knowledge, write a JavaScript function:
Function: {spec['function_name']}({spec['params']})
Description: {spec['description']}
Examples: {spec['examples']}

Return the JavaScript code with export default.""",

        "directional": f"""Write a JavaScript function for {spec['name']}.

HINTS:
- Use iterative approach (for/while loop)
- Handle edge cases first (0, 1, negative numbers)
- Use descriptive variable names
- Keep it simple and readable

Function: {spec['function_name']}({spec['params']})
Description: {spec['description']}
Examples: {spec['examples']}

Return JavaScript code with export default.""",

        "tot": f"""Write a JavaScript function for {spec['name']}.

Explore multiple implementation approaches:

APPROACH A - Iterative:
Consider using a for/while loop

APPROACH B - Recursive:
Consider a recursive solution

APPROACH C - Functional:
Consider using array methods like reduce

Evaluate each approach for:
- Readability
- Performance
- Edge case handling

Choose the best approach and implement:
Function: {spec['function_name']}({spec['params']})
Description: {spec['description']}
Examples: {spec['examples']}

Return the best JavaScript implementation with export default.""",

        "self_consistency": f"""Write a JavaScript function for {spec['name']}.

Generate the solution using three different mental approaches:
1. As a beginner - focus on clarity
2. As an expert - focus on elegance
3. As a code reviewer - focus on correctness

Function: {spec['function_name']}({spec['params']})
Description: {spec['description']}
Examples: {spec['examples']}

After considering all three perspectives, provide the final balanced JavaScript implementation with export default.""",
    }


def extract_code(response: str) -> str:
    """Extract JavaScript code from LLM response."""
    # Try to find code in markdown blocks
    code_blocks = re.findall(r'```(?:javascript|js)?\s*([\s\S]*?)```', response)
    if code_blocks:
        # Return the last code block (usually the final answer)
        return code_blocks[-1].strip()

    # Try to find export default function
    match = re.search(r'(export\s+default\s+function[\s\S]+?}\s*$)', response, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Return cleaned response
    return response.strip()


def normalize_code(code: str) -> str:
    """Normalize code for comparison (remove comments, normalize whitespace)."""
    # Remove single-line comments
    code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
    # Remove multi-line comments
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    # Normalize whitespace
    code = re.sub(r'\s+', ' ', code)
    # Remove leading/trailing whitespace
    code = code.strip()
    return code


def calculate_similarity(generated: str, reference: str) -> Dict[str, float]:
    """Calculate various similarity metrics between generated and reference code."""
    gen_normalized = normalize_code(generated)
    ref_normalized = normalize_code(reference)

    # Sequence matcher similarity
    seq_similarity = difflib.SequenceMatcher(None, gen_normalized, ref_normalized).ratio()

    # Line-by-line similarity
    gen_lines = [l.strip() for l in generated.split('\n') if l.strip()]
    ref_lines = [l.strip() for l in reference.split('\n') if l.strip()]
    line_similarity = difflib.SequenceMatcher(None, gen_lines, ref_lines).ratio()

    # Token-based similarity (split on non-alphanumeric)
    gen_tokens = set(re.findall(r'\w+', generated))
    ref_tokens = set(re.findall(r'\w+', reference))
    if gen_tokens or ref_tokens:
        token_overlap = len(gen_tokens & ref_tokens) / len(gen_tokens | ref_tokens)
    else:
        token_overlap = 0

    # Check for key structural elements
    structural_score = 0
    structural_checks = [
        ('export default', 'has_export'),
        ('function', 'has_function'),
        ('return', 'has_return'),
        ('for', 'has_loop'),
        ('while', 'has_loop'),
    ]

    for pattern, _ in structural_checks:
        if pattern in generated.lower():
            structural_score += 1
    structural_score = structural_score / len(structural_checks)

    return {
        "sequence_similarity": round(seq_similarity, 3),
        "line_similarity": round(line_similarity, 3),
        "token_overlap": round(token_overlap, 3),
        "structural_score": round(structural_score, 3),
        "combined_score": round((seq_similarity + line_similarity + token_overlap) / 3, 3),
    }


def test_code_correctness(code: str, algo_key: str) -> Dict[str, any]:
    """
    Test if generated code would produce correct results.
    Note: This is a heuristic check since we can't execute JS directly.
    """
    test_cases = TEST_CASES.get(algo_key, [])

    # Check if code has expected structure
    checks = {
        "has_function": "function" in code,
        "has_export": "export" in code,
        "has_return": "return" in code,
        "is_syntactically_complete": code.count('{') == code.count('}'),
    }

    # Algorithm-specific checks
    if algo_key == "factorial":
        checks["handles_base_case"] = any(x in code for x in ["<= 1", "=== 0", "=== 1", "== 0", "== 1"])
        checks["has_multiplication"] = "*=" in code or "* " in code

    elif algo_key == "fibonacci":
        checks["handles_base_case"] = any(x in code for x in ["=== 1", "== 1", "<= 1"])
        checks["has_array"] = "[" in code and "]" in code
        checks["tracks_previous"] = any(x in code.lower() for x in ["prev", "previous", "last"])

    elif algo_key == "euclidean_gcd":
        checks["uses_modulo"] = "%" in code
        checks["handles_recursion_or_loop"] = "euclidean" in code.lower() or "while" in code
        checks["handles_negative"] = "Math.abs" in code or "abs" in code.lower()

    elif algo_key == "is_prime":
        checks["handles_edge_cases"] = any(x in code for x in ["<= 1", "< 2", "=== 1"])
        checks["checks_divisibility"] = "%" in code
        checks["uses_sqrt_optimization"] = "sqrt" in code.lower() or "Math.sqrt" in code

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "correctness_score": round(passed / total, 3) if total > 0 else 0,
    }


@dataclass
class CodeBenchmarkResult:
    """Result of a single code generation benchmark."""
    algorithm: str
    style: str
    model: str
    generated_code: str
    similarity_metrics: Dict[str, float]
    correctness_metrics: Dict[str, any]
    input_tokens: int
    output_tokens: int
    total_tokens: int


def run_code_benchmark(
    model_key: str,
    algorithms: Optional[List[str]] = None,
    styles: Optional[List[str]] = None,
) -> List[CodeBenchmarkResult]:
    """Run code generation benchmark."""

    if algorithms is None:
        algorithms = list(ALGORITHM_SPECS.keys())

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

    for algo_key in algorithms:
        prompts = create_prompts(algo_key)
        reference = REFERENCE_CODE[algo_key]

        print(f"\n{'='*60}")
        print(f"Algorithm: {ALGORITHM_SPECS[algo_key]['name']}")
        print(f"{'='*60}")

        for style in styles:
            if style not in prompts:
                continue

            prompt = prompts[style]
            print(f"\n  Testing style: {style}...")

            try:
                response = client.invoke(prompt, model_config)
                generated = extract_code(response["response"])

                similarity = calculate_similarity(generated, reference)
                correctness = test_code_correctness(generated, algo_key)

                result = CodeBenchmarkResult(
                    algorithm=algo_key,
                    style=style,
                    model=model_key,
                    generated_code=generated,
                    similarity_metrics=similarity,
                    correctness_metrics=correctness,
                    input_tokens=response["input_tokens"],
                    output_tokens=response["output_tokens"],
                    total_tokens=response["input_tokens"] + response["output_tokens"],
                )
                results.append(result)

                print(f"    Similarity: {similarity['combined_score']:.1%}")
                print(f"    Correctness: {correctness['correctness_score']:.1%}")
                print(f"    Tokens: {result.total_tokens}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    return results


def format_results_table(results: List[CodeBenchmarkResult]) -> str:
    """Format results as a markdown table."""

    # Group by algorithm
    by_algo = {}
    for r in results:
        if r.algorithm not in by_algo:
            by_algo[r.algorithm] = {}
        by_algo[r.algorithm][r.style] = r

    output = []

    for algo_key, style_results in by_algo.items():
        algo_name = ALGORITHM_SPECS[algo_key]["name"]
        output.append(f"\n### {algo_name}\n")
        output.append("| Style | Similarity | Correctness | Tokens |")
        output.append("|-------|------------|-------------|--------|")

        for style, r in sorted(style_results.items()):
            sim = r.similarity_metrics["combined_score"]
            corr = r.correctness_metrics["correctness_score"]
            tokens = r.total_tokens
            output.append(f"| {style} | {sim:.1%} | {corr:.1%} | {tokens} |")

    return "\n".join(output)


def summarize_results(results: List[CodeBenchmarkResult]) -> Dict:
    """Generate summary statistics."""

    # Aggregate by style
    by_style = {}
    for r in results:
        if r.style not in by_style:
            by_style[r.style] = {
                "similarities": [],
                "correctness": [],
                "tokens": [],
            }
        by_style[r.style]["similarities"].append(r.similarity_metrics["combined_score"])
        by_style[r.style]["correctness"].append(r.correctness_metrics["correctness_score"])
        by_style[r.style]["tokens"].append(r.total_tokens)

    summary = {}
    for style, data in by_style.items():
        summary[style] = {
            "avg_similarity": round(sum(data["similarities"]) / len(data["similarities"]), 3),
            "avg_correctness": round(sum(data["correctness"]) / len(data["correctness"]), 3),
            "avg_tokens": round(sum(data["tokens"]) / len(data["tokens"])),
            "count": len(data["similarities"]),
        }

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Code Generation Benchmark")
    parser.add_argument("--model", default="nova-micro",
                        help="Model to test (nova-micro, mistral-7b, claude-haiku, etc.)")
    parser.add_argument("--algorithms", nargs="+", default=None,
                        help="Algorithms to test (factorial, fibonacci, euclidean_gcd, is_prime)")
    parser.add_argument("--styles", nargs="+", default=None,
                        help="Styles to test")
    parser.add_argument("--output", default=None,
                        help="Output file for results JSON")
    args = parser.parse_args()

    print(f"\nCode Generation Benchmark")
    print(f"Model: {args.model}")
    print(f"Algorithms: {args.algorithms or 'all'}")
    print(f"Styles: {args.styles or 'all'}")

    results = run_code_benchmark(
        model_key=args.model,
        algorithms=args.algorithms,
        styles=args.styles,
    )

    # Print results table
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(format_results_table(results))

    # Print summary
    summary = summarize_results(results)
    print("\n" + "="*60)
    print("SUMMARY BY STYLE")
    print("="*60)
    print(f"{'Style':<20} {'Similarity':>12} {'Correctness':>12} {'Tokens':>10}")
    print("-" * 56)

    for style, data in sorted(summary.items(), key=lambda x: -x[1]["avg_similarity"]):
        print(f"{style:<20} {data['avg_similarity']:>11.1%} {data['avg_correctness']:>11.1%} {data['avg_tokens']:>10}")

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "results": [
                {
                    "algorithm": r.algorithm,
                    "style": r.style,
                    "similarity": r.similarity_metrics,
                    "correctness": r.correctness_metrics,
                    "tokens": r.total_tokens,
                    "generated_code": r.generated_code,
                }
                for r in results
            ],
            "summary": summary,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
