"""
Unified Evaluation Module for Prompting Styles Research

Provides standardized evaluation methods:
- Exact match (with normalization)
- Fuzzy match (substring after normalization)
- F1 score (token-level overlap)
- Contains match (answer anywhere in response)
- Keywords check (required terms present)
- Code execution (run code against test cases)
- Code linting (pylint/syntax checking)

Inspired by OpenAI Evals and DeepEval approaches.
"""

import re
import string
import subprocess
import tempfile
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, Union
from collections import Counter
from enum import Enum
from pathlib import Path

# Optional OpenAI import for embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class EvalType(Enum):
    """Evaluation method types."""
    EXACT = "exact"           # Exact match after normalization
    FUZZY = "fuzzy"           # Substring match after normalization
    CONTAINS = "contains"     # Expected answer appears anywhere in response
    KEYWORDS = "keywords"     # Required keywords present
    NUMERIC = "numeric"       # Numeric comparison with tolerance
    CODE_EXEC = "code_exec"   # Execute code against test cases
    CRITERIA = "criteria"     # Custom criteria functions
    LLM_JUDGE = "llm_judge"   # LLM-as-judge evaluation (Opus 4.5)


@dataclass
class EvalResult:
    """Result of evaluating a single response."""
    correct: bool
    score: float  # 0.0 to 1.0, allows partial credit
    method: str
    details: dict = field(default_factory=dict)
    reason: str = ""


# =============================================================================
# Text Normalization (from OpenAI Evals)
# =============================================================================

def normalize(s: str) -> str:
    """
    Normalize text for comparison.
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Collapse whitespace
    """
    if not s:
        return ""
    s = s.lower()
    # Remove punctuation
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def extract_number(s: str) -> Optional[float]:
    """Extract the first number from a string."""
    # Match numbers including decimals, with optional $ or %
    patterns = [
        r'\$?([\d,]+\.?\d*)',  # $11.20 or 11.20 or 1,000
        r'([\d,]+\.?\d*)%?',   # 25% or 25
    ]
    for pattern in patterns:
        match = re.search(pattern, s.replace(',', ''))
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except ValueError:
                continue
    return None


def extract_all_numbers(s: str) -> list[float]:
    """Extract all numbers from a string."""
    numbers = re.findall(r'[\d,]+\.?\d*', s.replace(',', ''))
    result = []
    for n in numbers:
        try:
            result.append(float(n))
        except ValueError:
            continue
    return result


# =============================================================================
# Matching Functions
# =============================================================================

def exact_match(response: str, expected: Union[str, list[str]]) -> bool:
    """Check if normalized response exactly matches any expected answer."""
    if isinstance(expected, str):
        expected = [expected]

    norm_response = normalize(response)
    return any(normalize(exp) == norm_response for exp in expected)


def fuzzy_match(response: str, expected: Union[str, list[str]]) -> bool:
    """
    Check if normalized response contains or is contained by expected answer.
    More lenient than exact match.
    """
    if isinstance(expected, str):
        expected = [expected]

    norm_response = normalize(response)
    if not norm_response:
        return False

    for exp in expected:
        norm_exp = normalize(exp)
        if not norm_exp:
            continue
        # Either response contains expected, or expected contains response
        if norm_exp in norm_response or norm_response in norm_exp:
            return True
    return False


def contains_match(response: str, expected: Union[str, list[str]]) -> bool:
    """Check if any expected answer appears anywhere in the response."""
    if isinstance(expected, str):
        expected = [expected]

    norm_response = normalize(response)
    return any(normalize(exp) in norm_response for exp in expected)


def numeric_match(response: str, expected: float, tolerance: float = 0.01) -> bool:
    """Check if response contains the expected number within tolerance."""
    numbers = extract_all_numbers(response)
    for num in numbers:
        if abs(num - expected) <= tolerance:
            return True
    return False


def keywords_match(response: str, keywords: list[str], threshold: float = 0.5) -> tuple[bool, float]:
    """
    Check if response contains required keywords.
    Returns (passed, score) where score is fraction of keywords found.
    """
    if not keywords:
        return True, 1.0

    response_lower = response.lower()
    found = sum(1 for kw in keywords if kw.lower() in response_lower)
    score = found / len(keywords)
    return score >= threshold, score


# =============================================================================
# F1 Score (Token-level, from OpenAI Evals)
# =============================================================================

def f1_score(prediction: str, answers: Union[str, list[str]]) -> float:
    """
    Compute F1 score based on token overlap.
    Returns best F1 across all valid answers.
    """
    if isinstance(answers, str):
        answers = [answers]

    def _f1_single(pred: str, truth: str) -> float:
        pred_tokens = normalize(pred).split()
        truth_tokens = normalize(truth).split()

        if not pred_tokens or not truth_tokens:
            return 1.0 if pred_tokens == truth_tokens else 0.0

        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    return max(_f1_single(prediction, answer) for answer in answers)


# =============================================================================
# Code Evaluation
# =============================================================================

def extract_code(response: str, language: str = "python") -> str:
    """Extract code from LLM response, handling markdown blocks."""
    # Try to find code in markdown blocks
    patterns = [
        rf'```{language}\s*([\s\S]*?)```',
        rf'```{language[:2]}\s*([\s\S]*?)```',  # py, js
        r'```\s*([\s\S]*?)```',  # Generic code block
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].strip()  # Return last code block

    # If no code blocks, return the whole response (might be just code)
    return response.strip()


def lint_python_code(code: str) -> dict:
    """
    Run pylint on Python code and return results.
    Returns: {valid: bool, score: float, errors: list, warnings: list}
    """
    result = {
        "valid": False,
        "syntax_valid": False,
        "score": 0.0,
        "errors": [],
        "warnings": [],
        "pylint_score": 0.0,
    }

    # First check syntax
    try:
        compile(code, "<string>", "exec")
        result["syntax_valid"] = True
    except SyntaxError as e:
        result["errors"].append(f"SyntaxError: {e.msg} (line {e.lineno})")
        return result

    # Run pylint
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        proc = subprocess.run(
            ['pylint', temp_path, '--output-format=json', '--disable=C0114,C0115,C0116'],  # Disable docstring warnings
            capture_output=True,
            text=True,
            timeout=30
        )

        if proc.stdout:
            try:
                messages = json.loads(proc.stdout)
                for msg in messages:
                    if msg.get('type') == 'error':
                        result["errors"].append(f"{msg.get('symbol')}: {msg.get('message')} (line {msg.get('line')})")
                    elif msg.get('type') == 'warning':
                        result["warnings"].append(f"{msg.get('symbol')}: {msg.get('message')}")
            except json.JSONDecodeError:
                pass

        # Extract pylint score from stderr
        score_match = re.search(r'rated at ([\d.]+)/10', proc.stderr)
        if score_match:
            result["pylint_score"] = float(score_match.group(1)) / 10.0

        result["valid"] = len(result["errors"]) == 0
        result["score"] = result["pylint_score"] if result["valid"] else 0.0

    except FileNotFoundError:
        # pylint not installed, just use syntax check
        result["valid"] = result["syntax_valid"]
        result["score"] = 1.0 if result["syntax_valid"] else 0.0
        result["warnings"].append("pylint not installed, using syntax check only")
    except subprocess.TimeoutExpired:
        result["errors"].append("Pylint timed out")
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return result


def lint_javascript_code(code: str) -> dict:
    """
    Basic JavaScript syntax validation.
    Returns: {valid: bool, score: float, errors: list}
    """
    result = {
        "valid": False,
        "syntax_valid": False,
        "score": 0.0,
        "errors": [],
        "warnings": [],
    }

    # Basic structural checks
    checks = {
        "has_function": "function" in code or "=>" in code,
        "balanced_braces": code.count('{') == code.count('}'),
        "balanced_parens": code.count('(') == code.count(')'),
        "balanced_brackets": code.count('[') == code.count(']'),
        "has_return": "return" in code,
    }

    result["syntax_valid"] = all([
        checks["balanced_braces"],
        checks["balanced_parens"],
        checks["balanced_brackets"],
    ])

    if not checks["balanced_braces"]:
        result["errors"].append("Unbalanced curly braces")
    if not checks["balanced_parens"]:
        result["errors"].append("Unbalanced parentheses")
    if not checks["balanced_brackets"]:
        result["errors"].append("Unbalanced brackets")

    # Try to run with Node.js syntax check if available
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        proc = subprocess.run(
            ['node', '--check', temp_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        if proc.returncode == 0:
            result["syntax_valid"] = True
            result["valid"] = True
            result["score"] = 1.0
        else:
            result["errors"].append(proc.stderr.strip())
    except FileNotFoundError:
        # Node not installed, use basic checks
        result["valid"] = result["syntax_valid"]
        result["score"] = 1.0 if result["syntax_valid"] else 0.0
        result["warnings"].append("Node.js not installed, using basic checks only")
    except subprocess.TimeoutExpired:
        result["errors"].append("Syntax check timed out")
    finally:
        Path(temp_path).unlink(missing_ok=True)

    return result


def execute_python_code(code: str, test_cases: list[dict], timeout: int = 10) -> dict:
    """
    Execute Python code against test cases.

    test_cases: [{"input": x, "expected": y, "function": "func_name"}, ...]

    Returns: {passed: int, total: int, score: float, results: list}
    """
    result = {
        "passed": 0,
        "total": len(test_cases),
        "score": 0.0,
        "results": [],
        "execution_error": None,
    }

    if not test_cases:
        return {"passed": 0, "total": 0, "score": 1.0, "results": [], "execution_error": None}

    for i, test in enumerate(test_cases):
        test_result = {"input": test["input"], "expected": test["expected"], "passed": False, "actual": None, "error": None}

        # Build execution code
        func_name = test.get("function", "solution")
        test_input = test["input"]

        # Handle different input types
        if isinstance(test_input, tuple):
            args_str = ", ".join(repr(a) for a in test_input)
        else:
            args_str = repr(test_input)

        exec_code = f"""
{code}

result = {func_name}({args_str})
print(repr(result))
"""

        try:
            proc = subprocess.run(
                ['python', '-c', exec_code],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if proc.returncode == 0:
                actual = proc.stdout.strip()
                try:
                    actual_value = eval(actual)
                    test_result["actual"] = actual_value

                    if actual_value == test["expected"]:
                        test_result["passed"] = True
                        result["passed"] += 1
                    elif isinstance(test["expected"], float) and isinstance(actual_value, (int, float)):
                        # Allow float tolerance
                        if abs(actual_value - test["expected"]) < 0.001:
                            test_result["passed"] = True
                            result["passed"] += 1
                except:
                    test_result["actual"] = actual
            else:
                test_result["error"] = proc.stderr.strip()

        except subprocess.TimeoutExpired:
            test_result["error"] = "Execution timed out"
        except Exception as e:
            test_result["error"] = str(e)

        result["results"].append(test_result)

    result["score"] = result["passed"] / result["total"] if result["total"] > 0 else 0.0
    return result


# =============================================================================
# Main Evaluation Function
# =============================================================================

@dataclass
class TestCase:
    """Test case definition with evaluation config."""
    id: str
    name: str
    category: str  # "math", "logic", "writing", "code", "creative"
    task: str  # The actual prompt/task description

    # Evaluation type
    eval_type: EvalType = EvalType.CONTAINS

    # Ground truth - what we compare against
    expected_answers: list[str] = field(default_factory=list)  # Valid answers
    expected_number: Optional[float] = None  # For numeric comparisons
    expected_keywords: list[str] = field(default_factory=list)  # Required keywords

    # For code evaluation
    code_language: str = "python"
    test_cases: list[dict] = field(default_factory=list)  # For code execution
    function_name: str = "solution"

    # Custom criteria functions
    criteria: list[Callable[[str], bool]] = field(default_factory=list)

    # Thresholds
    keyword_threshold: float = 0.5
    numeric_tolerance: float = 0.01


def evaluate(response: str, test_case: TestCase) -> EvalResult:
    """
    Universal evaluator that dispatches to appropriate method based on eval_type.
    """
    eval_type = test_case.eval_type

    if eval_type == EvalType.EXACT:
        matched = exact_match(response, test_case.expected_answers)
        return EvalResult(
            correct=matched,
            score=1.0 if matched else 0.0,
            method="exact",
            details={"normalized_response": normalize(response)[:100]},
            reason="Exact match after normalization" if matched else "No exact match found"
        )

    elif eval_type == EvalType.FUZZY:
        matched = fuzzy_match(response, test_case.expected_answers)
        f1 = f1_score(response, test_case.expected_answers)
        return EvalResult(
            correct=matched,
            score=max(f1, 1.0 if matched else 0.0),
            method="fuzzy",
            details={"fuzzy_match": matched, "f1_score": f1},
            reason=f"Fuzzy match: {matched}, F1: {f1:.2f}"
        )

    elif eval_type == EvalType.CONTAINS:
        matched = contains_match(response, test_case.expected_answers)
        return EvalResult(
            correct=matched,
            score=1.0 if matched else 0.0,
            method="contains",
            details={"found_in_response": matched},
            reason="Expected answer found in response" if matched else "Expected answer not found"
        )

    elif eval_type == EvalType.NUMERIC:
        if test_case.expected_number is not None:
            matched = numeric_match(response, test_case.expected_number, test_case.numeric_tolerance)
            extracted = extract_all_numbers(response)
            return EvalResult(
                correct=matched,
                score=1.0 if matched else 0.0,
                method="numeric",
                details={"expected": test_case.expected_number, "extracted_numbers": extracted[:5]},
                reason=f"Expected {test_case.expected_number}, found: {extracted[:3]}"
            )
        else:
            # Fall back to contains match with expected_answers
            matched = contains_match(response, test_case.expected_answers)
            return EvalResult(correct=matched, score=1.0 if matched else 0.0, method="numeric_fallback")

    elif eval_type == EvalType.KEYWORDS:
        passed, score = keywords_match(response, test_case.expected_keywords, test_case.keyword_threshold)
        found = [kw for kw in test_case.expected_keywords if kw.lower() in response.lower()]
        return EvalResult(
            correct=passed,
            score=score,
            method="keywords",
            details={"keywords_found": found, "keywords_expected": test_case.expected_keywords},
            reason=f"Found {len(found)}/{len(test_case.expected_keywords)} keywords"
        )

    elif eval_type == EvalType.CODE_EXEC:
        code = extract_code(response, test_case.code_language)

        # First lint the code
        if test_case.code_language == "python":
            lint_result = lint_python_code(code)
            exec_result = execute_python_code(code, test_case.test_cases) if lint_result["syntax_valid"] else None
        else:  # javascript
            lint_result = lint_javascript_code(code)
            exec_result = None  # JS execution not implemented yet

        if exec_result:
            score = (lint_result["score"] * 0.3) + (exec_result["score"] * 0.7)  # Weight execution higher
            passed = exec_result["passed"] == exec_result["total"] and lint_result["syntax_valid"]
        else:
            score = lint_result["score"]
            passed = lint_result["valid"]

        return EvalResult(
            correct=passed,
            score=score,
            method="code_exec",
            details={
                "lint": lint_result,
                "execution": exec_result,
                "code_extracted": code[:200] + "..." if len(code) > 200 else code
            },
            reason=f"Lint: {lint_result['valid']}, Exec: {exec_result['passed'] if exec_result else 'N/A'}/{exec_result['total'] if exec_result else 0}"
        )

    elif eval_type == EvalType.CRITERIA:
        if not test_case.criteria:
            return EvalResult(correct=True, score=1.0, method="criteria", reason="No criteria defined")

        results = {}
        for i, criterion in enumerate(test_case.criteria):
            try:
                results[f"criterion_{i}"] = criterion(response)
            except Exception as e:
                results[f"criterion_{i}"] = False

        passed_count = sum(1 for v in results.values() if v)
        score = passed_count / len(results)

        return EvalResult(
            correct=score >= 0.5,
            score=score,
            method="criteria",
            details={"criteria_results": results},
            reason=f"Passed {passed_count}/{len(results)} criteria"
        )

    else:
        # Default to contains match
        matched = contains_match(response, test_case.expected_answers)
        return EvalResult(correct=matched, score=1.0 if matched else 0.0, method="default")


# =============================================================================
# LLM-as-Judge Evaluation (Opus 4.5)
# =============================================================================

# Lazy-load the client to avoid import issues
_llm_judge_client = None

def get_llm_judge_client():
    """Get or create the LLM judge client (lazy loading)."""
    global _llm_judge_client
    if _llm_judge_client is None:
        try:
            from multi_provider_client import MultiProviderClient, MODELS
            _llm_judge_client = (MultiProviderClient(), MODELS["claude-opus"])
        except ImportError:
            raise RuntimeError("multi_provider_client not available for LLM judge")
    return _llm_judge_client


@dataclass
class LLMJudgeResult:
    """Result from LLM-as-judge evaluation."""
    score: float           # 0.0 to 1.0
    correct: bool          # Binary pass/fail
    reasoning: str         # Why the judge gave this score
    correctness: float     # Factual correctness (0-1)
    completeness: float    # How complete is the answer (0-1)
    clarity: float         # How clear/well-structured (0-1)
    relevance: float       # How relevant to the question (0-1)
    raw_response: str      # Raw judge response


LLM_JUDGE_PROMPT = """You are an expert evaluator assessing LLM responses. Your job is to objectively judge how well a response answers a given task.

## Task Given to the LLM
{task}

## Expected Answer (Ground Truth)
{expected}

## LLM's Response
{response}

## Evaluation Criteria
Score each dimension from 0.0 to 1.0:

1. **Correctness** (0-1): Is the answer factually correct? Does it match the expected answer?
   - 1.0 = Perfectly correct
   - 0.5 = Partially correct or minor errors
   - 0.0 = Incorrect

2. **Completeness** (0-1): Does the response fully address all parts of the task?
   - 1.0 = Fully complete
   - 0.5 = Missing some elements
   - 0.0 = Severely incomplete

3. **Clarity** (0-1): Is the response well-organized and easy to understand?
   - 1.0 = Crystal clear
   - 0.5 = Somewhat unclear
   - 0.0 = Confusing or incoherent

4. **Relevance** (0-1): Does the response stay on topic and address the actual question?
   - 1.0 = Perfectly relevant
   - 0.5 = Some irrelevant content
   - 0.0 = Off-topic

## Output Format
Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{{
    "correctness": <float 0-1>,
    "completeness": <float 0-1>,
    "clarity": <float 0-1>,
    "relevance": <float 0-1>,
    "overall_score": <float 0-1>,
    "pass": <boolean>,
    "reasoning": "<brief explanation of your scores>"
}}"""


def llm_judge_evaluate(
    response: str,
    task: str,
    expected: str,
    threshold: float = 0.5,
) -> LLMJudgeResult:
    """
    Use Opus 4.5 as a judge to evaluate a response.

    Args:
        response: The LLM's response to evaluate
        task: The original task/question
        expected: The expected/ground truth answer
        threshold: Score threshold for pass/fail (default 0.5)

    Returns:
        LLMJudgeResult with detailed scoring
    """
    client, model_config = get_llm_judge_client()

    prompt = LLM_JUDGE_PROMPT.format(
        task=task,
        expected=expected,
        response=response,  # Full response - Opus 4.5 has 200K context
    )

    try:
        result = client.invoke(prompt, model_config)
        raw_response = result["response"]

        # Parse JSON from response
        # Try to extract JSON if wrapped in markdown
        json_str = raw_response
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0]
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0]

        parsed = json.loads(json_str.strip())

        return LLMJudgeResult(
            score=parsed.get("overall_score", 0.0),
            correct=parsed.get("pass", False),
            reasoning=parsed.get("reasoning", ""),
            correctness=parsed.get("correctness", 0.0),
            completeness=parsed.get("completeness", 0.0),
            clarity=parsed.get("clarity", 0.0),
            relevance=parsed.get("relevance", 0.0),
            raw_response=raw_response,
        )

    except json.JSONDecodeError as e:
        # If JSON parsing fails, try to extract scores manually
        return LLMJudgeResult(
            score=0.0,
            correct=False,
            reasoning=f"Failed to parse judge response: {e}",
            correctness=0.0,
            completeness=0.0,
            clarity=0.0,
            relevance=0.0,
            raw_response=raw_response if 'raw_response' in dir() else str(e),
        )
    except Exception as e:
        return LLMJudgeResult(
            score=0.0,
            correct=False,
            reasoning=f"Judge evaluation failed: {e}",
            correctness=0.0,
            completeness=0.0,
            clarity=0.0,
            relevance=0.0,
            raw_response=str(e),
        )


def evaluate_with_llm_judge(
    response: str,
    test_case: 'TestCase',
    deterministic_result: Optional[EvalResult] = None,
) -> dict:
    """
    Combine deterministic evaluation with LLM judge for comprehensive scoring.

    Returns a dict with both deterministic and LLM judge results.
    """
    # Get deterministic result if not provided
    if deterministic_result is None:
        deterministic_result = evaluate(response, test_case)

    # Build expected answer string for judge
    expected_parts = []
    if test_case.expected_answers:
        expected_parts.append(f"Valid answers: {', '.join(test_case.expected_answers)}")
    if test_case.expected_number is not None:
        expected_parts.append(f"Expected number: {test_case.expected_number}")
    if test_case.expected_keywords:
        expected_parts.append(f"Should mention: {', '.join(test_case.expected_keywords)}")

    expected = "\n".join(expected_parts) if expected_parts else "No specific expected answer defined"

    # Get LLM judge evaluation
    judge_result = llm_judge_evaluate(
        response=response,
        task=test_case.task,
        expected=expected,
    )

    # Combine scores: 60% deterministic, 40% LLM judge
    combined_score = (deterministic_result.score * 0.6) + (judge_result.score * 0.4)

    return {
        "deterministic": {
            "correct": deterministic_result.correct,
            "score": deterministic_result.score,
            "method": deterministic_result.method,
            "reason": deterministic_result.reason,
        },
        "llm_judge": {
            "correct": judge_result.correct,
            "score": judge_result.score,
            "correctness": judge_result.correctness,
            "completeness": judge_result.completeness,
            "clarity": judge_result.clarity,
            "relevance": judge_result.relevance,
            "reasoning": judge_result.reasoning,
        },
        "combined": {
            "score": combined_score,
            "correct": combined_score >= 0.5,
        }
    }


# =============================================================================
# Verbalized Sampling (VS) Parsing
# Based on arXiv:2510.01171 "Verbalized Sampling: How to Mitigate Mode Collapse"
# =============================================================================

@dataclass
class VSResponse:
    """A single response from Verbalized Sampling output."""
    text: str
    probability: float
    index: int


@dataclass
class VSResult:
    """Parsed result from Verbalized Sampling."""
    responses: list[VSResponse]
    highest_prob_response: str
    highest_prob: float
    total_responses: int
    parse_success: bool
    raw_output: str


def parse_verbalized_sampling(response: str) -> VSResult:
    """
    Parse a Verbalized Sampling response that contains multiple responses with probabilities.

    Expected format:
    Response 1 (Prob: 0.25): [text]
    Response 2 (Prob: 0.20): [text]
    ...

    Returns VSResult with parsed responses and the highest-probability answer.
    """
    responses = []

    # Pattern to match "Response N (Prob: X.XX): text" format
    # Also handles variations like "Response 1 (Probability: 0.25):"
    patterns = [
        r'Response\s*(\d+)\s*\(Prob(?:ability)?:\s*([\d.]+)\):\s*(.+?)(?=Response\s*\d+\s*\(|$)',
        r'(\d+)\.\s*\(Prob(?:ability)?:\s*([\d.]+)\):\s*(.+?)(?=\d+\.\s*\(|$)',
        r'Option\s*(\d+)\s*\(Prob(?:ability)?:\s*([\d.]+)\):\s*(.+?)(?=Option\s*\d+\s*\(|$)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            for match in matches:
                idx = int(match[0])
                prob = float(match[1])
                text = match[2].strip()
                responses.append(VSResponse(text=text, probability=prob, index=idx))
            break

    # If no structured format found, try to extract any probabilities mentioned
    if not responses:
        # Fallback: look for any "X.XX" probability patterns with surrounding text
        prob_pattern = r'(?:^|\n)(.+?)(?:\(|\[)?(?:prob(?:ability)?[:\s]*)?(\d+\.?\d*)\)?(?:\]|\))?'
        matches = re.findall(prob_pattern, response, re.IGNORECASE)
        for i, (text, prob_str) in enumerate(matches[:5]):  # Max 5 responses
            try:
                prob = float(prob_str)
                if 0 <= prob <= 1:
                    responses.append(VSResponse(text=text.strip(), probability=prob, index=i+1))
            except ValueError:
                continue

    # Sort by probability (highest first)
    responses.sort(key=lambda r: r.probability, reverse=True)

    if responses:
        return VSResult(
            responses=responses,
            highest_prob_response=responses[0].text,
            highest_prob=responses[0].probability,
            total_responses=len(responses),
            parse_success=True,
            raw_output=response
        )
    else:
        # Parse failed - return whole response as single item
        return VSResult(
            responses=[VSResponse(text=response, probability=1.0, index=1)],
            highest_prob_response=response,
            highest_prob=1.0,
            total_responses=1,
            parse_success=False,
            raw_output=response
        )


def evaluate_vs_response(response: str, test_case: 'TestCase') -> tuple[EvalResult, VSResult]:
    """
    Evaluate a Verbalized Sampling response.

    Returns tuple of (EvalResult for highest-prob answer, VSResult with all parsed responses).
    """
    vs_result = parse_verbalized_sampling(response)

    # Evaluate the highest probability response
    eval_result = evaluate(vs_result.highest_prob_response, test_case)

    # Add VS-specific details
    eval_result.details["vs_parsed"] = vs_result.parse_success
    eval_result.details["vs_total_responses"] = vs_result.total_responses
    eval_result.details["vs_highest_prob"] = vs_result.highest_prob

    # Also check if ANY response is correct (for diversity bonus)
    any_correct = False
    correct_count = 0
    for vs_resp in vs_result.responses:
        resp_eval = evaluate(vs_resp.text, test_case)
        if resp_eval.correct:
            any_correct = True
            correct_count += 1

    eval_result.details["vs_any_correct"] = any_correct
    eval_result.details["vs_correct_count"] = correct_count

    return eval_result, vs_result


# =============================================================================
# Diversity Metrics
# Based on arXiv:2510.01171 methodology for measuring output diversity
# =============================================================================

def calculate_lexical_diversity_rouge(texts: list[str]) -> float:
    """
    Calculate lexical diversity using ROUGE-L overlap.
    Lower ROUGE-L = higher diversity (less overlap between texts).

    Returns diversity score 0-1 where 1 = maximum diversity.
    """
    if len(texts) < 2:
        return 1.0

    def lcs_length(s1: list[str], s2: list[str]) -> int:
        """Longest Common Subsequence length."""
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def rouge_l(text1: str, text2: str) -> float:
        """Calculate ROUGE-L F1 score between two texts."""
        tokens1 = normalize(text1).split()
        tokens2 = normalize(text2).split()

        if not tokens1 or not tokens2:
            return 0.0

        lcs = lcs_length(tokens1, tokens2)

        precision = lcs / len(tokens1) if tokens1 else 0
        recall = lcs / len(tokens2) if tokens2 else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    # Calculate mean pairwise ROUGE-L
    total_rouge = 0.0
    pairs = 0

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total_rouge += rouge_l(texts[i], texts[j])
            pairs += 1

    if pairs == 0:
        return 1.0

    mean_rouge = total_rouge / pairs

    # Convert to diversity: 1 - ROUGE-L (lower overlap = higher diversity)
    return 1.0 - mean_rouge


def calculate_semantic_diversity_simple(texts: list[str]) -> float:
    """
    Calculate semantic diversity using token-level Jaccard similarity.
    This is a simpler alternative when embeddings aren't available.

    Returns diversity score 0-1 where 1 = maximum diversity.
    """
    if len(texts) < 2:
        return 1.0

    def jaccard_similarity(text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between tokenized texts."""
        tokens1 = set(normalize(text1).split())
        tokens2 = set(normalize(text2).split())

        if not tokens1 and not tokens2:
            return 1.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0

    # Calculate mean pairwise Jaccard similarity
    total_sim = 0.0
    pairs = 0

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            total_sim += jaccard_similarity(texts[i], texts[j])
            pairs += 1

    if pairs == 0:
        return 1.0

    mean_sim = total_sim / pairs

    # Convert to diversity: 1 - similarity
    return 1.0 - mean_sim


def calculate_ngram_diversity(texts: list[str], n: int = 2) -> float:
    """
    Calculate n-gram diversity (distinct-n metric).

    Returns the ratio of unique n-grams to total n-grams across all texts.
    Higher ratio = more diverse vocabulary usage.
    """
    if not texts:
        return 0.0

    all_ngrams = []

    for text in texts:
        tokens = normalize(text).split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            all_ngrams.append(ngram)

    if not all_ngrams:
        return 1.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)

    return unique_ngrams / total_ngrams


def calculate_embedding_diversity(texts: list[str], api_key: str = None) -> Optional[float]:
    """
    Calculate semantic diversity using OpenAI embeddings (text-embedding-3-small).
    This matches the methodology from the VS paper (arXiv:2510.01171).

    Returns diversity score 0-1 where 1 = maximum diversity, or None if unavailable.
    """
    if not OPENAI_AVAILABLE:
        return None

    if len(texts) < 2:
        return 1.0

    # Get API key from parameter, environment, or .env file
    if not api_key:
        api_key = os.environ.get("OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")

    # Try loading from .env file if not in environment
    if not api_key:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.startswith("OPENAI_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break

    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)

        # Get embeddings for all texts
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        embeddings = [item.embedding for item in response.data]

        # Calculate mean pairwise cosine similarity
        def cosine_similarity(v1: list[float], v2: list[float]) -> float:
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a * a for a in v1) ** 0.5
            norm2 = sum(b * b for b in v2) ** 0.5
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot_product / (norm1 * norm2)

        total_sim = 0.0
        pairs = 0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                # Clip negative similarities to 0 (as per paper methodology)
                total_sim += max(0, sim)
                pairs += 1

        if pairs == 0:
            return 1.0

        mean_sim = total_sim / pairs

        # Convert to diversity: 1 - similarity
        return 1.0 - mean_sim

    except Exception as e:
        print(f"Warning: Embedding diversity calculation failed: {e}")
        return None


@dataclass
class DiversityResult:
    """Result of diversity evaluation."""
    lexical_diversity: float      # Based on ROUGE-L (0-1, higher = more diverse)
    semantic_diversity: float     # Based on Jaccard similarity (0-1)
    embedding_diversity: Optional[float]  # Based on OpenAI embeddings (0-1) - paper methodology
    bigram_diversity: float       # Distinct-2 metric (0-1)
    trigram_diversity: float      # Distinct-3 metric (0-1)
    combined_diversity: float     # Weighted average
    num_responses: int
    details: dict


def evaluate_diversity(texts: list[str], use_embeddings: bool = True) -> DiversityResult:
    """
    Comprehensive diversity evaluation for a set of texts.

    Args:
        texts: List of text responses to evaluate for diversity
        use_embeddings: Whether to use OpenAI embeddings (adds API cost but more accurate)

    Returns:
        DiversityResult with multiple diversity metrics
    """
    if not texts:
        return DiversityResult(
            lexical_diversity=0.0,
            semantic_diversity=0.0,
            embedding_diversity=None,
            bigram_diversity=0.0,
            trigram_diversity=0.0,
            combined_diversity=0.0,
            num_responses=0,
            details={}
        )

    lexical = calculate_lexical_diversity_rouge(texts)
    semantic = calculate_semantic_diversity_simple(texts)
    bigram = calculate_ngram_diversity(texts, n=2)
    trigram = calculate_ngram_diversity(texts, n=3)

    # Try to get embedding-based diversity (paper methodology)
    embedding = None
    if use_embeddings:
        embedding = calculate_embedding_diversity(texts)

    # Combined score: prefer embedding if available, otherwise use lexical+semantic
    if embedding is not None:
        # Paper methodology: embedding-based semantic diversity is primary
        combined = (embedding * 0.50 + lexical * 0.20 + bigram * 0.15 + trigram * 0.15)
    else:
        # Fallback: weight lexical and Jaccard semantic
        combined = (lexical * 0.35 + semantic * 0.35 + bigram * 0.15 + trigram * 0.15)

    return DiversityResult(
        lexical_diversity=round(lexical, 4),
        semantic_diversity=round(semantic, 4),
        embedding_diversity=round(embedding, 4) if embedding is not None else None,
        bigram_diversity=round(bigram, 4),
        trigram_diversity=round(trigram, 4),
        combined_diversity=round(combined, 4),
        num_responses=len(texts),
        details={
            "texts_evaluated": len(texts),
            "avg_text_length": sum(len(t.split()) for t in texts) / len(texts) if texts else 0,
            "embedding_used": embedding is not None
        }
    )


def evaluate_vs_diversity(vs_result: VSResult) -> DiversityResult:
    """
    Evaluate diversity of responses from Verbalized Sampling.

    Args:
        vs_result: Parsed VSResult from parse_verbalized_sampling()

    Returns:
        DiversityResult measuring how diverse the VS outputs are
    """
    texts = [r.text for r in vs_result.responses]
    return evaluate_diversity(texts)


# =============================================================================
# Batch Evaluation Helpers
# =============================================================================

def evaluate_batch(responses: list[str], test_case: TestCase) -> list[EvalResult]:
    """Evaluate multiple responses against the same test case."""
    return [evaluate(response, test_case) for response in responses]


def calculate_accuracy(results: list[EvalResult]) -> float:
    """Calculate accuracy from a list of results."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)


def calculate_avg_score(results: list[EvalResult]) -> float:
    """Calculate average score from a list of results."""
    if not results:
        return 0.0
    return sum(r.score for r in results) / len(results)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing evaluation module...\n")

    # Test normalize
    print("1. Normalization:")
    print(f"   'The Answer is $11.20!' -> '{normalize('The Answer is $11.20!')}'")

    # Test fuzzy match
    print("\n2. Fuzzy Match:")
    print(f"   fuzzy_match('The answer is 11.20', '11.20') = {fuzzy_match('The answer is 11.20', '11.20')}")
    print(f"   fuzzy_match('$11.20', ['11.20', '11.2']) = {fuzzy_match('$11.20', ['11.20', '11.2'])}")

    # Test F1 score
    print("\n3. F1 Score:")
    print(f"   f1_score('the cat sat', 'the cat sat on mat') = {f1_score('the cat sat', 'the cat sat on mat'):.3f}")

    # Test numeric match
    print("\n4. Numeric Match:")
    print(f"   numeric_match('The answer is $11.20', 11.20) = {numeric_match('The answer is $11.20', 11.20)}")
    print(f"   numeric_match('25% increase', 25.0) = {numeric_match('25% increase', 25.0)}")

    # Test keyword match
    print("\n5. Keyword Match:")
    passed, score = keywords_match("Cloud computing offers scalability and cost savings", ["scalability", "cost", "security"])
    print(f"   keywords_match(..., ['scalability', 'cost', 'security']) = ({passed}, {score:.2f})")

    # Test Python linting
    print("\n6. Python Linting:")
    good_code = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
    bad_code = "def factorial(n)\n    return n"
    print(f"   Good code syntax valid: {lint_python_code(good_code)['syntax_valid']}")
    print(f"   Bad code syntax valid: {lint_python_code(bad_code)['syntax_valid']}")

    # Test code execution
    print("\n7. Code Execution:")
    code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
"""
    test_cases = [
        {"input": 5, "expected": 120, "function": "factorial"},
        {"input": 0, "expected": 1, "function": "factorial"},
    ]
    result = execute_python_code(code, test_cases)
    print(f"   Factorial tests: {result['passed']}/{result['total']} passed")

    # Test full evaluation
    print("\n8. Full Evaluation:")
    tc = TestCase(
        id="test-1",
        name="Discount Calculation",
        category="math",
        task="Calculate 7 apples at $2 each with 20% discount",
        eval_type=EvalType.NUMERIC,
        expected_number=11.20,
        expected_answers=["11.20", "$11.20", "11.2"]
    )
    response = "Let me calculate: 7 x $2 = $14. With 20% off: $14 - $2.80 = $11.20"
    result = evaluate(response, tc)
    print(f"   Response: '{response[:50]}...'")
    print(f"   Result: correct={result.correct}, score={result.score:.2f}, reason='{result.reason}'")

    # Test Verbalized Sampling parsing
    print("\n9. Verbalized Sampling Parsing:")
    vs_response = """Response 1 (Prob: 0.35): Calculate 7 × $2 = $14, then 20% off = $14 - $2.80 = $11.20
Response 2 (Prob: 0.25): 7 apples at $2 = $14. Discounted price = $14 × 0.80 = $11.20
Response 3 (Prob: 0.20): Unit price after discount = $2 × 0.80 = $1.60. Total = 7 × $1.60 = $11.20
Response 4 (Prob: 0.12): Total = $14. Discount = $14 × 0.20 = $2.80. Final = $11.20
Response 5 (Prob: 0.08): Using 80% of full price: $14 × 80/100 = $11.20"""
    vs_result = parse_verbalized_sampling(vs_response)
    print(f"   Parsed {vs_result.total_responses} responses, parse_success={vs_result.parse_success}")
    print(f"   Highest prob ({vs_result.highest_prob:.2f}): '{vs_result.highest_prob_response[:50]}...'")

    # Test VS evaluation
    eval_result, _ = evaluate_vs_response(vs_response, tc)
    print(f"   VS Eval: correct={eval_result.correct}, any_correct={eval_result.details.get('vs_any_correct')}")

    # Test diversity metrics
    print("\n10. Diversity Metrics:")
    diverse_texts = [
        "The bear walked through the forest, searching for honey.",
        "A spacecraft landed on Mars, its crew stepping into red dust.",
        "The chef prepared a delicate soufflé in the quiet kitchen.",
        "Quantum computers revolutionized cryptography overnight.",
        "The ancient library held secrets from civilizations long forgotten."
    ]
    div_result = evaluate_diversity(diverse_texts)
    print(f"   Lexical diversity: {div_result.lexical_diversity:.3f}")
    print(f"   Semantic diversity: {div_result.semantic_diversity:.3f}")
    print(f"   Bigram diversity: {div_result.bigram_diversity:.3f}")
    print(f"   Combined diversity: {div_result.combined_diversity:.3f}")

    similar_texts = [
        "The bear walked through the forest looking for food.",
        "The bear walked through the woods searching for food.",
        "The bear walked through the forest seeking food.",
        "A bear was walking through the forest looking for food.",
        "The bear wandered through the forest looking for food."
    ]
    div_result_sim = evaluate_diversity(similar_texts)
    print(f"\n   Similar texts:")
    print(f"   Combined diversity: {div_result_sim.combined_diversity:.3f} (lower = more similar)")

    print("\n[OK] All tests completed!")
