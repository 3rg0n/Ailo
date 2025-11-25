"""
Tools for Agentic Prompting Research

Provides real executable tools that LLMs can call:
- Calculator: Mathematical operations
- Search: Simulated knowledge retrieval
- Python executor: Run generated code
"""

import re
import math
import json
from typing import Any, Callable
from dataclasses import dataclass


@dataclass
class ToolResult:
    """Result from a tool execution."""
    tool_name: str
    input_args: str
    output: str
    success: bool
    error: str = None


class Calculator:
    """
    Calculator tool that evaluates mathematical expressions safely.

    Supports: +, -, *, /, **, sqrt, sin, cos, tan, log, abs, round
    """

    SAFE_FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'abs': abs,
        'round': round,
        'pow': pow,
        'min': min,
        'max': max,
    }

    SAFE_NAMES = {
        'pi': math.pi,
        'e': math.e,
    }

    @classmethod
    def calculate(cls, expression: str) -> ToolResult:
        """
        Safely evaluate a mathematical expression.

        Examples:
            calculate("2 + 2") -> 4
            calculate("sqrt(16)") -> 4.0
            calculate("7 * 2 * 0.8") -> 11.2
        """
        try:
            # Clean the expression
            expr = expression.strip()

            # Remove any dangerous characters
            if any(c in expr for c in ['import', 'exec', 'eval', '__', 'open', 'file']):
                return ToolResult(
                    tool_name="calculator",
                    input_args=expression,
                    output="",
                    success=False,
                    error="Expression contains forbidden characters"
                )

            # Build safe evaluation context
            safe_dict = {
                '__builtins__': {},
                **cls.SAFE_FUNCTIONS,
                **cls.SAFE_NAMES,
            }

            # Evaluate
            result = eval(expr, safe_dict)

            # Format result
            if isinstance(result, float):
                # Round to reasonable precision
                if result == int(result):
                    result = int(result)
                else:
                    result = round(result, 4)

            return ToolResult(
                tool_name="calculator",
                input_args=expression,
                output=str(result),
                success=True
            )

        except Exception as e:
            return ToolResult(
                tool_name="calculator",
                input_args=expression,
                output="",
                success=False,
                error=str(e)
            )


class SearchTool:
    """
    Simulated search/knowledge retrieval tool.

    Contains pre-loaded facts for benchmark scenarios.
    In production, this would connect to a real search API or vector DB.
    """

    KNOWLEDGE_BASE = {
        # Math/Science facts
        "speed of light": "The speed of light in vacuum is approximately 299,792,458 meters per second (about 3 x 10^8 m/s).",
        "pi": "Pi (π) is approximately 3.14159265359. It's the ratio of a circle's circumference to its diameter.",
        "pythagorean theorem": "The Pythagorean theorem states that in a right triangle, a² + b² = c², where c is the hypotenuse.",

        # Business facts
        "cloud computing benefits": "Key benefits of cloud computing: 1) Cost savings through pay-as-you-go, 2) Scalability on demand, 3) Reduced IT maintenance, 4) Global accessibility, 5) Disaster recovery.",
        "remote work statistics": "Studies show remote workers are 13% more productive on average. 77% report higher productivity. Main challenges: isolation (67%), communication (41%), work-life balance (35%).",

        # Tech facts
        "react vs vue": "React: Library by Meta, JSX syntax, large ecosystem, 200k+ GitHub stars. Vue: Framework by Evan You, template syntax, gentler learning curve, 200k+ GitHub stars. Both use virtual DOM.",
        "python list comprehension": "List comprehension syntax: [expression for item in iterable if condition]. Example: [x**2 for x in range(10) if x % 2 == 0] produces [0, 4, 16, 36, 64].",

        # General knowledge
        "machine learning basics": "Machine learning is a subset of AI where systems learn from data. Types: Supervised (labeled data), Unsupervised (find patterns), Reinforcement (learn from rewards).",
        "discount calculation": "To calculate discount: 1) Find discount amount = Original Price × Discount Rate, 2) Final Price = Original Price - Discount Amount. Example: $100 with 20% off = $100 - $20 = $80.",
    }

    @classmethod
    def search(cls, query: str) -> ToolResult:
        """
        Search the knowledge base for relevant information.

        Returns the most relevant fact based on keyword matching.
        """
        query_lower = query.lower()

        # Find best matching entry
        best_match = None
        best_score = 0

        for key, value in cls.KNOWLEDGE_BASE.items():
            # Simple keyword matching
            key_words = set(key.lower().split())
            query_words = set(query_lower.split())

            # Score based on word overlap
            overlap = len(key_words & query_words)
            if overlap > best_score:
                best_score = overlap
                best_match = (key, value)

            # Also check if key is substring of query
            if key.lower() in query_lower:
                best_match = (key, value)
                break

        if best_match:
            return ToolResult(
                tool_name="search",
                input_args=query,
                output=f"[{best_match[0]}]: {best_match[1]}",
                success=True
            )
        else:
            return ToolResult(
                tool_name="search",
                input_args=query,
                output="No relevant information found.",
                success=True  # Not finding is still a valid result
            )


class CodeQualityChecker:
    """
    Simple code quality checker for generated Python code.

    Checks for:
    - Syntax validity
    - Basic style issues
    - Common errors
    """

    @classmethod
    def check(cls, code: str) -> dict:
        """
        Check code quality and return a score with feedback.

        Returns dict with:
        - valid: bool - Does the code parse?
        - score: float 0-100 - Quality score
        - issues: list[str] - Issues found
        """
        issues = []
        score = 100

        # Check 1: Syntax validity
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            return {
                "valid": False,
                "score": 0,
                "issues": [f"Syntax error: {e}"]
            }

        # Check 2: Code length (penalize overly verbose)
        lines = [l for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
        if len(lines) > 20:
            issues.append("Code is overly verbose (>20 lines)")
            score -= 10

        # Check 3: Has result variable
        if "result" not in code and "answer" not in code:
            issues.append("Missing 'result' or 'answer' variable")
            score -= 20

        # Check 4: No forbidden imports
        if "import os" in code or "import sys" in code or "import subprocess" in code:
            issues.append("Contains potentially unsafe imports")
            score -= 30

        # Check 5: Basic style - variable naming
        import re
        single_letter_vars = re.findall(r'\b([a-z])\s*=', code)
        meaningful_vars = re.findall(r'\b([a-z_][a-z0-9_]{2,})\s*=', code, re.IGNORECASE)
        if len(single_letter_vars) > 3 and len(meaningful_vars) < 2:
            issues.append("Poor variable naming (too many single-letter vars)")
            score -= 10

        # Check 6: Has comments for complex code
        if len(lines) > 5 and "#" not in code:
            issues.append("Complex code without comments")
            score -= 5

        return {
            "valid": True,
            "score": max(0, score),
            "issues": issues
        }


class PythonExecutor:
    """
    Safe Python code executor for PAL (Program-Aided Language Models).

    Executes simple Python code and returns the result.
    Heavily restricted for safety.
    """

    ALLOWED_BUILTINS = {
        'abs': abs,
        'all': all,
        'any': any,
        'bool': bool,
        'dict': dict,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'int': int,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'pow': pow,
        'print': print,
        'range': range,
        'reversed': reversed,
        'round': round,
        'set': set,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'zip': zip,
    }

    @classmethod
    def execute(cls, code: str) -> ToolResult:
        """
        Execute Python code safely and return the result.

        The code should end with a variable named 'result' or 'answer'
        that contains the final output.
        """
        # Security checks
        forbidden = ['import', 'exec', 'eval', '__', 'open', 'file', 'os.', 'sys.',
                     'subprocess', 'compile', 'globals', 'locals', 'getattr', 'setattr']

        code_lower = code.lower()
        for forbidden_word in forbidden:
            if forbidden_word in code_lower:
                return ToolResult(
                    tool_name="python",
                    input_args=code[:100] + "..." if len(code) > 100 else code,
                    output="",
                    success=False,
                    error=f"Forbidden operation: {forbidden_word}"
                )

        try:
            # Create restricted execution environment
            safe_globals = {
                '__builtins__': cls.ALLOWED_BUILTINS,
                'math': math,
            }
            safe_locals = {}

            # Execute the code
            exec(code, safe_globals, safe_locals)

            # Look for result
            result = safe_locals.get('result') or safe_locals.get('answer') or safe_locals.get('output')

            if result is not None:
                return ToolResult(
                    tool_name="python",
                    input_args=code[:100] + "..." if len(code) > 100 else code,
                    output=str(result),
                    success=True
                )
            else:
                # Return all local variables if no explicit result
                return ToolResult(
                    tool_name="python",
                    input_args=code[:100] + "..." if len(code) > 100 else code,
                    output=str(safe_locals),
                    success=True
                )

        except Exception as e:
            return ToolResult(
                tool_name="python",
                input_args=code[:100] + "..." if len(code) > 100 else code,
                output="",
                success=False,
                error=str(e)
            )


class ToolRegistry:
    """Registry of available tools with their descriptions."""

    TOOLS = {
        "calculator": {
            "name": "calculator",
            "description": "Evaluate mathematical expressions. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, abs, round, pi, e.",
            "usage": "calculator(expression)",
            "examples": ["calculator(2 + 2)", "calculator(sqrt(16))", "calculator(100 * 0.8)"],
            "function": Calculator.calculate,
        },
        "search": {
            "name": "search",
            "description": "Search for factual information. Returns relevant knowledge from the database.",
            "usage": "search(query)",
            "examples": ["search(speed of light)", "search(python list comprehension)"],
            "function": SearchTool.search,
        },
        "python": {
            "name": "python",
            "description": "Execute Python code. Code should set a variable named 'result' with the answer.",
            "usage": "python(code)",
            "examples": ["python(result = sum([1,2,3,4,5]))", "python(result = [x**2 for x in range(5)])"],
            "function": PythonExecutor.execute,
        },
    }

    @classmethod
    def get_tool_descriptions(cls) -> str:
        """Get formatted descriptions of all tools for prompts."""
        lines = ["Available Tools:"]
        for name, tool in cls.TOOLS.items():
            lines.append(f"\n{tool['name']}: {tool['description']}")
            lines.append(f"  Usage: {tool['usage']}")
            lines.append(f"  Examples: {', '.join(tool['examples'])}")
        return "\n".join(lines)

    @classmethod
    def execute_tool(cls, tool_name: str, args: str) -> ToolResult:
        """Execute a tool by name with given arguments."""
        tool_name = tool_name.lower().strip()

        if tool_name not in cls.TOOLS:
            return ToolResult(
                tool_name=tool_name,
                input_args=args,
                output="",
                success=False,
                error=f"Unknown tool: {tool_name}. Available: {list(cls.TOOLS.keys())}"
            )

        return cls.TOOLS[tool_name]["function"](args)

    @classmethod
    def parse_tool_call(cls, text: str) -> tuple[str, str] | None:
        """
        Parse a tool call from LLM output.

        Supports formats:
        - calculator(2 + 2)
        - Action: calculator
          Input: 2 + 2
        - Tool: search
          Query: python list comprehension
        """
        # Format 1: function(args)
        match = re.search(r'(\w+)\s*\(\s*(.+?)\s*\)', text, re.DOTALL)
        if match:
            tool_name = match.group(1).lower()
            if tool_name in cls.TOOLS:
                return (tool_name, match.group(2))

        # Format 2: Action/Tool + Input/Query
        action_match = re.search(r'(?:Action|Tool):\s*(\w+)', text, re.IGNORECASE)
        input_match = re.search(r'(?:Input|Query|Args):\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)

        if action_match and input_match:
            tool_name = action_match.group(1).lower()
            if tool_name in cls.TOOLS:
                return (tool_name, input_match.group(1).strip())

        return None


# Test the tools
if __name__ == "__main__":
    print("=== Testing Calculator ===")
    tests = [
        "2 + 2",
        "7 * 2",
        "7 * 2 * 0.8",  # Apple discount problem
        "100000 - 80000",
        "(100000 - 80000) / 80000 * 100",  # Percentage increase
        "sqrt(16)",
        "pi * 2",
    ]
    for expr in tests:
        result = Calculator.calculate(expr)
        print(f"  {expr} = {result.output}" if result.success else f"  {expr} ERROR: {result.error}")

    print("\n=== Testing Search ===")
    queries = [
        "cloud computing benefits",
        "how to calculate discount",
        "react vs vue comparison",
    ]
    for query in queries:
        result = SearchTool.search(query)
        print(f"  Q: {query}")
        print(f"  A: {result.output[:100]}...")

    print("\n=== Testing Python Executor ===")
    codes = [
        "result = sum([1, 2, 3, 4, 5])",
        "result = [x**2 for x in range(5)]",
        """
# Calculate apple discount
price_per_apple = 2
num_apples = 7
total = price_per_apple * num_apples
discount = 0.20 if num_apples >= 5 else 0
final_price = total * (1 - discount)
result = final_price
""",
    ]
    for code in codes:
        result = PythonExecutor.execute(code)
        print(f"  Code: {code.strip()[:50]}...")
        print(f"  Result: {result.output}" if result.success else f"  ERROR: {result.error}")

    print("\n=== Tool Descriptions ===")
    print(ToolRegistry.get_tool_descriptions())
