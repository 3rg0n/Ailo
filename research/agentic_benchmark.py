"""
Agentic Prompting Benchmark Runner

Tests prompting techniques that require multi-turn or tool use:
- ReAct: Reason + Act loop with real tool execution
- PAL: Program-Aided Language - generate and execute code
- Prompt Chaining: Multi-step with output passing
- Reflexion: Generate, critique, and retry

Each technique is compared against zero-shot baseline.
"""

import json
import time
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from bedrock_client import BedrockClient, ALL_MODELS, ModelConfig
from tools import ToolRegistry, Calculator, PythonExecutor, SearchTool, ToolResult

console = Console()

MAX_REACT_STEPS = 5
MAX_REFLEXION_ATTEMPTS = 2


@dataclass
class AgenticResult:
    """Result from an agentic prompt execution."""
    prompt_id: str
    technique: str
    model_id: str
    final_answer: str
    steps: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    num_llm_calls: int = 0
    num_tool_calls: int = 0
    latency_ms: float = 0
    success: bool = False
    criteria_results: dict = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class AgenticTestCase:
    """Test case for agentic benchmarks."""
    id: str
    name: str
    task: str
    expected_answer: str
    check_answer: callable  # Function to verify answer


# ============================================================================
# Test Cases for Agentic Benchmarks
# ============================================================================

AGENTIC_TEST_CASES = [
    AgenticTestCase(
        id="agent-math-01",
        name="Apple Discount",
        task="A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount. How much would 7 apples cost? Give the final price.",
        expected_answer="11.20",
        check_answer=lambda ans: "11.2" in ans or "11.20" in ans,
    ),
    AgenticTestCase(
        id="agent-math-02",
        name="Percentage Increase",
        task="A company's revenue increased from $80,000 to $100,000. What is the percentage increase?",
        expected_answer="25%",
        check_answer=lambda ans: "25" in ans,
    ),
    AgenticTestCase(
        id="agent-math-03",
        name="Compound Calculation",
        task="If you invest $1000 at 5% annual interest, how much will you have after 3 years with simple interest?",
        expected_answer="1150",
        check_answer=lambda ans: "1150" in ans or "1,150" in ans,
    ),
    AgenticTestCase(
        id="agent-knowledge-01",
        name="Cloud Benefits",
        task="What are 3 key benefits of cloud computing for businesses?",
        expected_answer="cost, scalability, accessibility",
        check_answer=lambda ans: sum(1 for kw in ["cost", "scal", "access", "flexib"] if kw.lower() in ans.lower()) >= 2,
    ),
    AgenticTestCase(
        id="agent-code-01",
        name="List Sum",
        task="Calculate the sum of squares of the first 10 positive integers (1^2 + 2^2 + ... + 10^2).",
        expected_answer="385",
        check_answer=lambda ans: "385" in ans,
    ),
]


class AgenticBenchmarkRunner:
    """Runs agentic benchmarks with tool execution."""

    def __init__(self, output_dir: str = "results"):
        self.client = BedrockClient()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def _call_llm(self, prompt: str, model_config: ModelConfig) -> dict:
        """Make a single LLM call and return response with metadata."""
        result = self.client.invoke(
            prompt=prompt,
            model_config=model_config,
            temperature=0.3,  # Lower temp for more consistent tool use
        )
        return result

    # ========================================================================
    # Zero-shot Baseline
    # ========================================================================

    def run_zero_shot(
        self,
        test_case: AgenticTestCase,
        model_config: ModelConfig,
    ) -> AgenticResult:
        """Run simple zero-shot prompt as baseline."""
        start_time = time.time()

        prompt = f"{test_case.task}\n\nProvide your answer clearly."

        result = self._call_llm(prompt, model_config)
        latency_ms = (time.time() - start_time) * 1000

        return AgenticResult(
            prompt_id=test_case.id,
            technique="zero_shot",
            model_id=model_config.model_id,
            final_answer=result["response"],
            steps=[{"type": "llm", "prompt": prompt[:100], "response": result["response"][:200]}],
            total_input_tokens=result["input_tokens"],
            total_output_tokens=result["output_tokens"],
            total_tokens=result["input_tokens"] + result["output_tokens"],
            num_llm_calls=1,
            num_tool_calls=0,
            latency_ms=latency_ms,
            success=test_case.check_answer(result["response"]),
            criteria_results={"correct_answer": test_case.check_answer(result["response"])},
            timestamp=datetime.now().isoformat(),
        )

    # ========================================================================
    # ReAct: Reason + Act Loop
    # ========================================================================

    def run_react(
        self,
        test_case: AgenticTestCase,
        model_config: ModelConfig,
    ) -> AgenticResult:
        """
        Run ReAct (Reason + Act) loop.

        The model alternates between:
        1. Thought: Reasoning about what to do
        2. Action: Calling a tool
        3. Observation: Seeing the tool result
        4. Repeat until Final Answer
        """
        start_time = time.time()

        tool_descriptions = ToolRegistry.get_tool_descriptions()

        system_prompt = f"""You are a helpful assistant that solves problems step by step using tools.

{tool_descriptions}

For each step, use this format:
Thought: [your reasoning about what to do next]
Action: [tool_name]
Input: [tool input]

After you receive an Observation with the tool result, continue reasoning.
When you have the final answer, respond with:
Thought: [final reasoning]
Final Answer: [your answer]

Always show your work and use tools to verify calculations."""

        conversation = f"{system_prompt}\n\nTask: {test_case.task}\n\nBegin:\n"

        steps = []
        total_input = 0
        total_output = 0
        num_tool_calls = 0
        final_answer = ""

        for step_num in range(MAX_REACT_STEPS):
            # Get LLM response
            result = self._call_llm(conversation, model_config)
            response = result["response"]
            total_input += result["input_tokens"]
            total_output += result["output_tokens"]

            steps.append({
                "step": step_num + 1,
                "type": "llm",
                "response": response[:300]
            })

            # Check for final answer
            if "Final Answer:" in response:
                final_match = re.search(r'Final Answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
                if final_match:
                    final_answer = final_match.group(1).strip()
                break

            # Parse tool call
            tool_call = ToolRegistry.parse_tool_call(response)
            if tool_call:
                tool_name, tool_args = tool_call
                tool_result = ToolRegistry.execute_tool(tool_name, tool_args)
                num_tool_calls += 1

                observation = f"Observation: {tool_result.output}" if tool_result.success else f"Observation: Error - {tool_result.error}"

                steps.append({
                    "step": step_num + 1,
                    "type": "tool",
                    "tool": tool_name,
                    "input": tool_args,
                    "output": tool_result.output if tool_result.success else tool_result.error
                })

                conversation += f"\n{response}\n{observation}\n"
            else:
                # No tool call found, model might be confused
                conversation += f"\n{response}\n\nPlease use a tool or provide Final Answer:\n"

        latency_ms = (time.time() - start_time) * 1000

        # If no explicit final answer, use last response
        if not final_answer:
            final_answer = response

        return AgenticResult(
            prompt_id=test_case.id,
            technique="react",
            model_id=model_config.model_id,
            final_answer=final_answer,
            steps=steps,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            num_llm_calls=len([s for s in steps if s["type"] == "llm"]),
            num_tool_calls=num_tool_calls,
            latency_ms=latency_ms,
            success=test_case.check_answer(final_answer),
            criteria_results={"correct_answer": test_case.check_answer(final_answer)},
            timestamp=datetime.now().isoformat(),
        )

    # ========================================================================
    # PAL: Program-Aided Language Models
    # ========================================================================

    def run_pal(
        self,
        test_case: AgenticTestCase,
        model_config: ModelConfig,
    ) -> AgenticResult:
        """
        Run PAL (Program-Aided Language).

        Model generates Python code, we execute it, return the result.
        """
        start_time = time.time()

        prompt = f"""Solve the following problem by writing Python code.

Problem: {test_case.task}

Write Python code that calculates the answer. Store the final answer in a variable called 'result'.
Only output the Python code, no explanations.

```python
"""

        steps = []
        total_input = 0
        total_output = 0

        # Get code from LLM
        result = self._call_llm(prompt, model_config)
        response = result["response"]
        total_input += result["input_tokens"]
        total_output += result["output_tokens"]

        # Extract code
        code = response
        if "```" in code:
            # Extract from code block
            code_match = re.search(r'```(?:python)?\s*(.+?)```', code, re.DOTALL)
            if code_match:
                code = code_match.group(1)

        steps.append({
            "type": "llm",
            "prompt": prompt[:100],
            "code": code[:300]
        })

        # Execute code
        exec_result = PythonExecutor.execute(code)

        steps.append({
            "type": "tool",
            "tool": "python",
            "input": code[:100],
            "output": exec_result.output if exec_result.success else exec_result.error
        })

        final_answer = exec_result.output if exec_result.success else f"Error: {exec_result.error}"

        latency_ms = (time.time() - start_time) * 1000

        return AgenticResult(
            prompt_id=test_case.id,
            technique="pal",
            model_id=model_config.model_id,
            final_answer=final_answer,
            steps=steps,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            num_llm_calls=1,
            num_tool_calls=1,
            latency_ms=latency_ms,
            success=test_case.check_answer(final_answer),
            criteria_results={"correct_answer": test_case.check_answer(final_answer)},
            timestamp=datetime.now().isoformat(),
        )

    # ========================================================================
    # Prompt Chaining
    # ========================================================================

    def run_prompt_chaining(
        self,
        test_case: AgenticTestCase,
        model_config: ModelConfig,
    ) -> AgenticResult:
        """
        Run Prompt Chaining.

        Break problem into steps, each step's output feeds into the next.
        """
        start_time = time.time()

        steps = []
        total_input = 0
        total_output = 0

        # Step 1: Understand and plan
        prompt1 = f"""Task: {test_case.task}

Step 1: Break down this problem into smaller steps. What calculations or lookups are needed?
List the steps clearly."""

        result1 = self._call_llm(prompt1, model_config)
        total_input += result1["input_tokens"]
        total_output += result1["output_tokens"]
        steps.append({"step": 1, "type": "llm", "purpose": "plan", "response": result1["response"][:200]})

        # Step 2: Execute the plan
        prompt2 = f"""Original Task: {test_case.task}

Your Plan:
{result1["response"]}

Step 2: Now execute each step of your plan. Show all calculations clearly."""

        result2 = self._call_llm(prompt2, model_config)
        total_input += result2["input_tokens"]
        total_output += result2["output_tokens"]
        steps.append({"step": 2, "type": "llm", "purpose": "execute", "response": result2["response"][:200]})

        # Step 3: Synthesize final answer
        prompt3 = f"""Original Task: {test_case.task}

Your Calculations:
{result2["response"]}

Step 3: Based on your work above, what is the final answer? State it clearly and concisely."""

        result3 = self._call_llm(prompt3, model_config)
        total_input += result3["input_tokens"]
        total_output += result3["output_tokens"]
        steps.append({"step": 3, "type": "llm", "purpose": "synthesize", "response": result3["response"][:200]})

        final_answer = result3["response"]
        latency_ms = (time.time() - start_time) * 1000

        return AgenticResult(
            prompt_id=test_case.id,
            technique="chaining",
            model_id=model_config.model_id,
            final_answer=final_answer,
            steps=steps,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            num_llm_calls=3,
            num_tool_calls=0,
            latency_ms=latency_ms,
            success=test_case.check_answer(final_answer),
            criteria_results={"correct_answer": test_case.check_answer(final_answer)},
            timestamp=datetime.now().isoformat(),
        )

    # ========================================================================
    # Reflexion
    # ========================================================================

    def run_reflexion(
        self,
        test_case: AgenticTestCase,
        model_config: ModelConfig,
    ) -> AgenticResult:
        """
        Run Reflexion.

        1. Generate initial answer
        2. Critique the answer
        3. If critique finds issues, regenerate
        """
        start_time = time.time()

        steps = []
        total_input = 0
        total_output = 0
        final_answer = ""

        for attempt in range(MAX_REFLEXION_ATTEMPTS):
            # Generate answer
            if attempt == 0:
                gen_prompt = f"""Task: {test_case.task}

Solve this problem step by step and provide your answer."""
            else:
                gen_prompt = f"""Task: {test_case.task}

Previous Attempt: {previous_answer}

Critique: {critique}

Based on the critique, solve this problem again more carefully. Fix any errors identified."""

            gen_result = self._call_llm(gen_prompt, model_config)
            total_input += gen_result["input_tokens"]
            total_output += gen_result["output_tokens"]
            previous_answer = gen_result["response"]

            steps.append({
                "attempt": attempt + 1,
                "type": "generate",
                "response": previous_answer[:200]
            })

            # Critique the answer
            critique_prompt = f"""Task: {test_case.task}

Proposed Answer:
{previous_answer}

Critique this answer:
1. Is the reasoning correct?
2. Are the calculations accurate?
3. Does it fully answer the question?
4. Are there any errors?

If the answer is correct, respond with "CORRECT".
If there are issues, explain what's wrong."""

            critique_result = self._call_llm(critique_prompt, model_config)
            total_input += critique_result["input_tokens"]
            total_output += critique_result["output_tokens"]
            critique = critique_result["response"]

            steps.append({
                "attempt": attempt + 1,
                "type": "critique",
                "response": critique[:200]
            })

            # Check if critique says it's correct
            if "CORRECT" in critique.upper() or "correct" in critique.lower()[:50]:
                final_answer = previous_answer
                break

        if not final_answer:
            final_answer = previous_answer

        latency_ms = (time.time() - start_time) * 1000

        return AgenticResult(
            prompt_id=test_case.id,
            technique="reflexion",
            model_id=model_config.model_id,
            final_answer=final_answer,
            steps=steps,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_tokens=total_input + total_output,
            num_llm_calls=len([s for s in steps if s["type"] in ["generate", "critique"]]),
            num_tool_calls=0,
            latency_ms=latency_ms,
            success=test_case.check_answer(final_answer),
            criteria_results={"correct_answer": test_case.check_answer(final_answer)},
            timestamp=datetime.now().isoformat(),
        )

    # ========================================================================
    # Run Full Benchmark
    # ========================================================================

    def run_benchmark(
        self,
        model_key: str = "nova-micro",
        test_ids: Optional[list[str]] = None,
        techniques: Optional[list[str]] = None,
    ) -> list[AgenticResult]:
        """Run full agentic benchmark suite."""
        model_config = ALL_MODELS[model_key]
        results = []

        if techniques is None:
            techniques = ["zero_shot", "react", "pal", "chaining", "reflexion"]

        tests = AGENTIC_TEST_CASES
        if test_ids:
            tests = [t for t in AGENTIC_TEST_CASES if t.id in test_ids]

        console.print(Panel(
            f"[bold]Agentic Prompting Benchmark[/bold]\n\n"
            f"Model: {model_config.name}\n"
            f"Tests: {len(tests)}\n"
            f"Techniques: {', '.join(techniques)}",
            title="Configuration"
        ))

        technique_methods = {
            "zero_shot": self.run_zero_shot,
            "react": self.run_react,
            "pal": self.run_pal,
            "chaining": self.run_prompt_chaining,
            "reflexion": self.run_reflexion,
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            total_runs = len(tests) * len(techniques)
            task = progress.add_task("Running...", total=total_runs)

            for test_case in tests:
                for technique in techniques:
                    progress.update(task, description=f"{test_case.name} - {technique}")

                    try:
                        method = technique_methods[technique]
                        result = method(test_case, model_config)
                        results.append(result)

                        status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
                        console.print(f"  {test_case.id} | {technique:10} | {status} | {result.total_tokens} tokens | {result.num_llm_calls} calls")

                    except Exception as e:
                        console.print(f"  [red]Error: {test_case.id} - {technique}: {e}[/red]")

                    progress.advance(task)
                    time.sleep(0.5)  # Rate limiting

        return results

    def print_summary(self, results: list[AgenticResult]):
        """Print summary of results."""
        console.print("\n")

        # Group by technique
        technique_stats = {}
        for r in results:
            if r.technique not in technique_stats:
                technique_stats[r.technique] = {
                    "passed": 0, "total": 0, "tokens": 0, "llm_calls": 0, "tool_calls": 0
                }
            technique_stats[r.technique]["total"] += 1
            technique_stats[r.technique]["passed"] += 1 if r.success else 0
            technique_stats[r.technique]["tokens"] += r.total_tokens
            technique_stats[r.technique]["llm_calls"] += r.num_llm_calls
            technique_stats[r.technique]["tool_calls"] += r.num_tool_calls

        # Summary table
        table = Table(title="Agentic Benchmark Results")
        table.add_column("Technique", style="cyan")
        table.add_column("Pass Rate", justify="center")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("Avg LLM Calls", justify="right")
        table.add_column("Avg Tool Calls", justify="right")

        baseline_rate = 0
        baseline_tokens = 0
        if "zero_shot" in technique_stats:
            zs = technique_stats["zero_shot"]
            baseline_rate = zs["passed"] / zs["total"] * 100 if zs["total"] > 0 else 0
            baseline_tokens = zs["tokens"] / zs["total"] if zs["total"] > 0 else 0

        for technique, stats in technique_stats.items():
            rate = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            avg_tokens = stats["tokens"] / stats["total"] if stats["total"] > 0 else 0
            avg_llm = stats["llm_calls"] / stats["total"] if stats["total"] > 0 else 0
            avg_tool = stats["tool_calls"] / stats["total"] if stats["total"] > 0 else 0

            rate_diff = f" ({rate - baseline_rate:+.0f}%)" if technique != "zero_shot" else " (baseline)"
            token_diff = f" ({(avg_tokens - baseline_tokens) / baseline_tokens * 100:+.0f}%)" if technique != "zero_shot" and baseline_tokens > 0 else ""

            table.add_row(
                technique,
                f"{rate:.0f}%{rate_diff}",
                f"{avg_tokens:.0f}{token_diff}",
                f"{avg_llm:.1f}",
                f"{avg_tool:.1f}",
            )

        console.print(table)

    def save_results(self, results: list[AgenticResult], model_key: str):
        """Save results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"agentic_{model_key}_{timestamp}.json"

        data = {
            "metadata": {
                "model": model_key,
                "timestamp": timestamp,
                "total_tests": len(results),
            },
            "results": [
                {
                    "prompt_id": r.prompt_id,
                    "technique": r.technique,
                    "model_id": r.model_id,
                    "final_answer": r.final_answer[:500],
                    "success": r.success,
                    "total_tokens": r.total_tokens,
                    "num_llm_calls": r.num_llm_calls,
                    "num_tool_calls": r.num_tool_calls,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        console.print(f"\n[green]Results saved to: {filename}[/green]")
        return filename


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run agentic prompting benchmark")
    parser.add_argument(
        "--model",
        default="nova-micro",
        choices=list(ALL_MODELS.keys()),
        help="Model to test"
    )
    parser.add_argument(
        "--techniques",
        nargs="*",
        choices=["zero_shot", "react", "pal", "chaining", "reflexion"],
        help="Specific techniques to test"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only run first 2 tests"
    )

    args = parser.parse_args()

    runner = AgenticBenchmarkRunner()

    test_ids = None
    if args.dry_run:
        test_ids = [AGENTIC_TEST_CASES[0].id, AGENTIC_TEST_CASES[1].id]
        console.print("[yellow]Dry run: testing first 2 cases only[/yellow]\n")

    results = runner.run_benchmark(
        model_key=args.model,
        test_ids=test_ids,
        techniques=args.techniques,
    )

    if results:
        runner.print_summary(results)
        runner.save_results(results, args.model)


if __name__ == "__main__":
    main()
