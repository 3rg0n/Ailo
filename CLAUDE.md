# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains **prompt engineering research** that empirically compares 9 prompting techniques across multiple LLM models. Originally started as "Ailo" (a structured prompting framework), it evolved into a research project after testing revealed nuanced effectiveness across different models and use cases.

**Core Finding**: No single prompting technique is universally best. Budget models benefit from CoT, premium models perform well with zero-shot, and use case matters more than technique complexity.

## Repository Structure

```
├── readme.md                          # Main documentation and research findings
├── research/                          # All benchmark code and results
│   ├── unified_benchmark_v2.py       # Main benchmark runner (LATEST)
│   ├── multi_provider_client.py      # Unified API client (Bedrock, OpenAI, Gemini)
│   ├── evaluation.py                 # Evaluation module with multiple matching strategies
│   ├── test_prompts_v4.py           # Test suite with 9 prompting styles (LATEST)
│   ├── code_benchmark_v2.py         # Code generation benchmarks
│   ├── agentic_benchmark.py         # Agentic/tool-use benchmarks
│   └── results/                      # JSON benchmark results
```

## Running Benchmarks

### Prerequisites

The research directory requires API credentials in `research/.env`:
```bash
AWS_ACCESS_KEY_ID=...          # For AWS Bedrock
AWS_SECRET_ACCESS_KEY=...
OPENAI_KEY=...                 # For OpenAI models
GEMINI_KEY=...                 # For Google Gemini
```

Python dependencies (assumed installed):
- `boto3` - AWS Bedrock
- `openai` - OpenAI API
- `google-generativeai` - Google Gemini

### Running the Main Benchmark

```bash
cd research

# Run unified benchmark on a specific model
python unified_benchmark_v2.py --model gpt-4o-mini
python unified_benchmark_v2.py --model claude-sonnet
python unified_benchmark_v2.py --model gemini-2.0-flash

# Test specific prompting styles only
python unified_benchmark_v2.py --model nova-micro --styles zero_shot cot few_shot

# Enable LLM-as-judge evaluation with Opus 4.5 (adds cost but more accurate)
python unified_benchmark_v2.py --model gpt-4o-mini --llm-judge
```

Available models: `nova-micro`, `nova-lite`, `nova-pro`, `claude-haiku`, `claude-sonnet`, `claude-opus`, `mistral-7b`, `mistral-large`, `llama3-70b`, `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `o1-mini`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`

### Viewing Available Models

```bash
python multi_provider_client.py list
python multi_provider_client.py list bedrock
python multi_provider_client.py test claude-haiku
```

## Architecture Details

### Prompting Styles Tested

1. **zero_shot**: Plain natural language (baseline)
2. **few_shot**: Includes 1-3 examples demonstrating the task
3. **cot**: Chain-of-Thought with explicit step-by-step instructions
4. **schema**: Structured Ailo format (MODE/ACT/OBJ/TAGS/OUTPUT)
5. **meta**: LLM designs its own solving approach first
6. **gen_knowledge**: Generate domain knowledge before answering
7. **directional**: Hints/keywords to guide without revealing answer
8. **tot**: Tree-of-Thoughts (TRUE multi-turn, 4 API calls exploring paths)
9. **self_consistency**: Multiple solution methods reconciled (TRUE multi-turn, 4 calls)
10. **verbalized_sampling**: Ask for distribution of responses with probabilities (VS)

All styles are defined in `test_prompts_v4.py` as `MultiStylePrompt` objects.

### Verbalized Sampling (VS) - arXiv:2510.01171

VS is a technique from the paper "Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity". Instead of asking for a single response, VS prompts the model to generate a distribution of responses with probabilities:

```
Generate 5 different solutions to this problem, each with a probability score (0.0-1.0).

Format:
Response 1 (Prob: X.XX): [solution]
Response 2 (Prob: X.XX): [solution]
...
```

**Key findings from the paper:**
- Mode collapse stems from "typicality bias" in preference data (humans prefer familiar text)
- VS increases diversity by 1.6-2.1x without sacrificing quality
- Larger models benefit MORE from VS (emergent scaling trend)
- Best for creative tasks, synthetic data generation, open-ended QA

**VS-specific metrics tracked:**
- `diversity.combined`: 0-1 score measuring output diversity (lexical + semantic)
- `vs_parse_success`: Whether the structured VS output was parsed correctly
- `any_correct_rate`: Whether at least 1 of the 5 responses is correct
- `top-1_accuracy`: Accuracy of the highest-probability response

### Evaluation System

The evaluation module (`evaluation.py`) supports multiple strategies:

**Correctness evaluation:**
- **Exact match**: Normalized string comparison
- **Fuzzy match**: Substring matching with F1 scoring
- **Numeric**: Extract numbers with tolerance (e.g., $11.20 ≈ 11.2)
- **Keywords**: Check for required terms with threshold
- **Contains**: Answer appears anywhere in response
- **Code execution**: Run Python code against test cases with linting
- **LLM-as-judge**: Opus 4.5 scores correctness, completeness, clarity, relevance

**Diversity evaluation (for VS):**
- **Lexical diversity**: 1 - mean ROUGE-L overlap (lower overlap = more diverse)
- **Semantic diversity**: 1 - mean Jaccard similarity of tokens
- **N-gram diversity**: Distinct-2 and Distinct-3 metrics
- **Combined diversity**: Weighted average (35% lexical + 35% semantic + 15% bigram + 15% trigram)

Each test case specifies its `EvalType` and expected outputs. Combined scoring: 60% deterministic + 40% LLM judge.

### Multi-Provider Client

`MultiProviderClient` provides a unified interface across:
- **AWS Bedrock**: Supports Claude, Nova, Mistral, Llama
- **OpenAI**: GPT-4o, GPT-4o-mini, o1-mini, GPT-3.5-turbo
- **Google Gemini**: 2.0 Flash, 1.5 Flash, 1.5 Pro, 2.0 Flash Thinking

Key features:
- Lazy provider initialization (only loads libraries when needed)
- Automatic token counting across providers
- Rate limiting and retry logic for Gemini
- Multi-turn conversation support for ToT and Self-Consistency
- Loads credentials from `.env` file in research directory

### Multi-Turn Techniques

ToT and Self-Consistency use **TRUE multi-turn** conversations:

**ToT (Tree-of-Thoughts)**:
1. Turn 1: Explore Path A (direct approach)
2. Turn 2: Explore Path B (alternative method)
3. Turn 3: Explore Path C (verification approach)
4. Turn 4: Synthesize and select best answer

**Self-Consistency**:
1. Turn 1: Solve with Method 1 (standard)
2. Turn 2: Solve with Method 2 (alternative)
3. Turn 3: Solve with Method 3 (verification)
4. Turn 4: Compare and reconcile answers

These require 4 API calls each and use full conversation history, resulting in 7-17x token overhead compared to single-call techniques.

## Adding New Test Cases

To add a new benchmark test:

1. Open `research/test_prompts_v4.py`
2. Add a new `MultiStylePrompt` object to `MULTI_STYLE_PROMPTS` list
3. Define evaluation config (`eval_type`, `expected_answers`, `expected_number`, or `expected_keywords`)
4. Write prompt variants for each style (minimum: `zero_shot`)

Example:
```python
MultiStylePrompt(
    id="logic-puzzle-01",
    name="Logic Puzzle",
    category=TaskCategory.LOGIC,
    eval_type=EvalType.CONTAINS,
    expected_answers=["Tuesday", "tuesday"],

    zero_shot="If yesterday was Monday, what is tomorrow?",

    cot="""If yesterday was Monday, what is tomorrow?
    Let's work through this step-by-step:
    1. Identify what day today is
    2. Calculate what day tomorrow will be""",

    schema="""ACT = Solve
    OBJ = Date logic puzzle
    TAGS = [Format:Final answer only]
    INPUT = If yesterday was Monday, what is tomorrow?
    OUTPUT = Day of the week"""
)
```

## Working with Documentation

The main documentation (`readme.md`) follows this structure:
1. Project story (Ailo → Research pivot)
2. Key findings with data tables
3. Results by model tier (budget/mid/premium)
4. Use case recommendations
5. Methodology details
6. Instructions for running benchmarks

When updating findings:
- Preserve the markdown table formatting
- Keep Combined Score as primary metric (60% deterministic + 40% LLM judge)
- Update both accuracy percentages and token counts
- Include model tier breakdowns (budget/mid/premium)
