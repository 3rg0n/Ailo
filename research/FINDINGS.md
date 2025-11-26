# Prompting Style Research: Detailed Findings

This document contains the full methodology, results, and analysis from our prompting style benchmarks. For a quick summary, see the main [README](../readme.md).

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Prompting Techniques](#prompting-techniques)
3. [Testing Methodology](#testing-methodology)
4. [Results by Model](#results-by-model)
5. [Results by Use Case](#results-by-use-case)
6. [Agentic Techniques](#agentic-techniques)
7. [Recommendations](#recommendations)
8. [Running the Benchmarks](#running-the-benchmarks)

---

## Executive Summary

### Efficiency Ranking

```
                          Accuracy/Token Efficiency
                          ─────────────────────────────────────────────────
few_shot       ████████████████████   Best ROI: +16% acc, -25% tokens
directional    ██████████████████░░   +22% acc, -10% tokens
zero_shot      ████████████████░░░░   Baseline (great on premium)
schema         ███████████████░░░░░   +5-16% acc, variable overhead
cot            ██████████████░░░░░░   +22% acc, +15% tokens
gen_knowledge  █████████████░░░░░░░   +22% acc, +5% tokens
meta           ███████████░░░░░░░░░   +11% acc, +46% tokens
tot            █████████░░░░░░░░░░░   Often WORSE, +80-130% tokens
self_consist   █████████░░░░░░░░░░░   Same acc, +44-118% tokens
```

### Key Insight

**Simple techniques win.** Few-shot and directional stimulus consistently outperform complex reasoning chains like Tree of Thoughts and Self-Consistency, while using fewer tokens.

---

## Prompting Techniques

### Single-Prompt Techniques (9 tested)

| Technique | Description | Token Overhead |
|-----------|-------------|----------------|
| **Zero-shot** | Plain natural language prompt | Baseline |
| **Few-shot** | Include 1-2 examples before the task | -25% to +30% |
| **Chain-of-Thought (CoT)** | Explicit step-by-step reasoning | +15% to +68% |
| **Schema** | Structured format (ACT/OBJ/TAGS) | +4% to +86% |
| **Meta Prompting** | LLM designs its own approach first | +46% to +123% |
| **Generate Knowledge** | Generate relevant facts, then answer | +5% to +75% |
| **Directional Stimulus** | Hints/keywords to guide response | -10% to +27% |
| **Tree of Thoughts (ToT)** | Explore multiple solution paths | +80% to +166% |
| **Self-Consistency** | Multiple approaches, then reconcile | +44% to +211% |

### Agentic Techniques (4 tested)

| Technique | Description | Infrastructure |
|-----------|-------------|----------------|
| **ReAct** | Reason + Act loop with tools | Calculator, Search, Python |
| **PAL** | Generate and execute Python code | Safe Python sandbox |
| **Prompt Chaining** | Multi-step with output passing | 3-step orchestrator |
| **Reflexion** | Generate, critique, retry | 2-attempt loop |

### Not Testable

| Style | Why |
|-------|-----|
| RAG | Requires vector DB + document corpus |
| ART | Requires external API tools |
| APE | Meta-optimization loop |
| Active-Prompt | Human-in-loop |
| Multimodal CoT | Image input |
| Graph Prompting | Complex graph structure |

---

## Testing Methodology

### How Each Technique Was Tested

Every prompting technique was tested with the **same task** presented in different formats. The only variable is the prompting style.

#### Example: Math Problem

| Technique | Prompt Structure |
|-----------|------------------|
| **Zero-shot** | "Calculate: 7 apples at $2 each with 20% discount" |
| **Few-shot** | "Example: 3 items at $5 = $15. Now solve: 7 apples..." |
| **CoT** | "Solve step by step: 1) Calculate total 2) Apply discount..." |
| **Schema** | `ACT=Calculate OBJ=Price TAGS=[ShowWork]` |
| **Meta** | "First, decide how to solve this. Then execute." |
| **Gen-Knowledge** | "Recall: Discount formula is... Now apply to: 7 apples..." |
| **Directional** | "Calculate price. HINTS: Total=$14, discount=20%" |
| **ToT** | "Try 3 approaches: A) Direct calc B) Unit price C) Ratio" |
| **Self-Consistency** | "Solve 3 ways, compare answers, give final result" |

### Evaluation Criteria

```
CRITERIA TYPES
├── Format Compliance
│   ├── has_bullets: Response contains bullet points
│   ├── has_table: Response has markdown table
│   ├── has_code: Response includes code block
│   └── has_numbered_list: Response has numbered items
│
├── Constraint Adherence
│   ├── is_concise: Under word limit
│   ├── exactly_N: Exactly N items provided
│   └── length_range: Within word count bounds
│
└── Content Accuracy
    ├── correct_answer: Contains expected value (e.g., "11.20")
    ├── contains_keywords: Has required terms
    └── shows_work: Demonstrates reasoning steps
```

### Test Suite

**Single-Prompt Benchmark** (9 test cases):
- Writing: Executive summary, persona adherence
- Reasoning: Math problems, logic puzzles, percentages
- Creative: Story ideas
- Analysis: Framework comparison, pros/cons
- Technical: Code explanation

**Agentic Benchmark** (5 test cases):
- Math: Apple discount ($11.20), percentage increase (25%), compound interest
- Knowledge: Cloud computing benefits
- Code: Sum of squares (385)

### Metrics Captured

```python
{
    "pass_rate": "% of criteria passed",
    "total_tokens": "input + output tokens",
    "input_tokens": "prompt size",
    "output_tokens": "response size",
    "latency_ms": "wall-clock time",
    "num_llm_calls": "API calls made",
    "num_tool_calls": "tools executed (agentic only)"
}
```

---

## Results by Model

### Model Tiers

| Tier | Models | Cost (per 1K tokens) |
|------|--------|---------------------|
| **Budget** | Nova Micro, Mistral 7B | $0.00004 - $0.00015 |
| **Mid** | Claude Haiku 4.5, Nova Lite | $0.0006 - $0.001 |
| **Premium** | Claude Sonnet/Opus, Mistral Large | $0.003 - $0.015 |

---

### Budget Tier

#### Amazon Nova Micro

```
Style              Pass Rate                                      vs Zero   Tokens
─────────────────────────────────────────────────────────────────────────────────
few_shot           ████████████████████  100.0%                    +5.6%     317   WINNER
schema             ████████████████████  100.0%                    +5.6%     440
directional        ████████████████████  100.0%                    +5.6%     410
zero_shot          ███████████████████░   94.4%                  baseline    323
cot                ███████████████████░   94.4%                    same      543
meta               ███████████████████░   94.4%                    same      686
gen_knowledge      ███████████████████░   94.4%                    same      566
self_consistency   ███████████████████░   94.4%                    same      706
tot                █████████████████░░░   83.3%                   -11.1%     746   WORSE!
```

**Winner**: Few-shot (+5.6% accuracy, -2% tokens)

#### Mistral 7B

```
Style              Pass Rate                                      vs Zero   Tokens
─────────────────────────────────────────────────────────────────────────────────
directional        ███████████████████░   94.4%                   +22.2%     375   WINNER
cot                ███████████████████░   94.4%                   +22.2%     480
gen_knowledge      ███████████████████░   94.4%                   +22.2%     440
few_shot           ██████████████████░░   88.9%                   +16.7%     316   Best ROI
schema             ██████████████████░░   88.9%                   +16.7%     402
self_consistency   ██████████████████░░   88.9%                   +16.7%     601
meta               █████████████████░░░   83.3%                   +11.1%     610
tot                ████████████████░░░░   77.8%                    +5.6%     753
zero_shot          ██████████████░░░░░░   72.2%                  baseline    419
```

**Winner**: Directional (+22.2% accuracy, -10.4% tokens)
**Runner-up**: Few-shot (+16.7% accuracy, -24.6% tokens)

---

### Mid Tier

#### Claude Haiku 4.5

```
Style              Pass Rate                                      vs Zero   Tokens
─────────────────────────────────────────────────────────────────────────────────
few_shot           ███████████████████░   94.4%                    +5.6%     297   WINNER
schema             ███████████████████░   94.4%                    +5.6%     480
zero_shot          ██████████████████░░   88.9%                  baseline    313
cot                ██████████████████░░   88.9%                    same      559
gen_knowledge      ██████████████████░░   88.9%                    same      508
directional        █████████████████░░░   83.3%                    -5.6%     379
tot                █████████████████░░░   83.3%                    -5.6%     728
meta               ████████████████░░░░   77.8%                   -11.1%     821
self_consistency   ████████████████░░░░   77.8%                   -11.1%     759
```

**Winner**: Few-shot (+5.6% accuracy, -5.1% tokens)

**Key insight**: Already 88.9% accurate on zero-shot. Complex techniques (Meta, ToT, Self-Consistency) actually hurt performance.

---

### Gemini Models

#### Gemini 2.0 Flash

```
Style              Pass Rate                                      vs Zero   Tokens
─────────────────────────────────────────────────────────────────────────────────
cot                ████████████████████  100.0%                   +11.1%     697
schema             ████████████████████  100.0%                   +11.1%     414   WINNER
few_shot           ███████████████████░   94.4%                    +5.5%     336
directional        ███████████████████░   94.4%                    +5.5%     506
zero_shot          ██████████████████░░   88.9%                  baseline    675
meta               ██████████████████░░   88.9%                    same     1038
gen_knowledge      █████████████████░░░   83.3%                    -5.6%     692
tot                █████████████████░░░   83.3%                    -5.6%    1087
self_consistency   █████████████████░░░   83.3%                    -5.6%    1062
```

**Winner**: Schema (+11.1% accuracy, fewer tokens than CoT)

---

## Results by Use Case

### Code Generation

Tested on 4 JavaScript algorithms (factorial, fibonacci, GCD, primality) measuring similarity to reference implementations:

```
Style              Similarity                                   Correctness   Tokens
─────────────────────────────────────────────────────────────────────────────────────
few_shot           ████████████████████   53.9%                    89.1%       315   WINNER
zero_shot          ████████████████░░░░   41.1%                    84.5%       227
schema             ███████████████░░░░░   37.7%                    90.3%       270
tot                ██████████████░░░░░░   35.3%                    75.8%      1211
cot                ███████████░░░░░░░░░   29.0%                    72.8%       646
directional        ██████████░░░░░░░░░░   24.7%                    76.4%       514
self_consistency   █████████░░░░░░░░░░░   21.7%                    71.4%       943
meta               ████████░░░░░░░░░░░░   18.9%                    64.5%       735
gen_knowledge      ███████░░░░░░░░░░░░░   16.7%                    56.1%       604
```

**Why few-shot wins for code:**
1. Examples demonstrate expected style and structure
2. Models learn naming conventions from examples
3. Avoids verbose explanations that dilute output
4. Lower token overhead than reasoning techniques

#### Per-Model Code Results

```
Model           Best Style      Similarity   Correctness   Tokens
─────────────────────────────────────────────────────────────────
Nova Micro      Zero-shot         51.6%        89.3%         157
Claude Haiku    Few-shot          60.2%        85.1%         262
Mistral 7B      Few-shot          56.8%        92.8%         332
```

**Surprise**: Zero-shot performs well on Nova Micro (51.6%) but poorly on Mistral 7B (27.3%), suggesting model architecture matters significantly.

---

## Agentic Techniques

### Nova Micro - Agentic Benchmark

```
Technique      Pass Rate                              Tokens   LLM Calls   Tool Calls
──────────────────────────────────────────────────────────────────────────────────────
zero_shot      ████████████████████  100%               284        1           0
PAL            ████████████████████  100%               116        1           1   BEST!
chaining       ████████████████████  100%              2014        3           0
reflexion      ████████████████████  100%              1209        2           0
react          ████████████████░░░░   80%               476        1           0
```

**PAL saves 59% tokens** by generating concise code instead of verbose reasoning.

### Mistral 7B - Agentic Benchmark

```
Technique      Pass Rate                              Tokens   Overhead
──────────────────────────────────────────────────────────────────────
zero_shot      ████████████████░░░░   80%               263   baseline
react          ████████████████░░░░   80%               592    +125%
chaining       ████████████████░░░░   80%              1413    +438%
PAL            ████████░░░░░░░░░░░░   40%               146     -44%   Code quality issues
```

**Warning**: For budget models, PAL fails more often because generated code has errors. Reflexion helps by allowing retry.

---

## Recommendations

### By Model Tier

| Model Tier | Best Technique | Why |
|------------|----------------|-----|
| **Budget** (Mistral 7B) | Few-shot or Directional | +16-22% accuracy, saves tokens |
| **Mid** (Claude Haiku) | Few-shot | +5.6% accuracy, -5% tokens |
| **Premium** (Claude Sonnet/Opus) | Zero-shot | Already 95%+ accurate |

### By Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Code generation** | Few-shot | 53.9% similarity, demonstrates style |
| **Math/Logic** | CoT or Directional | Step-by-step reasoning helps |
| **Format-critical** | Schema | Explicit constraints |
| **General tasks** | Few-shot (budget) / Zero-shot (premium) | Best ROI |
| **Token-sensitive** | Few-shot or Directional | Can save tokens |
| **Accuracy-critical** | CoT or Gen-Knowledge | Higher accuracy |
| **Quick prototyping** | Zero-shot | Fastest iteration |

### When to Use Each Style

#### Zero-shot
- Premium models
- Simple, well-defined tasks
- Quick iteration
- Token cost is critical

#### Few-shot (Best ROI for Budget Models)
- Budget/smaller models
- Format clarification needed
- Good representative examples available
- Want accuracy boost without token penalty

#### Chain-of-Thought
- Mathematical calculations
- Multi-step logical reasoning
- When showing work is valuable
- Debugging complex problems

#### Schema (Structured)
- Multiple constraints must be met
- Specific output format required
- Persona/context is critical
- Complex multi-part tasks

#### Directional Stimulus
- Expected answer pattern is known
- Guide without full examples
- Lower overhead than CoT
- Combining with other techniques

#### Tree of Thoughts / Self-Consistency
- High-stakes decisions only
- Token cost not a concern
- **Generally NOT recommended** due to high overhead

---

## Running the Benchmarks

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys (.env file)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
OPENAI_KEY=your_key
GEMINI_KEY=your_key
```

### Run Benchmarks

```bash
cd research

# Single-prompt benchmark (9 styles)
python multi_style_benchmark_v2.py --model nova-micro

# Agentic benchmark (ReAct, PAL, Chaining, Reflexion)
python agentic_benchmark.py --model nova-micro

# Code generation benchmark
python code_benchmark.py --model claude-haiku --output results/code_benchmark.json

# Multi-provider benchmark
python unified_benchmark.py --model gemini-2.0-flash

# List available models
python multi_provider_client.py list
```

### Repository Structure

```
research/
├── bedrock_client.py           # AWS Bedrock API client
├── multi_provider_client.py    # OpenAI, Gemini, Bedrock unified client
├── unified_benchmark.py        # Cross-provider benchmark runner
├── multi_style_benchmark_v2.py # 9-style comprehensive runner
├── agentic_benchmark.py        # Agentic techniques (ReAct, PAL, etc.)
├── code_benchmark.py           # Code generation benchmark
├── test_prompts_v2.py          # 3-style test cases
├── test_prompts_v3.py          # 9-style test cases
├── tools.py                    # Tool implementations for agentic
├── generate_report.py          # Report generator
├── results/                    # Benchmark results (JSON)
└── FINDINGS.md                 # This document
```

---

## Limitations

1. **Sample Size**: 9 test cases per comprehensive run
2. **Single-call Simulation**: ToT and Self-Consistency simulated in single prompt
3. **Model Versions**: Results may vary with model updates
4. **Task Types**: General tasks; specialized domains may differ
5. **Evaluation**: Automated criteria; subjective quality not measured

---

## Key Takeaways

1. **Simple often wins**: Few-shot and Directional beat complex techniques
2. **ROI matters**: Consider accuracy gain vs token cost
3. **Model tier matters**: Budget models benefit most from prompting techniques
4. **Premium models are robust**: Zero-shot works fine for Claude/GPT-4
5. **Overhead adds up**: ToT/Self-Consistency rarely justify their cost

---

*Generated from benchmark data collected November 2025*
*9 prompting techniques tested across AWS Bedrock, OpenAI, and Google Gemini models*
