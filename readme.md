# Prompting Styles Research

**An empirical comparison of prompting techniques across LLM models and use cases.**

---

## The Story

This project started as **Ailo** — a structured prompting framework using schema-based prompts (ACT/OBJ/TAGS) to optimize AI communication. The hypothesis was that structured prompts would consistently outperform natural language.

Then we tested it.

What we found was more nuanced: **schema prompting helps in some cases, but simpler techniques often win**. Few-shot examples beat complex reasoning chains. Premium models don't need elaborate prompts. Token overhead from fancy techniques rarely pays off.

So we pivoted. Instead of promoting one prompting style, we built a **research framework** to answer: *"Which prompting technique should I use for my model and use case?"*

This repository contains:
- Benchmark tooling for 9+ prompting techniques
- Results across budget, mid-tier, and premium models
- Data-driven recommendations by use case

---

## Table of Contents

1. [Key Findings](#key-findings)
2. [Results by Model](#results-by-model)
3. [Code Generation Results](#code-generation-results)
4. [Agentic Techniques](#agentic-techniques)
5. [Methodology](#methodology)
6. [Running Benchmarks](#running-benchmarks)
7. [What Happened to Ailo](#what-happened-to-ailo)

---

## Key Findings

### Efficiency Ranking (Accuracy per Token)

```
few_shot       ████████████████████   Best ROI: +16% accuracy, -25% tokens
directional    ██████████████████░░   +22% accuracy, -10% tokens
zero_shot      ████████████████░░░░   Baseline (works great on premium models)
schema         ███████████████░░░░░   +5-16% accuracy, variable overhead
cot            ██████████████░░░░░░   +22% accuracy, +15% tokens
gen_knowledge  █████████████░░░░░░░   +22% accuracy, +5% tokens
meta           ███████████░░░░░░░░░   +11% accuracy, +46% tokens
tot            █████████░░░░░░░░░░░   Often WORSE accuracy, +80-130% tokens
self_consist   █████████░░░░░░░░░░░   Same accuracy, +44-118% tokens
```

### Winner by Model Tier

```
BUDGET MODELS (Mistral 7B, Nova Micro)
──────────────────────────────────────────────────────────────────────────
Technique      Accuracy   vs Zero    Tokens     Verdict
──────────────────────────────────────────────────────────────────────────
few_shot         88.9%    +16.7%      -25%      WINNER: Best ROI
directional      94.4%    +22.2%      -10%      Runner-up
cot              94.4%    +22.2%      +15%      Good if tokens don't matter
zero_shot        72.2%   baseline   baseline   Needs help

MID-TIER MODELS (Claude Haiku)
──────────────────────────────────────────────────────────────────────────
few_shot         94.4%     +5.6%       -5%      WINNER: Slight edge
schema           94.4%     +5.6%      +53%      Good for format control
zero_shot        88.9%   baseline   baseline   Already solid

PREMIUM MODELS (Claude Sonnet/Opus, GPT-4)
──────────────────────────────────────────────────────────────────────────
zero_shot        95%+    baseline   baseline   WINNER: Already excellent
few_shot          ~0%     minimal     -5%      Marginal benefit
complex           ↓      can hurt    +50%+    Often counterproductive
```

### Recommendations by Use Case

```
USE CASE           RECOMMENDED          WHY
─────────────────────────────────────────────────────────────────────────
Code generation    few_shot             53.9% similarity to reference
Math/Logic         cot or directional   Step-by-step reasoning helps
Format-critical    schema               Explicit constraints
General tasks      few_shot (budget)    Best accuracy/token tradeoff
                   zero_shot (premium)  Already accurate
Token-sensitive    few_shot             Can SAVE tokens while improving
Quick prototyping  zero_shot            Fastest iteration
```

### Five Takeaways

1. **Simple often wins** — Few-shot and directional beat complex techniques
2. **ROI matters** — Consider accuracy gain vs token cost
3. **Model tier matters** — Budget models benefit most from prompting techniques
4. **Premium models are robust** — Zero-shot works fine for Claude/GPT-4
5. **Complex techniques rarely justify cost** — ToT/Self-Consistency add overhead without gains

---

## Results by Model

### Model Tiers Tested

| Tier | Models | Cost (per 1K tokens) |
|------|--------|---------------------|
| **Budget** | Nova Micro, Mistral 7B, GPT-4o-mini | $0.00004 - $0.00015 |
| **Mid** | Claude Haiku 4.5, Nova Lite | $0.0006 - $0.001 |
| **Premium** | Claude Sonnet/Opus, GPT-4o, Mistral Large | $0.003 - $0.015 |

---

### Claude Opus 4.5 (Premium)

```
Style              Pass Rate                                      vs Zero   Tokens
─────────────────────────────────────────────────────────────────────────────────
few_shot           ███████████████████░   94.4%                   +11.1%     261   WINNER
gen_knowledge      ███████████████████░   94.4%                   +11.1%     472
self_consistency   ███████████████████░   94.4%                   +11.1%     715
cot                ██████████████████░░   88.9%                    +5.6%     441
schema             ██████████████████░░   88.9%                    +5.6%     481
directional        ██████████████████░░   88.9%                    +5.6%     298
zero_shot          █████████████████░░░   83.3%                  baseline    253
meta               █████████████████░░░   83.3%                    same      709
tot                █████████████████░░░   83.3%                    same      534
```

### Claude Haiku 4.5 (Mid)

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

**Key insight**: Already 88.9% accurate on zero-shot. Complex techniques (Meta, ToT, Self-Consistency) actually hurt performance.

### Gemini 2.0 Flash

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

### Mistral 7B (Budget)

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

### Amazon Nova Micro (Budget)

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

---

## Code Generation Results

Tested on 4 JavaScript algorithms (factorial, fibonacci, GCD, primality) measuring similarity to reference implementations from [javascript-algorithms](https://github.com/trekhleb/javascript-algorithms):

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

### Per-Model Code Results

```
Model              Best Style      Similarity   Correctness   Tokens
─────────────────────────────────────────────────────────────────────
Gemini 2.0 Flash   Few-shot          57.9%        96.4%         263
Claude Haiku       Few-shot          60.2%        85.1%         262
Mistral 7B         Few-shot          56.8%        92.8%         332
Nova Micro         Zero-shot         51.6%        89.3%         157
```

**Why few-shot wins for code:**
1. Examples demonstrate expected style and structure
2. Models learn naming conventions from examples
3. Avoids verbose explanations that dilute output
4. Lower token overhead than reasoning techniques

---

## Agentic Techniques

Techniques requiring tool execution or multi-turn orchestration:

| Technique | Description | Best For |
|-----------|-------------|----------|
| **ReAct** | Reason + Act loop with tools | Tool-heavy tasks |
| **PAL** | Generate & execute Python code | Math (saves 59% tokens) |
| **Chaining** | Multi-step orchestration | Complex workflows |
| **Reflexion** | Generate, critique, retry | Error recovery |

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

**Warning**: For budget models, PAL fails more often because generated code has errors.

---

## Methodology

### Prompting Techniques Tested

| Technique | Description | Token Overhead |
|-----------|-------------|----------------|
| **zero_shot** | Plain natural language | Baseline |
| **few_shot** | 1-2 examples before task | -25% to +30% |
| **cot** | Step-by-step reasoning | +15% to +68% |
| **schema** | Structured ACT/OBJ/TAGS | +4% to +86% |
| **meta** | LLM designs approach first | +46% to +123% |
| **gen_knowledge** | Generate facts, then answer | +5% to +75% |
| **directional** | Hints/keywords to guide | -10% to +27% |
| **tot** | Multiple solution paths | +80% to +166% |
| **self_consistency** | Multiple approaches, reconcile | +44% to +211% |

### How Each Technique Was Tested

Every technique was tested with the **same task** presented in different formats:

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

**Code Benchmark** (4 algorithms):
- Factorial, Fibonacci, Euclidean GCD, Primality test
- Compared against reference implementations

**Agentic Benchmark** (5 test cases):
- Math with tools, knowledge retrieval, code execution

### Limitations

1. **Sample Size**: 9 test cases per comprehensive run
2. **Single-call Simulation**: ToT and Self-Consistency simulated in single prompt
3. **Model Versions**: Results may vary with model updates
4. **Task Types**: General tasks; specialized domains may differ
5. **Evaluation**: Automated criteria; subjective quality not measured

---

## Running Benchmarks

### Setup

```bash
cd research

# Install dependencies
pip install boto3 openai google-generativeai

# Create .env file with API keys
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
OPENAI_KEY=your_key
GEMINI_KEY=your_key
```

### Run Benchmarks

```bash
# Test a model connection
python multi_provider_client.py claude-haiku

# Run comprehensive benchmark (9 styles)
python unified_benchmark.py --model gemini-2.0-flash

# Run code generation benchmark
python code_benchmark.py --model gpt-4o-mini --output results/code_gpt4o.json

# Run agentic benchmark
python agentic_benchmark.py --model nova-micro

# List all available models
python multi_provider_client.py list
```

### Supported Models

| Provider | Models |
|----------|--------|
| AWS Bedrock | Claude (Haiku, Sonnet, Opus), Nova (Micro, Lite), Mistral (7B, Large), Llama |
| OpenAI | GPT-4o, GPT-4o-mini, GPT-3.5-turbo, o1-mini |
| Google | Gemini 2.0 Flash, 1.5 Flash, 1.5 Pro |

### Repository Structure

```
ailo/
├── readme.md                 # This file
├── research/
│   ├── context.md            # Session context for continuity
│   ├── bedrock_client.py     # AWS Bedrock API client
│   ├── multi_provider_client.py   # Unified client (Bedrock, OpenAI, Gemini)
│   ├── unified_benchmark.py       # Cross-provider benchmark runner
│   ├── code_benchmark.py          # Code generation benchmark
│   ├── agentic_benchmark.py       # ReAct, PAL, Chaining tests
│   ├── test_prompts_v3.py         # Test case definitions
│   └── results/                   # Raw benchmark data (JSON)
```

---

## What Happened to Ailo?

This project started as **Ailo** — a structured prompting framework. The original hypothesis was that schema-based prompts would consistently outperform natural language.

### The Ailo Schema

```
CONTEXT = [Background / why you need this]
PERSONA = [Role for AI to adopt: mentor, critic, analyst...]
MODE    = [Task type: Generate, Evaluate, Compare, Plan...]
ACT     = [What you want done]
OBJ     = [The subject/object to work on]
TAGS    = [
    Format: [list, table, code, JSON...]
    Length: [short, 200 words, 5 bullets...]
    Style:  [formal, casual, technical...]
    Audience: [beginner, expert, executive...]
    Constraints: [no jargon, max 3 steps...]
]
OUTPUT  = [Delivery format: text, code, file...]
```

**Example:**
```
PERSONA = Business analyst briefing an executive
MODE = Summarize
ACT = Summarize
OBJ = Climate policy report
TAGS = [Format:Bullets, Length:5, Audience:Executive, Constraints:No jargon]
OUTPUT = Text
```

### What We Learned

Schema prompting (tested as "schema" style in our benchmarks) performs well for:
- **Format-critical tasks** — When specific output structure matters
- **Complex multi-constraint tasks** — Multiple requirements to satisfy
- **Gemini models** — +11.1% accuracy vs zero-shot

But simpler techniques often win:
- **Few-shot** beats schema for code generation (53.9% vs 37.7% similarity)
- **Directional stimulus** matches schema accuracy with less overhead
- **Zero-shot** works fine on premium models

The research shows when to use schema prompting — and when simpler approaches work better.

---

## License

MIT License — Use this research freely.

---

*Empirical prompting research, November 2025*
