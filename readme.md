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

> **Methodology Note**: Results below use dual evaluation: deterministic checks + LLM-as-judge (Opus 4.5) scoring correctness, completeness, clarity, and relevance. Combined score = 60% deterministic + 40% LLM judge.

### Combined Score by Style (Average Across 6 Models)

```
Style              Combined Score    Tokens    Key Insight
──────────────────────────────────────────────────────────────────────
schema             94.3%             416       Best overall, clear constraints
cot                93.7%             532       Strong reasoning, higher cost
directional        92.5%             421       Good balance accuracy/tokens
few_shot           91.5%             302       MOST EFFICIENT: -12% tokens
gen_knowledge      91.9%             533       Moderate gains, extra tokens
zero_shot          93.1%             325       Baseline: surprisingly strong
meta               87.4%             658       Often over-complicates
tot                80.2%           4,806       TRUE MULTI-TURN: Not worth cost
self_consistency   81.8%           5,491       TRUE MULTI-TURN: 17x tokens, poor ROI
```

> **V3 Update**: ToT and Self-Consistency now use TRUE multi-turn conversations (4 API calls each) instead of single-call simulations. This increased tokens 7-17x but reduced quality scores due to context drift.

### Winner by Model Tier (V2 with LLM Judge)

```
BUDGET MODELS (Mistral 7B, Nova Micro)
──────────────────────────────────────────────────────────────────────────
Style          Combined    Correctness  Clarity    Tokens    Verdict
──────────────────────────────────────────────────────────────────────────
cot              95.9%       0.98        0.94       453      WINNER: Best quality
schema           94.1%       0.98        0.99       416      High clarity
zero_shot        93.1%       0.96        0.99       325      Strong baseline
few_shot         85.3%*      0.90        0.93       302      *Can hurt small models

*Note: Mistral 7B few_shot dropped to 87.5% accuracy vs 100% for other styles

MID-TIER MODELS (Claude Haiku 4.5)
──────────────────────────────────────────────────────────────────────────
zero_shot        93.5%       1.00        1.00       300      WINNER: Perfect clarity
cot              92.9%       0.99        0.99       554      Great reasoning
schema           92.8%       0.99        0.99       480      Format control
self_consistency 88.0%       0.88        0.95       479      Higher cost, less gain

PREMIUM MODELS (Claude Sonnet 4.5, Mistral Large)
──────────────────────────────────────────────────────────────────────────
zero_shot        95.4%       0.98        0.99       291      WINNER: Excellent baseline
cot              94.5%       0.96        0.95       530      Marginal improvement
schema           93.3%       0.99        0.99       522      Format precision
tot              88.9%       0.94        0.93       745      Overkill for these models
```

### Prompt Style by Use Case

| Use Case | Recommended Style | Why | Token Impact |
|----------|-------------------|-----|--------------|
| **Code Generation** | `few_shot` | Examples demonstrate structure/style | -8% tokens |
| **Documentation** | `schema` | Highest clarity (0.99), format control | +28% tokens |
| **Math/Logic** | `cot` | Step-by-step reasoning, 95.9% combined | +63% tokens |
| **Agents/Agentic** | `zero_shot` + tools | Let tools handle complexity | Baseline |
| **Data Analysis** | `cot` or `gen_knowledge` | Reasoning + domain context | +52-63% |
| **Creative Writing** | `directional` | Hints guide without constraining | +35% tokens |
| **API/Integration** | `schema` | Structured output, predictable format | +28% tokens |
| **Quick Prototyping** | `zero_shot` | Fast iteration, 93.1% baseline | Baseline |

### Detailed Use Case Guide

**Code Generation**
```
Best:  few_shot (show 1-2 examples of desired code style)
Why:   Models learn naming conventions and structure from examples
Avoid: self_consistency (adds tokens, no accuracy gain)
```

**Documentation Writing**
```
Best:  schema (ACT=Write OBJ=Documentation TAGS=[Format:Markdown])
Why:   Explicit format constraints ensure consistent output
Alt:   directional (provide outline hints)
```

**Math & Logic Problems**
```
Best:  cot (Chain-of-Thought)
Why:   Step-by-step reasoning catches errors, 95.9% combined
Alt:   gen_knowledge (recall formulas first, then solve)
```

**Agentic Workflows**
```
Best:  zero_shot + tool descriptions
Why:   Let tools handle complexity; prompts should be simple triggers
Note:  Complex prompting often interferes with tool selection
```

**Data Analysis**
```
Best:  cot (for reasoning through data)
Alt:   gen_knowledge (recall statistical concepts first)
Why:   Explicit reasoning prevents calculation errors
```

### Quick Reference

```
USE CASE           RECOMMENDED          WHY
─────────────────────────────────────────────────────────────────────────
Code generation    few_shot             Examples > explanations
Documentation      schema               Format control, high clarity
Math/Logic         cot                  95.9% combined, best reasoning
Agents/Agentic     zero_shot            Keep prompts simple, let tools work
Data analysis      cot / gen_knowledge  Reasoning prevents errors
Creative           directional          Hints without constraints
Budget models      cot > schema         Step-by-step helps smaller models
Premium models     zero_shot            Already excellent, save tokens
Token-sensitive    few_shot             302 avg tokens (lowest)
Avoid              self_consistency     78.2%, worst ROI
```

### Six Takeaways (V3 Results with TRUE Multi-Turn)

1. **Zero-shot is better than expected** — 93.1% combined score, works great on modern models
2. **CoT wins for reasoning** — Especially on budget models (95.9% combined)
3. **TRUE multi-turn ToT/SC is NOT worth it** — V3 tested real 4-turn conversations: tokens increased 7-17x (4,806-5,491 avg) but quality DROPPED due to context drift and repetition
4. **Schema provides clarity** — Highest clarity scores (0.99) across models
5. **Few-shot can backfire** — On small models like Mistral 7B, accuracy dropped 12.5%
6. **Multi-turn hurts small models most** — Mistral 7B accuracy dropped to 75% with TRUE multi-turn ToT/SC (vs 100% single-call)

---

## Results by Model

### Model Tiers Tested

| Tier | Models | Cost (per 1K tokens) |
|------|--------|---------------------|
| **Budget** | Nova Micro, Mistral 7B, GPT-4o-mini | $0.00004 - $0.00015 |
| **Mid** | Claude Haiku 4.5, Nova Lite | $0.0006 - $0.001 |
| **Premium** | Claude Sonnet 4.5, Mistral Large, GPT-4o | $0.003 - $0.015 |

---

### V2 Results with LLM Judge (Opus 4.5)

> All models tested on 8 prompts × 9 styles = 72 evaluations per model.
> Scores combine deterministic checks (60%) + LLM judge scores (40%).

### Claude Sonnet 4.5 (Premium)

```
Style              Combined   Accuracy   Correctness  Completeness  Clarity   Tokens
───────────────────────────────────────────────────────────────────────────────────
cot                  93.9%     100.0%       0.99         0.97        0.98       630   BEST
schema               94.3%     100.0%       0.99         0.99        0.99       522
zero_shot            91.7%     100.0%       0.99         1.00        1.00       282   Baseline
few_shot             89.5%     100.0%       0.99         0.96        1.00       276
directional          91.4%     100.0%       0.99         0.96        1.00       349
meta                 89.9%     100.0%       0.99         0.94        0.99       688
tot                  88.4%     100.0%       0.99         0.95        0.94       673
self_consistency     70.0%     100.0%       0.99         0.94        1.00       411   WORST
```

**Key insight**: 100% accuracy across all styles. CoT provides best combined quality but zero-shot is already excellent.

### GPT-4o-mini (Budget)

```
Style              Combined   Accuracy   Correctness  Completeness  Clarity   Tokens
───────────────────────────────────────────────────────────────────────────────────
gen_knowledge        97.0%     100.0%       0.99         0.95        0.96       450   BEST
zero_shot            94.0%     100.0%       1.00         0.99        1.00       296
cot                  94.0%     100.0%       0.97         0.94        0.97       471
schema               94.0%     100.0%       0.99         0.97        0.99       400
tot                  93.0%     100.0%       0.96         0.88        0.90       701
few_shot             92.0%     100.0%       0.99         0.99        1.00       271   EFFICIENT
self_consistency     92.0%     100.0%       0.95         0.89        0.92       702
directional          90.0%     100.0%       0.96         0.89        0.97       400
meta                 87.0%     100.0%       0.85         0.74        0.85       625   WORST
```

**Key insight**: 100% accuracy across all styles. `gen_knowledge` achieves highest combined score. `few_shot` saves tokens.

### Claude Haiku 4.5 (Mid)

```
Style              Combined   Accuracy   Correctness  Completeness  Clarity   Tokens
───────────────────────────────────────────────────────────────────────────────────
zero_shot            93.5%     100.0%       1.00         1.00        1.00       300   WINNER
cot                  92.9%     100.0%       0.99         1.00        0.99       554
schema               92.8%     100.0%       0.99         0.99        0.99       480
few_shot             91.6%     100.0%       0.99         1.00        1.00       291
directional          90.7%     100.0%       1.00         0.99        0.98       353
gen_knowledge        91.3%     100.0%       0.99         1.00        0.99       471
meta                 84.9%     100.0%       0.98         0.99        0.96       622
tot                  91.0%     100.0%       1.00         0.98        0.95       567
self_consistency     88.0%     100.0%       0.88         0.95        0.95       479
```

**Key insight**: Perfect accuracy across all styles. Zero-shot achieves perfect clarity and completeness scores.

### Gemini 2.0 Flash (Mid-Premium)

```
Style              Combined   Accuracy   Correctness  Completeness  Clarity   Tokens
───────────────────────────────────────────────────────────────────────────────────
few_shot             92.7%     100.0%       1.00         0.96        0.99       241   BEST ROI
zero_shot            91.0%     100.0%       1.00         0.96        0.99       495
cot                  90.6%     100.0%       0.91         0.83        0.91       637
schema               91.0%     100.0%       0.99         0.97        0.98       439
directional          88.5%      87.5%       0.99         0.89        0.95       445
meta                 81.9%      87.5%       0.96         0.83        0.94       637
tot                  88.9%     100.0%       0.96         0.91        0.96       745
self_consistency     58.0%     100.0%       0.91         0.73        0.94       634   WORST
```

**Key insight**: Few-shot wins with fewest tokens. Self-consistency has worst combined score.

### Mistral 7B (Budget)

```
Style              Combined   Accuracy   Correctness  Completeness  Clarity   Tokens
───────────────────────────────────────────────────────────────────────────────────
cot                  95.9%     100.0%       0.98         0.95        0.95       453   WINNER
zero_shot            94.6%     100.0%       0.96         0.97        0.99       320
schema               92.0%     100.0%       0.98         0.97        0.98       438
directional          92.2%     100.0%       0.96         0.94        0.94       423
few_shot             80.3%      87.5%       0.82         0.91        0.85       320   CAUTION
gen_knowledge        87.6%      87.5%       0.91         0.87        0.94       494
meta                 83.7%     100.0%       0.93         0.79        0.96       563
tot                  89.6%     100.0%       0.92         0.88        0.91       702
self_consistency     68.0%     100.0%       0.84         0.87        0.95       894
```

**Key insight**: CoT provides best quality. Few-shot HURTS accuracy on this smaller model (87.5% vs 100%).

### Mistral Large (Premium)

```
Style              Combined   Accuracy   Correctness  Completeness  Clarity   Tokens
───────────────────────────────────────────────────────────────────────────────────
zero_shot            95.4%     100.0%       0.97         0.98        0.99       300   WINNER
cot                  95.0%     100.0%       0.93         0.91        0.93       429
few_shot             93.0%     100.0%       0.99         0.98        1.00       290
schema               92.6%     100.0%       0.99         0.96        0.98       398
directional          93.9%     100.0%       0.98         0.96        0.99       394
gen_knowledge        94.7%     100.0%       0.96         0.96        0.96       565
meta                 92.7%     100.0%       0.99         0.91        0.95       543
tot                  89.4%     100.0%       0.96         0.91        0.93       691
self_consistency     58.0%     100.0%       0.91         0.87        0.95       635   WORST
```

**Key insight**: Zero-shot is best. Self-consistency provides worst combined score despite 100% accuracy.

### Amazon Nova Micro (Budget)

```
Style              Combined   Accuracy   Correctness  Completeness  Clarity   Tokens
───────────────────────────────────────────────────────────────────────────────────
schema               94.1%     100.0%       0.98         0.98        0.99       440   WINNER
zero_shot            92.2%     100.0%       0.98         0.96        0.99       325
cot                  91.8%     100.0%       0.96         0.92        0.95       543
few_shot             91.5%     100.0%       0.98         0.95        0.99       317
directional          92.3%     100.0%       0.98         0.96        0.98       410
gen_knowledge        90.9%     100.0%       0.96         0.93        0.95       566
meta                 88.8%     100.0%       0.96         0.91        0.95       686
tot                  87.9%     100.0%       0.96         0.90        0.91       746
self_consistency     90.9%     100.0%       0.93         0.94        0.94       706
```

**Key insight**: Schema provides highest combined score. All styles achieve 100% accuracy.

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

## V3 TRUE Multi-Turn Results

V3 updated Tree of Thoughts (ToT) and Self-Consistency to use **real multi-turn conversations** instead of single-call simulations. Each technique now makes 4 separate API calls with conversation history.

### Multi-Turn Implementation

```
ToT (Tree of Thoughts) - 4 Turns:
  Turn 1: "Solve using Path A (direct approach)"
  Turn 2: "Now solve using Path B (alternative method)"
  Turn 3: "Now solve using Path C (verification/estimation)"
  Turn 4: "Evaluate all paths and give FINAL ANSWER"

Self-Consistency - 4 Turns:
  Turn 1: "Solve using Method 1 (standard calculation)"
  Turn 2: "Solve using Method 2 (alternative approach)"
  Turn 3: "Solve using Method 3 (verification/cross-check)"
  Turn 4: "Compare all methods and reconcile to FINAL ANSWER"
```

### Multi-Turn Results by Model

```
Model              ToT Tokens   ToT Score   SC Tokens   SC Score   Accuracy
─────────────────────────────────────────────────────────────────────────────
Nova Micro           5,376       82.7%        5,576       81.6%      100%
Claude Haiku 4.5     4,756       86.2%        6,202       87.0%      100%
GPT-4o-mini          4,705       83.4%        5,152       84.6%      100%
Gemini 2.0 Flash     4,725       80.6%        5,810       82.3%      100%
Mistral Large        3,855       72.4%        4,407       80.9%      100%
Claude Sonnet 4.5    5,533       88.7%        6,963       88.9%      100%
Mistral 7B           3,693       67.2%        4,329       69.4%       75%   ⚠️
```

### Why Multi-Turn Underperforms

1. **Context Drift**: Models lose focus across turns, repeating earlier reasoning instead of building on it
2. **Token Explosion**: 7-17x more tokens than single-call approaches for similar accuracy
3. **Clarity Degradation**: Clarity scores dropped to 0.50-0.87 (vs 0.98+ for single-call styles)
4. **Small Model Failure**: Mistral 7B accuracy dropped from 100% (single-call) to 75% (multi-turn)

### Recommendation

**Avoid TRUE multi-turn ToT/Self-Consistency for most use cases.** Single-call CoT (93.7% combined, 532 tokens) outperforms multi-turn ToT (80.2% combined, 4,806 tokens) at 1/9th the cost.

Use multi-turn only when:
- You need explicit exploration of multiple solution paths for auditing
- Token cost is not a concern
- Using premium models (Claude Sonnet 4.5 maintained 88.9% quality)

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

| Technique | Description | Token Overhead | API Calls |
|-----------|-------------|----------------|-----------|
| **zero_shot** | Plain natural language | Baseline | 1 |
| **few_shot** | 1-2 examples before task | -25% to +30% | 1 |
| **cot** | Step-by-step reasoning | +15% to +68% | 1 |
| **schema** | Structured ACT/OBJ/TAGS | +4% to +86% | 1 |
| **meta** | LLM designs approach first | +46% to +123% | 1 |
| **gen_knowledge** | Generate facts, then answer | +5% to +75% | 1 |
| **directional** | Hints/keywords to guide | -10% to +27% | 1 |
| **tot** | TRUE multi-turn: 3 paths + synthesis | +1,100% to +1,700% | 4 |
| **self_consistency** | TRUE multi-turn: 3 methods + reconcile | +1,200% to +2,000% | 4 |

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
| **ToT** | 4 turns: Path A → Path B → Path C → Synthesis (TRUE multi-turn) |
| **Self-Consistency** | 4 turns: Method 1 → Method 2 → Method 3 → Reconcile (TRUE multi-turn) |

### Evaluation Criteria (V2)

The V2 benchmark uses **dual evaluation**:

**1. Deterministic Evaluation (60% weight)**
```
EVAL TYPES
├── numeric     - Extract numbers, compare with tolerance (±0.01)
├── contains    - Check if expected answer appears in response
├── keywords    - Count required keywords found (threshold 0.5)
├── exact       - Normalized string match
├── fuzzy       - Substring match + F1 token overlap
└── code_exec   - Lint + execute against test cases
```

**2. LLM-as-Judge (40% weight) - Opus 4.5**

| Dimension | Question Asked | What It Measures |
|-----------|----------------|------------------|
| **Correctness** | Is the answer factually correct? | Did the model arrive at the right answer? For math, is the number right? For logic, is the conclusion valid? |
| **Completeness** | Does it fully address the task? | Are all parts of the question answered? Nothing missing or skipped? |
| **Clarity** | Is it well-organized and clear? | Easy to follow? Good structure? No rambling or disjointed reasoning? |
| **Relevance** | Does it stay on topic? | No tangents, unnecessary content, or off-topic information? |

Each dimension scored 0.0 to 1.0. The combined LLM judge score averages all four.

> **Why multi-turn hurt clarity**: Multi-turn ToT/SC responses scored 0.50-0.70 on clarity because the final synthesis often referenced "Path A" without restating conclusions, repeated earlier reasoning, or produced disjointed summaries assuming context from prior turns. Single-call techniques kept everything in one coherent response, scoring 0.95-1.0.

### Test Suite (V2)

**Unified Benchmark** (8 test cases × 9 styles = 72 per model):
- Math: Discount calculation, percentage calculation
- Logic: Pet logic puzzle
- Writing: Executive summary
- Analysis: Framework comparison, pros/cons
- Technical: Code explanation
- Creative: Story ideas

**Code Benchmark** (6 algorithms):
- Factorial, Fibonacci, GCD, Primality, Reverse String, Two Sum
- Execution-based evaluation (syntax + test cases)

### Limitations

1. **Sample Size**: 8 test cases per comprehensive run
2. **Multi-Turn Context Drift**: TRUE multi-turn ToT/SC suffer from context drift, which may partially explain quality degradation
3. **Model Versions**: Results may vary with model updates
4. **LLM Judge Bias**: Opus 4.5 may favor certain styles (mitigated by 60/40 weighting)

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

# Run V2 benchmark WITHOUT LLM judge (faster, cheaper)
python unified_benchmark_v2.py --model gemini-2.0-flash

# Run V2 benchmark WITH LLM judge (recommended for quality)
python unified_benchmark_v2.py --model claude-sonnet --llm-judge

# Run specific styles only
python unified_benchmark_v2.py --model nova-micro --styles zero_shot few_shot cot

# Run code generation benchmark with execution
python code_benchmark_v2.py --model gpt-4o-mini

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
├── README.md                      # This file
├── research/
│   ├── evaluation.py              # Unified evaluation module (V2)
│   ├── multi_provider_client.py   # Unified client (Bedrock, OpenAI, Gemini)
│   ├── unified_benchmark_v2.py    # Main benchmark runner with LLM judge
│   ├── code_benchmark_v2.py       # Code generation with execution
│   ├── test_prompts_v4.py         # Multi-style prompt definitions
│   └── results/                   # Raw benchmark data (JSON)
│       ├── unified_v2_*.json      # V2 benchmark results
│       └── code_v2_*.json         # Code benchmark results
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

### What We Learned (V2 Update)

Schema prompting (tested as "schema" style in our benchmarks) performs well for:
- **Best combined quality** — 94.3% combined score (highest average across all models)
- **Clarity optimization** — Consistently achieves 0.99 clarity scores
- **Budget models like Nova Micro** — 94.1% combined, outperforming zero-shot

But the V2 results show key nuances:
- **Zero-shot is better than expected** — 93.1% combined, works great on modern models
- **CoT wins for reasoning** — 95.9% combined on budget models
- **Self-consistency is wasteful** — 78.2% combined, worst ROI across all styles
- **Few-shot can hurt small models** — Mistral 7B accuracy dropped 12.5% with few-shot

The research shows prompting style choice depends on model tier and task type.

---

## License

MIT License — Use this research freely.

---

*V3: Empirical prompting research with TRUE multi-turn and LLM-as-Judge evaluation, November 2025*
