# Prompting Styles Research

**An empirical comparison of prompting techniques across LLM models and use cases.**

---

## The Story

This project started as **Ailo** — a structured prompting framework using schema-based prompts (ACT/OBJ/TAGS) to optimize AI communication. The hypothesis was that structured prompts would consistently outperform natural language.

Then we tested it.

What we found was more nuanced: **schema prompting helps in some cases, but simpler techniques often win**. Few-shot examples beat complex reasoning chains. Premium models don't need elaborate prompts. Token overhead from fancy techniques rarely pays off.

So we pivoted. Instead of promoting one prompting style, we built a **research framework** to answer: *"Which prompting technique should I use for my model and use case?"*

This repository now contains:
- Benchmark tooling for 9+ prompting techniques
- Results across budget, mid-tier, and premium models
- Data-driven recommendations by use case

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

### By Use Case

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

---

## Results by Model

### Gemini 2.0 Flash

```
Style              Pass Rate   vs Zero   Tokens
────────────────────────────────────────────────
cot                  100.0%    +11.1%      697
schema               100.0%    +11.1%      414    WINNER (fewer tokens)
few_shot              94.4%     +5.5%      336
directional           94.4%     +5.5%      506
zero_shot             88.9%   baseline     675
meta                  88.9%     +0.0%     1038
gen_knowledge         83.3%     -5.6%      692
tot                   83.3%     -5.6%     1087
self_consistency      83.3%     -5.6%     1062
```

### Claude Haiku 4.5

```
Style              Pass Rate   vs Zero   Token Diff
────────────────────────────────────────────────────
few_shot             94.4%     +5.6%       -5.1%    WINNER
schema               94.4%     +5.6%      +53.4%
zero_shot            88.9%   baseline    baseline
cot                  88.9%      same      +78.9%
gen_knowledge        88.9%      same      +62.3%
directional          83.3%     -5.6%      +21.3%
tot                  83.3%     -5.6%     +132.7%
meta                 77.8%    -11.1%     +162.4%
self_consistency     77.8%    -11.1%     +142.7%
```

### Mistral 7B

```
Style              Pass Rate   vs Zero   Token Diff
────────────────────────────────────────────────────
directional          94.4%    +22.2%      -10.4%    WINNER
cot                  94.4%    +22.2%      +14.5%
gen_knowledge        94.4%    +22.2%       +5.0%
few_shot             88.9%    +16.7%      -24.6%    Best ROI
schema               88.9%    +16.7%       -3.9%
self_consistency     88.9%    +16.7%      +43.5%
meta                 83.3%    +11.1%      +45.8%
tot                  77.8%     +5.6%      +79.8%
zero_shot            72.2%   baseline    baseline
```

### Amazon Nova Micro

```
Style              Pass Rate   vs Zero   Token Diff
────────────────────────────────────────────────────
few_shot            100.0%     +5.6%       -1.9%    WINNER
schema              100.0%     +5.6%      +36.2%
directional         100.0%     +5.6%      +26.9%
zero_shot            94.4%   baseline    baseline
cot                  94.4%      same      +67.9%
meta                 94.4%      same     +112.0%
gen_knowledge        94.4%      same      +75.1%
self_consistency     94.4%      same     +118.4%
tot                  83.3%    -11.1%     +130.8%    WORSE!
```

---

## Code Generation Results

Tested on 4 JavaScript algorithms (factorial, fibonacci, GCD, primality) measuring similarity to reference implementations:

```
Style              Similarity   Correctness   Tokens
────────────────────────────────────────────────────
few_shot             53.9%        89.1%        315    WINNER
zero_shot            41.1%        84.5%        227
schema               37.7%        90.3%        270
tot                  35.3%        75.8%       1211
cot                  29.0%        72.8%        646
self_consistency     21.7%        71.4%        943
directional          24.7%        76.4%        514
meta                 18.9%        64.5%        735
gen_knowledge        16.7%        56.1%        604
```

**Why few-shot wins for code:**
1. Examples demonstrate expected style and structure
2. Models learn naming conventions from examples
3. Avoids verbose explanations that dilute output
4. Lower token overhead than reasoning techniques

---

## Prompting Techniques Tested

| Technique | Description | Typical Overhead |
|-----------|-------------|------------------|
| **zero_shot** | Plain natural language | Baseline |
| **few_shot** | 1-2 examples before task | -25% to +30% |
| **cot** | Step-by-step reasoning | +15% to +68% |
| **schema** | Structured ACT/OBJ/TAGS | +4% to +86% |
| **meta** | LLM designs approach first | +46% to +123% |
| **gen_knowledge** | Generate facts, then answer | +5% to +75% |
| **directional** | Hints/keywords to guide | -10% to +27% |
| **tot** | Multiple solution paths | +80% to +166% |
| **self_consistency** | Multiple approaches, reconcile | +44% to +211% |

### Agentic Techniques (with tools)

| Technique | Description | Best For |
|-----------|-------------|----------|
| **ReAct** | Reason + Act loop | Tool-heavy tasks |
| **PAL** | Generate & execute code | Math (saves 59% tokens) |
| **Chaining** | Multi-step orchestration | Complex workflows |
| **Reflexion** | Generate, critique, retry | Error recovery |

---

## Takeaways

1. **Simple often wins** — Few-shot and directional beat complex techniques
2. **ROI matters** — Consider accuracy gain vs token cost
3. **Model tier matters** — Budget models benefit most from prompting techniques
4. **Premium models are robust** — Zero-shot works fine for Claude/GPT-4
5. **Complex techniques rarely justify cost** — ToT/Self-Consistency add overhead without gains

---

## Repository Structure

```
ailo/
├── readme.md              # This file
├── research/
│   ├── FINDINGS.md        # Detailed findings and methodology
│   ├── bedrock_client.py  # AWS Bedrock API client
│   ├── multi_provider_client.py  # OpenAI, Gemini, Bedrock
│   ├── unified_benchmark.py      # Cross-provider benchmark
│   ├── code_benchmark.py         # Code generation tests
│   ├── agentic_benchmark.py      # ReAct, PAL, Chaining tests
│   └── results/           # Raw benchmark data (JSON)
```

## Running Benchmarks

```bash
cd research

# Test a model
python multi_provider_client.py claude-haiku

# Run comprehensive benchmark
python unified_benchmark.py --model gemini-2.0-flash

# Run code generation benchmark
python code_benchmark.py --model gpt-4o-mini --output results/code_gpt4o.json

# List available models
python multi_provider_client.py list
```

### Supported Models

| Provider | Models |
|----------|--------|
| AWS Bedrock | Claude (Haiku, Sonnet, Opus), Nova (Micro, Lite), Mistral (7B, Small, Large), Llama |
| OpenAI | GPT-4o, GPT-4o-mini, GPT-3.5-turbo, o1-mini |
| Google | Gemini 2.0 Flash, 1.5 Flash, 1.5 Pro |

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
