# Prompting Style Research: Data-Driven Guidance

## Overview

This research compares different prompting techniques across multiple LLM models to provide data-backed guidance on when to use each style. Rather than promoting any single approach, we measure empirical outcomes.

## Prompting Styles Tested

### Fully Tested (9 techniques)

| Style | Description | Token Overhead |
|-------|-------------|----------------|
| **Zero-shot** | Plain natural language prompt | Baseline |
| **Few-shot** | Include 1-2 examples before the task | -25% to +30% |
| **Chain-of-Thought (CoT)** | Explicit step-by-step reasoning | +15% to +68% |
| **Schema** | Structured format (ACT/OBJ/TAGS) | +4% to +86% |
| **Meta Prompting** | LLM designs its own approach first | +46% to +123% |
| **Generate Knowledge** | Generate relevant facts, then answer | +5% to +75% |
| **Directional Stimulus** | Hints/keywords to guide response | -10% to +27% |
| **Tree of Thoughts** | Explore multiple solution paths | +80% to +166% |
| **Self-Consistency** | Multiple approaches, then reconcile | +44% to +211% |

### Agentic Techniques (Now Tested!)

These techniques require tool execution or multi-turn orchestration. We built an agentic test harness with real tools:

| Style | Description | Infrastructure |
|-------|-------------|----------------|
| **ReAct** | Reason + Act loop with tools | Calculator, Search, Python executor |
| **PAL** | Generate and execute Python code | Safe Python sandbox |
| **Prompt Chaining** | Multi-step with output passing | 3-step orchestrator |
| **Reflexion** | Generate, critique, retry | 2-attempt loop |

### Not Testable (require external systems)

| Style | Why Not Testable |
|-------|------------------|
| RAG | Requires vector DB + document corpus |
| ART | Requires external API tools |
| APE | Meta-optimization loop |
| Active-Prompt | Human-in-loop |
| Multimodal CoT | Image input |
| Graph Prompting | Complex graph structure |

## Key Findings

### Finding 1: Few-shot is the Hidden Champion

```
MISTRAL 7B RESULTS (Budget Model)
Style              Pass Rate    vs Zero-shot    Token Overhead
--------------------------------------------------------------
Zero-shot            72.2%       baseline        baseline
Few-shot             88.9%       +16.7%          -24.6%  <-- BEST ROI!
Directional          94.4%       +22.2%          -10.4%
Gen-Knowledge        94.4%       +22.2%          +5.0%
CoT                  94.4%       +22.2%          +14.5%
```

**Insight**: Few-shot prompting improved accuracy AND reduced tokens on budget models. The examples help the model understand the task without verbose instructions.

### Finding 2: Directional Stimulus is Underrated

Adding simple hints/keywords (e.g., "HINTS: Calculate total first, then apply discount") achieves similar improvements to CoT but with far less token overhead.

```
DIRECTIONAL vs COT (Mistral 7B)
Directional: +22.2% improvement, -10.4% tokens
CoT:         +22.2% improvement, +14.5% tokens
```

**Winner**: Directional Stimulus for same accuracy at lower cost.

### Finding 3: Complex Techniques Often Underperform

Tree of Thoughts and Self-Consistency add massive token overhead (80-211%) without corresponding accuracy gains:

```
NOVA MICRO (Budget Model)
Style              Pass Rate    Token Overhead
----------------------------------------------
Zero-shot            94.4%       baseline
Tree of Thoughts     83.3%       +130.8%  <-- WORSE accuracy!
Self-Consistency     94.4%       +118.4%  <-- Same accuracy, 2x tokens
```

**Insight**: Simpler techniques often win. Complex prompting adds cognitive load for the model.

### Finding 4: Best ROI by Model Tier

| Model Tier | Best Technique | Why |
|------------|---------------|-----|
| **Budget** (Mistral 7B) | Few-shot or Directional | +16-22% accuracy, saves tokens |
| **Mid** (Nova Micro) | Schema or Few-shot | +5.6% accuracy, moderate overhead |
| **Premium** (Claude) | Zero-shot | Already 95%+ accurate |

### Finding 5: PAL is a Game-Changer for Math Tasks

Program-Aided Language (PAL) generates and executes code, achieving:

```
NOVA MICRO - AGENTIC BENCHMARK (5 math/code tasks)
Technique      Pass Rate    vs Zero    Tokens    LLM Calls   Tool Calls
-----------------------------------------------------------------------
Zero-shot        100%      baseline      284         1           0
ReAct             80%       -20%         476         1           0
PAL              100%        same        116        1           1  <-- BEST!
Chaining         100%        same       2014         3           0
Reflexion        100%        same       1209         2           0
```

**PAL saves 59% tokens** by generating concise code instead of verbose reasoning. The Python executor does the heavy lifting.

### Finding 6: Agentic Overhead is Often Not Worth It

```
MISTRAL 7B - AGENTIC BENCHMARK
Technique      Pass Rate    Tokens    Overhead
----------------------------------------------
Zero-shot         80%        263      baseline
ReAct             80%        592       +125%   <-- Same accuracy, 2x tokens
Chaining          80%       1413       +438%   <-- Same accuracy, 5x tokens
PAL               40%        146        -44%   <-- Code quality issues
```

For budget models, PAL fails more often because generated code has errors. Reflexion helps by allowing retry.

## Comprehensive Results

### Amazon Nova Micro (Budget)

```
Style              Pass Rate    vs Zero    Tokens    Token Diff
----------------------------------------------------------------
Zero-Shot            94.4%      baseline      323     baseline
Few-Shot            100.0%       +5.6%        317       -1.9%
CoT                  94.4%        same        543      +67.9%
Schema              100.0%       +5.6%        440      +36.2%
Meta                 94.4%        same        686     +112.0%
Gen-Knowledge        94.4%        same        566      +75.1%
Directional         100.0%       +5.6%        410      +26.9%
ToT                  83.3%      -11.1%        746     +130.8%
Self-Consistency     94.4%        same        706     +118.4%

WINNER: Few-shot (+5.6% accuracy, -2% tokens)
```

### Mistral 7B (Budget)

```
Style              Pass Rate    vs Zero    Tokens    Token Diff
----------------------------------------------------------------
Zero-Shot            72.2%      baseline      419     baseline
Few-Shot             88.9%      +16.7%        316      -24.6%
CoT                  94.4%      +22.2%        480      +14.5%
Schema               88.9%      +16.7%        402       -3.9%
Meta                 83.3%      +11.1%        610      +45.8%
Gen-Knowledge        94.4%      +22.2%        440       +5.0%
Directional          94.4%      +22.2%        375      -10.4%
ToT                  77.8%       +5.6%        753      +79.8%
Self-Consistency     88.9%      +16.7%        601      +43.5%

WINNER: Directional (+22.2% accuracy, -10.4% tokens)
RUNNER-UP: Few-shot (+16.7% accuracy, -24.6% tokens)
```

## Recommendations Matrix

| Use Case | Model Tier | Recommended Style | Why |
|----------|------------|-------------------|-----|
| General tasks | Premium | Zero-shot | Already accurate |
| General tasks | Budget | Few-shot | Best ROI |
| Math/Logic | Any | CoT or Directional | Reasoning benefits |
| Format-critical | Any | Schema | Explicit constraints |
| Persona-driven | Budget | Schema or Few-shot | Examples help |
| Token-sensitive | Budget | Few-shot or Directional | Can save tokens |
| Accuracy-critical | Budget | CoT or Gen-Knowledge | Higher accuracy |
| Quick prototyping | Any | Zero-shot | Fastest iteration |

## When to Use Each Style

### Zero-shot (Baseline)
- Premium models that follow instructions well
- Simple, well-defined tasks
- When iterating quickly on prompts
- Token cost is critical

### Few-shot (Best ROI for Budget Models)
- Budget/smaller models
- Tasks where examples clarify the expected format
- When you have good representative examples
- Want accuracy boost without token penalty

### Chain-of-Thought
- Mathematical calculations
- Multi-step logical reasoning
- When showing work is valuable
- Debugging complex problems

### Schema (Structured)
- Multiple constraints must be met
- Specific output format required
- Persona/context is critical
- Complex multi-part tasks

### Directional Stimulus
- You know the expected answer pattern
- Want to guide without full examples
- Lower token overhead than CoT
- Combining with other techniques

### Generate Knowledge
- Tasks requiring factual background
- When model might lack specific knowledge
- Two-phase reasoning needed

### Meta Prompting
- Novel/unusual tasks
- When standard approaches might fail
- Letting model choose its approach

### Tree of Thoughts / Self-Consistency
- High-stakes decisions only
- When token cost is not a concern
- Complex problems benefiting from multiple perspectives
- Generally NOT recommended due to high overhead

## Methodology

### Test Suite (v3)
9 test cases across 5 categories:
- **Writing**: Executive summaries, persona adherence
- **Reasoning**: Math problems, logic puzzles, percentages
- **Creative**: Story ideas generation
- **Analysis**: Framework comparisons, pros/cons
- **Technical**: Code explanations

### Evaluation Criteria
Each response evaluated against 2-3 objective criteria:
- Format compliance (bullets, tables, code blocks)
- Constraint adherence (length, count)
- Content accuracy (correct answers, required elements)

### Models Tested
- **Budget**: Nova Micro, Mistral 7B
- **Mid**: Nova Lite, Claude Haiku 4.5
- **Premium**: Claude Sonnet 4.5, Mistral Large

## Running the Benchmarks

```bash
# Install dependencies
pip install -r requirements.txt

# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Run 3-style benchmark (quick)
python multi_style_benchmark.py --model nova-micro

# Run comprehensive 9-style benchmark
python multi_style_benchmark_v2.py --model mistral-7b

# Generate report
python generate_report.py
```

## Repository Structure

```
research/
├── bedrock_client.py           # AWS Bedrock API client
├── test_prompts_v2.py          # 3-style test cases (Zero/Schema/CoT)
├── test_prompts_v3.py          # 9-style comprehensive test cases
├── multi_style_benchmark.py    # 3-style benchmark runner
├── multi_style_benchmark_v2.py # 9-style comprehensive runner
├── generate_report.py          # Report generator
├── results/                    # Benchmark results (JSON)
└── FINDINGS.md                 # This document
```

## Limitations

1. **Sample Size**: 9 test cases per comprehensive run
2. **Single-call Simulation**: ToT and Self-Consistency simulated in single prompt
3. **Model Versions**: Results may vary with model updates
4. **Task Types**: General tasks; specialized domains may differ
5. **Evaluation**: Automated criteria; subjective quality not measured

## Key Takeaways

1. **Simple often wins**: Few-shot and Directional beat complex techniques
2. **ROI matters**: Consider accuracy gain vs token cost
3. **Model tier matters**: Budget models benefit most from prompting techniques
4. **Premium models are robust**: Zero-shot works fine for Claude/GPT-4
5. **Overhead adds up**: ToT/Self-Consistency rarely justify their cost

## License

MIT - Use this research and tooling freely.

---

*Generated from benchmark data collected November 2025*
*9 prompting techniques tested across multiple AWS Bedrock models*
