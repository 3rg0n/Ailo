# Session Context - Prompting Style Research

## Project Background

This project started as **Ailo** — a structured prompting framework using schema-based prompts (ACT/OBJ/TAGS). After testing against other prompting techniques, we pivoted to **empirical research comparing prompting styles**.

The hypothesis was that structured schema prompts would consistently outperform natural language. What we found was more nuanced: **simple techniques often win**.

## Project Goals

1. Compare 9+ prompting techniques across multiple LLM providers
2. Test on budget, mid-tier, and premium models
3. Measure accuracy gain vs token cost (ROI)
4. Provide data-driven recommendations by use case (math, coding, reasoning, etc.)
5. Use ASCII bar charts (like TOON project at C:\Dev\Github\toon) for visualization

## Repository Structure

```
ailo/
├── readme.md                 # Main doc with pivot narrative and key findings
├── .gitignore               # Root gitignore
├── research/
│   ├── FINDINGS.md          # Detailed findings with ASCII charts
│   ├── context.md           # THIS FILE - session context
│   ├── bedrock_client.py    # AWS Bedrock API client
│   ├── multi_provider_client.py  # Unified client (Bedrock, OpenAI, Gemini)
│   ├── unified_benchmark.py      # Cross-provider benchmark runner
│   ├── code_benchmark.py         # Code generation benchmark (4 JS algorithms)
│   ├── agentic_benchmark.py      # ReAct, PAL, Chaining, Reflexion tests
│   ├── .env.example              # API key template
│   └── results/                  # JSON benchmark results
```

## Current State (2025-11-26)

### Completed Benchmarks

**Unified Benchmark (9 prompting styles):**
- Claude Opus 4.5: Few-shot +11.1%, zero-shot 83.3%
- Claude Haiku 4.5: Few-shot +5.6%, -5% tokens
- Gemini 2.0 Flash: Schema/CoT 100%, +11.1%
- Gemini 1.5 Pro: Tested
- Mistral 7B: Directional +22.2%, -10% tokens
- Nova Micro: Few-shot +5.6%, -2% tokens

**Code Generation Benchmark (4 JS algorithms):**
- Nova Micro, Claude Haiku, Mistral 7B, Gemini 2.0 Flash
- Few-shot dominates: 53-58% similarity to reference
- ToT consistently worst (15-35%)

**Agentic Benchmark:**
- Nova Micro, Mistral 7B tested
- PAL saves 59% tokens on math tasks

### Blocked

- **OpenAI API**: Network/firewall blocking Python requests
  - PowerShell Invoke-WebRequest works
  - Python openai library gets connection errors
  - **Fix**: Reboot should resolve

### Key Findings

1. **Few-shot is the clear winner** - Best ROI across all models
2. **Schema/CoT best for Gemini** - +11.1% accuracy
3. **Directional underrated** - Same gains as CoT, fewer tokens
4. **Complex techniques hurt** - ToT/Self-Consistency add overhead without benefit
5. **Premium models don't need fancy prompting** - Zero-shot works fine
6. **Code generation loves few-shot** - 53-58% similarity vs 16-40% for others

## Environment Setup

```bash
# .env file in research/ folder (not committed)
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-key
OPENAI_KEY=your-key      # Blocked by network until reboot
GEMINI_KEY=your-key      # Working
```

## How to Continue

### After Reboot (OpenAI fix)

```bash
cd research

# Test OpenAI connection
python multi_provider_client.py gpt-4o-mini

# Run OpenAI benchmarks
python unified_benchmark.py --model gpt-4o-mini
python unified_benchmark.py --model gpt-4o
python code_benchmark.py --model gpt-4o-mini --output results/code_gpt4o_mini.json
```

### More Gemini Testing

```bash
python unified_benchmark.py --model gemini-1.5-flash
python code_benchmark.py --model gemini-1.5-flash --output results/code_gemini_1.5_flash.json
```

### List Available Models

```bash
python multi_provider_client.py list
```

### Update Documentation

After new benchmarks, update:
- `readme.md` - Add new model results to tables
- `FINDINGS.md` - Add detailed results with ASCII charts

## Git Status

- Branch: main
- 5 commits ahead of origin/main
- Last commit: "Pivot Ailo to prompting styles research project"
- Ready to push: `git push`

## Reference: TOON Visual Style

Use ASCII bar charts like TOON (C:\Dev\Github\toon):

```
few_shot       ████████████████████   Best ROI: +16% acc, -25% tokens
directional    ██████████████████░░   +22% acc, -10% tokens
zero_shot      ████████████████░░░░   Baseline
```

## Reference: Ailo Schema (for "What Happened to Ailo" section)

```
CONTEXT = [Background / why you need this]
PERSONA = [Role for AI: mentor, critic, analyst...]
MODE    = [Task type: Generate, Evaluate, Compare...]
ACT     = [What you want done]
OBJ     = [Subject/object to work on]
TAGS    = [Format, Length, Style, Audience, Constraints...]
OUTPUT  = [Delivery format: text, code, file...]
```

## Prompting Techniques Tested

| Technique | Description | Typical Result |
|-----------|-------------|----------------|
| zero_shot | Plain natural language | Baseline |
| few_shot | 1-2 examples before task | BEST ROI |
| cot | Step-by-step reasoning | +accuracy, +tokens |
| schema | Structured ACT/OBJ/TAGS | Good for Gemini |
| meta | LLM designs approach first | High overhead |
| gen_knowledge | Generate facts, then answer | Moderate |
| directional | Hints/keywords to guide | Underrated |
| tot | Multiple solution paths | Often WORSE |
| self_consistency | Multiple approaches, reconcile | High overhead |
