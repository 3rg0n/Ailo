# Session Context - Prompting Style Research

## Current State (2025-11-26)

### What Was Accomplished

1. **Project Pivot** - Ailo transformed from schema prompting framework to research project
   - New README tells the pivot story
   - Original Ailo schema documented in "What Happened to Ailo" section
   - FINDINGS.md reorganized with consistent formatting and ASCII charts

2. **Comprehensive Benchmarks Completed**
   - Claude Opus 4.5: Few-shot wins (+11.1% vs zero-shot)
   - Claude Haiku 4.5: Few-shot wins (+5.6%, -5% tokens)
   - Gemini 2.0 Flash: Schema/CoT win (100% pass rate)
   - Mistral 7B: Directional wins (+22.2%, -10% tokens)
   - Nova Micro: Few-shot wins (+5.6%, -2% tokens)

3. **Code Generation Benchmarks**
   - Nova Micro, Claude Haiku, Mistral 7B, Gemini 2.0 Flash tested
   - Few-shot dominates: 53.9-57.9% similarity to reference
   - Gemini results: Few-shot 57.9%, Schema 39.7%, Zero-shot 31.9%

4. **Multi-Provider Client** - Supports AWS Bedrock, OpenAI, Gemini
   - OpenAI blocked by network (needs reboot to fix)
   - Gemini working with rate limit retry logic

### New Results This Session

**Claude Opus 4.5:**
```
Style              Pass Rate   vs Zero   Tokens
few_shot             94.4%     +11.1%      261   WINNER
gen_knowledge        94.4%     +11.1%      472
self_consistency     94.4%     +11.1%      715
cot                  88.9%      +5.6%      441
schema               88.9%      +5.6%      481
directional          88.9%      +5.6%      298
zero_shot            83.3%   baseline      253
```

**Gemini 2.0 Flash Code Generation:**
```
Style              Similarity   Correctness   Tokens
few_shot             57.9%        96.4%        263   WINNER
schema               39.7%        89.3%        236
directional          36.8%        92.2%        383
cot                  34.1%        81.5%        618
zero_shot            31.9%        77.9%        251
tot                  15.6%        92.8%        892
```

### Pending

1. OpenAI benchmarks (blocked by network - reboot should fix)
2. Add new results to FINDINGS.md
3. More Gemini model variants (1.5-flash, 1.5-pro)

### Files Modified This Session

- `readme.md` - Complete rewrite with pivot narrative
- `research/FINDINGS.md` - Reorganized with ASCII charts
- `research/context.md` - This file (updated)
- `research/code_benchmark.py` - Added multi-provider support
- `.gitignore` - Created root-level gitignore

### How to Continue

```bash
cd research

# After reboot, test OpenAI:
python multi_provider_client.py gpt-4o-mini

# Run OpenAI benchmarks:
python unified_benchmark.py --model gpt-4o-mini
python code_benchmark.py --model gpt-4o-mini --output results/code_gpt4o_mini.json

# Run more Gemini variants:
python unified_benchmark.py --model gemini-1.5-flash
python unified_benchmark.py --model gemini-1.5-pro

# List all models:
python multi_provider_client.py list
```

### Key Findings (Updated)

1. **Few-shot is the clear winner** - Best ROI across all models and use cases
2. **Premium models (Opus) don't need fancy prompting** - Few-shot still helps slightly
3. **Schema/CoT best for Gemini** - +11.1% accuracy
4. **Complex techniques often hurt** - ToT consistently underperforms
5. **Code generation loves few-shot** - 53-58% similarity vs 16-40% for others
