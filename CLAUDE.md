# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ailo is a **structured prompting framework** designed to optimize communication with AI systems. It's a conceptual framework and documentation project (not a software codebase with build/test commands) that defines a schema for clear, hierarchical AI prompts.

## Core Concepts

### Schema Structure
Ailo uses a hierarchical schema built around:
- **ACT/OBJ** (Action/Object): Core of every prompt
- **TAGS**: Structured metadata (format, length, style, audience, constraints, etc.)
- **CONTEXT**: Optional background/rationale
- **PERSONA**: Optional role/perspective for AI
- **MODE**: Optional task type clarification (Generate, Evaluate, Rewrite, etc.)
- **OUTPUT**: Delivery format specification

### Response Templates
The framework defines five standard response patterns for AI interactions:
- **Confirm**: Accept and proceed with request
- **Reject**: Decline with reason
- **Modify**: Alter existing output
- **Clarify**: Request additional information
- **Deliver**: Provide final output

### JSON Compatibility
The entire schema is JSON-ready for integration with orchestration frameworks (LangChain, Haystack, AutoGPT, etc.)

## Architecture Philosophy

### Level 5 Prompting (Orchestration)
Ailo is designed to support:
- **Prompt chaining**: Output from one prompt becomes input to the next
- **Meta-prompting**: Prompts that design other prompts
- **Agentic workflows**: AI-driven decision trees for prompt sequences
- **Tool integration**: External API/database triggers
- **Automation pipelines**: Multi-step sequences with minimal human intervention

### Why It Works with AI
The schema aligns with how LLMs process information:
- Structured tokens (ACT=Summarize) reduce embedding ambiguity
- CONTEXT/PERSONA provide high-weight anchor tokens for attention mechanisms
- MODE activates specific reasoning paths
- OUTPUT constraints guide final token prediction

## Documentation Structure

The main documentation is in `readme.md` with these sections:
1. Introduction and AI fundamentals (tokenization, embeddings, attention)
2. Core schema definition
3. Tag library (comprehensive metadata options)
4. Response templates
5. Practical examples across domains (writing, research, technical, creative, business)
6. JSON version of schema
7. Context/Persona comparison
8. Prompt chaining patterns
9. Best practices
10. Orchestration and meta-prompting

## Working with This Repository

### Making Documentation Updates
When updating the framework documentation:
- Maintain the structured section numbering (1, 1.1, 1.2, 2, etc.)
- Keep code examples consistent with the schema format
- Preserve both plain-text and JSON examples
- Ensure tag definitions align between Section 3 (Tag Library) and Section 2 (Core Schema)

### Adding Examples
New examples should follow the established pattern:
```
CONTEXT = [Optional background]
PERSONA = [Optional role]
MODE = [Optional task type]
ACT = [Required action]
OBJ = [Required object]
TAGS = [Structured metadata]
OUTPUT = [Delivery format]
```

### Schema Evolution
When proposing schema changes:
- Update Core Schema (Section 2) first
- Ensure Tag Library (Section 3) reflects new options
- Add examples demonstrating the change
- Update JSON version (Section 6) for consistency
- Consider impact on orchestration patterns (Section 10)

## License

MIT License - Copyright (c) 2025 Ergon Copeland
