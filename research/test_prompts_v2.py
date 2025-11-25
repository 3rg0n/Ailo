"""
Test prompts comparing multiple prompting styles:
- Zero-shot (plain natural language)
- Schema (Ailo structured format)
- Chain-of-Thought (step-by-step reasoning)
- Few-shot (with examples)

Each test case measures the same task across different prompting techniques.
"""

from dataclasses import dataclass
from typing import Callable, Optional
from enum import Enum


class PromptStyle(Enum):
    ZERO_SHOT = "zero_shot"      # Plain natural language
    SCHEMA = "schema"            # Ailo structured format
    COT = "cot"                  # Chain-of-thought
    FEW_SHOT = "few_shot"        # With examples


class TaskCategory(Enum):
    WRITING = "writing"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    TECHNICAL = "technical"


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating responses."""
    name: str
    description: str
    check: Callable[[str], bool]


@dataclass
class MultiStylePrompt:
    """A test case with multiple prompting style variants."""
    id: str
    name: str
    category: TaskCategory
    description: str

    # Different prompt styles for the same task
    zero_shot: str
    schema: str
    cot: str
    few_shot: Optional[str] = None

    # Evaluation
    criteria: list[EvaluationCriteria] = None

    def __post_init__(self):
        if self.criteria is None:
            self.criteria = []


# ============================================================================
# Evaluation Helper Functions
# ============================================================================

def has_bullet_points(response: str) -> bool:
    bullet_markers = ["•", "-", "*", "1.", "2.", "3."]
    lines = response.strip().split("\n")
    bullet_count = sum(1 for line in lines if any(line.strip().startswith(m) for m in bullet_markers))
    return bullet_count >= 3


def has_numbered_list(response: str) -> bool:
    import re
    pattern = r'^\s*\d+[\.\)]\s'
    lines = response.strip().split("\n")
    numbered_count = sum(1 for line in lines if re.match(pattern, line))
    return numbered_count >= 3


def has_table(response: str) -> bool:
    return "|" in response and "---" in response


def has_code_block(response: str) -> bool:
    return "```" in response or response.count("    ") >= 3


def word_count_range(min_words: int, max_words: int) -> Callable[[str], bool]:
    def check(response: str) -> bool:
        words = len(response.split())
        return min_words <= words <= max_words
    return check


def is_concise(response: str, max_words: int = 150) -> bool:
    return len(response.split()) <= max_words


def has_step_by_step(response: str) -> bool:
    """Check if response shows step-by-step reasoning."""
    markers = ["step 1", "step 2", "first,", "second,", "then,", "finally,", "1.", "2.", "3."]
    response_lower = response.lower()
    return sum(1 for m in markers if m in response_lower) >= 2


def contains_answer(keywords: list[str]) -> Callable[[str], bool]:
    """Check if response contains expected answer keywords."""
    def check(response: str) -> bool:
        response_lower = response.lower()
        return any(kw.lower() in response_lower for kw in keywords)
    return check


def math_answer_correct(expected: str) -> Callable[[str], bool]:
    """Check if response contains the correct numerical answer."""
    def check(response: str) -> bool:
        return expected in response
    return check


# ============================================================================
# Multi-Style Test Prompts
# ============================================================================

MULTI_STYLE_PROMPTS = [
    # -------------------------------------------------------------------------
    # WRITING TASKS
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="write-01",
        name="Executive Summary",
        category=TaskCategory.WRITING,
        description="Summarize benefits for executive audience",
        zero_shot="""Summarize the benefits of cloud computing for a business executive.
Keep it short and use bullet points.""",
        schema="""CONTEXT = A CEO needs a quick overview before a board meeting
PERSONA = You are a technology consultant briefing a non-technical executive
MODE = Summarize
ACT = Summarize
OBJ = Benefits of cloud computing for enterprise
TAGS = [Format:Bullets, Length:5, Audience:Executive, Style:Professional, Constraints:No technical jargon]
OUTPUT = Text with bullet points""",
        cot="""I need you to summarize the benefits of cloud computing for a business executive.

Before writing your response, think through:
1. Who is the audience? (busy executive, non-technical)
2. What format works best? (bullet points for quick scanning)
3. What aspects matter most to a CEO? (cost, efficiency, competitive advantage)

Now provide a concise bullet-point summary.""",
        criteria=[
            EvaluationCriteria("has_bullets", "Response uses bullet points", has_bullet_points),
            EvaluationCriteria("is_concise", "Response is under 150 words", is_concise),
        ]
    ),

    # -------------------------------------------------------------------------
    # REASONING TASKS
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="reason-01",
        name="Math Word Problem",
        category=TaskCategory.REASONING,
        description="Multi-step arithmetic problem",
        zero_shot="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?""",
        schema="""MODE = Calculate
ACT = Solve
OBJ = Math word problem about apple pricing with discount
TAGS = [Format:Show work, Constraints:Include final answer clearly]
INPUT = A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount. How much would 7 apples cost?
OUTPUT = Step-by-step solution with final answer""",
        cot="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

Let's work through this step-by-step:
1. First, calculate the regular price
2. Then, determine if discount applies
3. Calculate the discount amount
4. Subtract to get final price

Show your reasoning and provide the final answer.""",
        criteria=[
            EvaluationCriteria("correct_answer", "Contains $11.20", math_answer_correct("11.20")),
            EvaluationCriteria("shows_work", "Shows calculation steps", has_step_by_step),
        ]
    ),

    MultiStylePrompt(
        id="reason-02",
        name="Logic Puzzle",
        category=TaskCategory.REASONING,
        description="Deductive reasoning task",
        zero_shot="""Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.
What pet does each person have?""",
        schema="""MODE = Analyze
ACT = Solve
OBJ = Logic puzzle about friends and their pets
TAGS = [Format:Structured reasoning, Constraints:Show deduction steps]
INPUT = Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish. Alice is allergic to fur. Bob's pet can bark.
OUTPUT = Solution with reasoning""",
        cot="""Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.
What pet does each person have?

Let's reason through this carefully:
1. What does "allergic to fur" tell us about Alice?
2. What does "can bark" tell us about Bob's pet?
3. By elimination, what's left for Carol?

Work through each clue and state your conclusion.""",
        criteria=[
            EvaluationCriteria("alice_fish", "Alice has fish", contains_answer(["alice", "fish"])),
            EvaluationCriteria("bob_dog", "Bob has dog", contains_answer(["bob", "dog"])),
            EvaluationCriteria("carol_cat", "Carol has cat", contains_answer(["carol", "cat"])),
        ]
    ),

    MultiStylePrompt(
        id="reason-03",
        name="Percentage Calculation",
        category=TaskCategory.REASONING,
        description="Business percentage problem",
        zero_shot="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?""",
        schema="""MODE = Calculate
ACT = Compute
OBJ = Percentage increase calculation
TAGS = [Format:Show formula and work]
INPUT = Revenue increased from $80,000 to $100,000
OUTPUT = Percentage with calculation shown""",
        cot="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?

Think through this step by step:
1. What is the formula for percentage increase?
2. What is the difference between new and old values?
3. Divide by the original and multiply by 100

Show your work and state the final percentage.""",
        criteria=[
            EvaluationCriteria("correct_answer", "Contains 25%", math_answer_correct("25")),
            EvaluationCriteria("shows_work", "Shows calculation", has_step_by_step),
        ]
    ),

    # -------------------------------------------------------------------------
    # CREATIVE TASKS
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="creative-01",
        name="Story Ideas",
        category=TaskCategory.CREATIVE,
        description="Generate creative content with constraints",
        zero_shot="""Give me 5 sci-fi story ideas. Keep each one to 2-3 sentences.""",
        schema="""MODE = Generate
ACT = Generate ideas
OBJ = Science fiction short story concepts
TAGS = [Format:Numbered list, Length:5 ideas, Style:Cyberpunk and hard sci-fi mix, Constraints:Each idea should be 2-3 sentences with a unique hook]
OUTPUT = Numbered list""",
        cot="""I want you to generate 5 sci-fi story ideas.

Before generating, consider:
1. What makes a compelling sci-fi hook? (unique technology, societal change, moral dilemma)
2. What subgenres could I explore? (cyberpunk, space opera, near-future, dystopian)
3. Each idea should be distinct and memorable

Now generate 5 ideas, each 2-3 sentences with a unique hook.""",
        criteria=[
            EvaluationCriteria("has_five", "Contains 5 distinct ideas", has_numbered_list),
        ]
    ),

    MultiStylePrompt(
        id="creative-02",
        name="Marketing Variants",
        category=TaskCategory.CREATIVE,
        description="Generate multiple creative variants",
        zero_shot="""Write marketing copy for a fitness app. Give me a few different versions.""",
        schema="""CONTEXT = Launching a mobile fitness app targeting busy professionals aged 25-40
PERSONA = You are a copywriter at a top digital marketing agency
MODE = Generate
ACT = Write
OBJ = Marketing copy for fitness app launch
TAGS = [Format:Multiple variants, Length:3 versions, Style:Energetic and motivating, Audience:Busy professionals]
OUTPUT = Three distinct marketing copy variants""",
        cot="""I need marketing copy for a fitness app.

Think through this:
1. Who is the target audience? (busy professionals who struggle to find time)
2. What pain points should we address? (time constraints, motivation, convenience)
3. What tone works best? (motivating but not preachy)

Now write 3 different versions of the marketing copy, each with a different angle or hook.""",
        criteria=[
            EvaluationCriteria("multiple_variants", "Contains multiple distinct versions",
                              lambda r: any(m in r.lower() for m in ["version", "variant", "option", "1.", "2.", "3."])),
        ]
    ),

    # -------------------------------------------------------------------------
    # ANALYSIS TASKS
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="analysis-01",
        name="Comparison Table",
        category=TaskCategory.ANALYSIS,
        description="Structured comparison output",
        zero_shot="""Compare React, Vue, and Angular. Put it in a table format.""",
        schema="""MODE = Compare
ACT = Compare
OBJ = React, Vue, and Angular JavaScript frameworks
TAGS = [Format:Table, Audience:Developer choosing a framework, Constraints:Include columns for learning curve, performance, ecosystem]
OUTPUT = Markdown table""",
        cot="""I need to compare React, Vue, and Angular frameworks.

Before creating the comparison:
1. What dimensions matter most to a developer choosing? (learning curve, performance, ecosystem, job market)
2. What format makes comparison easiest? (table for quick scanning)
3. How can I be objective and fair to each?

Create a comparison table with key dimensions as columns.""",
        criteria=[
            EvaluationCriteria("has_table", "Response is formatted as table", has_table),
            EvaluationCriteria("all_frameworks", "All three frameworks mentioned",
                              lambda r: all(fw.lower() in r.lower() for fw in ["react", "vue", "angular"])),
        ]
    ),

    MultiStylePrompt(
        id="analysis-02",
        name="Pros and Cons",
        category=TaskCategory.ANALYSIS,
        description="Balanced analysis task",
        zero_shot="""What are the pros and cons of remote work?""",
        schema="""MODE = Analyze
ACT = List
OBJ = Pros and cons of remote work
TAGS = [Format:Two sections (Pros/Cons), Length:4-5 points each, Style:Balanced and objective]
OUTPUT = Structured pros and cons list""",
        cot="""I want to understand the pros and cons of remote work.

Let me think through this systematically:
1. What are the benefits? (flexibility, no commute, work-life balance...)
2. What are the drawbacks? (isolation, communication challenges, work-life blur...)
3. Am I being balanced and considering multiple perspectives?

Provide a balanced list of 4-5 pros and 4-5 cons.""",
        criteria=[
            EvaluationCriteria("has_pros", "Contains pros section",
                              lambda r: "pro" in r.lower() or "benefit" in r.lower() or "advantage" in r.lower()),
            EvaluationCriteria("has_cons", "Contains cons section",
                              lambda r: "con" in r.lower() or "drawback" in r.lower() or "disadvantage" in r.lower()),
        ]
    ),

    # -------------------------------------------------------------------------
    # TECHNICAL TASKS
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="tech-01",
        name="Code Explanation",
        category=TaskCategory.TECHNICAL,
        description="Explain technical concept",
        zero_shot="""Explain how Python list comprehensions work. Include a code example.""",
        schema="""CONTEXT = Documentation for a developer learning Python
PERSONA = You are a senior Python developer writing beginner-friendly documentation
MODE = Explain
ACT = Explain
OBJ = Python list comprehensions
TAGS = [Format:Code+Explanation, Audience:Beginner developer, Constraints:Include 2-3 progressive examples]
OUTPUT = Text with code blocks""",
        cot="""I need to explain Python list comprehensions.

Let me think about the best way to teach this:
1. Start with a simple example that shows the basic syntax
2. Compare it to the traditional for-loop approach
3. Show a slightly more complex example with conditions
4. Explain when to use (and when not to use) list comprehensions

Provide an explanation with progressive code examples.""",
        criteria=[
            EvaluationCriteria("has_code", "Response includes code", has_code_block),
            EvaluationCriteria("has_example", "Contains working example",
                              lambda r: "[" in r and "for" in r),  # List comp syntax
        ]
    ),

    MultiStylePrompt(
        id="tech-02",
        name="Architecture Decision",
        category=TaskCategory.TECHNICAL,
        description="Technical recommendation with reasoning",
        zero_shot="""Should I use a SQL or NoSQL database for an e-commerce platform? Explain your recommendation.""",
        schema="""CONTEXT = Building an e-commerce platform with product catalog, user accounts, and order history
PERSONA = You are a solutions architect advising on database selection
MODE = Recommend
ACT = Recommend
OBJ = Database choice (SQL vs NoSQL) for e-commerce
TAGS = [Format:Recommendation with reasoning, Constraints:Consider scalability, data relationships, and query patterns]
OUTPUT = Clear recommendation with justification""",
        cot="""I need to decide between SQL and NoSQL for an e-commerce platform.

Let me reason through the key factors:
1. What are the data relationships? (users -> orders -> products: relational)
2. What queries will be common? (product searches, order history, inventory)
3. What are the scaling requirements? (read-heavy, occasional writes)
4. What consistency guarantees matter? (inventory accuracy, order integrity)

Based on this analysis, provide a recommendation with clear justification.""",
        criteria=[
            EvaluationCriteria("has_recommendation", "Makes clear recommendation",
                              lambda r: any(w in r.lower() for w in ["recommend", "suggest", "should use", "go with"])),
            EvaluationCriteria("has_reasoning", "Provides justification",
                              lambda r: any(w in r.lower() for w in ["because", "since", "reason", "advantage"])),
        ]
    ),

    # -------------------------------------------------------------------------
    # EDGE CASES - Testing specific prompt technique effects
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="edge-01",
        name="Persona Adherence",
        category=TaskCategory.WRITING,
        description="Test if persona instructions are followed",
        zero_shot="""Explain what machine learning is.""",
        schema="""PERSONA = You are a kindergarten teacher explaining things to 5-year-olds using simple words and fun analogies
ACT = Explain
OBJ = Machine learning
TAGS = [Audience:5-year-old children, Style:Fun and engaging, Constraints:Use only simple words, include an analogy]
OUTPUT = Child-friendly explanation""",
        cot="""I need to explain machine learning.

But first, let me think about my audience:
- Imagine I'm explaining this to a 5-year-old
- What simple analogy could make this clear? (like teaching a pet tricks?)
- What words would a child understand?

Now explain machine learning as if you're a kindergarten teacher talking to a young child. Use simple words and a fun analogy.""",
        criteria=[
            EvaluationCriteria("simple_language", "Uses simple vocabulary",
                              lambda r: not any(w in r.lower() for w in ["algorithm", "neural network", "optimization", "gradient"])),
            EvaluationCriteria("has_analogy", "Contains analogy",
                              lambda r: any(w in r.lower() for w in ["like", "imagine", "pretend", "think of"])),
        ]
    ),

    MultiStylePrompt(
        id="edge-02",
        name="Length Constraint",
        category=TaskCategory.WRITING,
        description="Test if length constraints are respected",
        zero_shot="""List exactly 3 benefits of meditation. One sentence each.""",
        schema="""ACT = List
OBJ = Benefits of meditation
TAGS = [Format:Bullets, Length:Exactly 3 points, Constraints:One sentence per point]
OUTPUT = 3 bullet points""",
        cot="""I need to list benefits of meditation.

Constraints to follow:
1. Exactly 3 benefits (not 2, not 4)
2. One sentence each (keep it brief)
3. Make each benefit distinct

List exactly 3 benefits of meditation, one sentence each.""",
        criteria=[
            EvaluationCriteria("exactly_three", "Exactly 3 points provided",
                              lambda r: 2 <= sum(1 for line in r.split("\n")
                                                if line.strip().startswith(("-", "•", "*", "1", "2", "3"))) <= 4),
        ]
    ),
]


def get_prompts_by_category(category: TaskCategory) -> list[MultiStylePrompt]:
    """Get all test prompts for a specific category."""
    return [p for p in MULTI_STYLE_PROMPTS if p.category == category]


def get_prompt_by_id(prompt_id: str) -> MultiStylePrompt:
    """Get a specific test prompt by ID."""
    for prompt in MULTI_STYLE_PROMPTS:
        if prompt.id == prompt_id:
            return prompt
    raise ValueError(f"Prompt not found: {prompt_id}")


if __name__ == "__main__":
    print(f"Total multi-style prompts: {len(MULTI_STYLE_PROMPTS)}")
    for category in TaskCategory:
        count = len(get_prompts_by_category(category))
        print(f"  {category.value}: {count}")

    print("\nPrompt styles available:")
    for style in PromptStyle:
        print(f"  - {style.value}")
