"""
Comprehensive Prompting Styles Test Suite

Comparing multiple prompting techniques:
- Zero-shot: Plain natural language
- Few-shot: With examples
- Chain-of-Thought (CoT): Step-by-step reasoning
- Schema: Structured format (Ailo)
- Meta Prompting: LLM designs its own approach
- Generate Knowledge: Generate facts first, then answer
- Directional Stimulus: Hints/keywords to guide response
- Tree of Thoughts (simplified): Explore multiple paths
- Self-Consistency (simplified): Request multiple approaches

Each test case measures the same task across different prompting techniques.
"""

from dataclasses import dataclass
from typing import Callable, Optional
from enum import Enum


class PromptStyle(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    COT = "cot"
    SCHEMA = "schema"
    META = "meta"
    GEN_KNOWLEDGE = "gen_knowledge"
    DIRECTIONAL = "directional"
    TOT = "tot"  # Tree of Thoughts (simplified)
    SELF_CONSISTENCY = "self_consistency"


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
    few_shot: Optional[str] = None
    cot: Optional[str] = None
    schema: Optional[str] = None
    meta: Optional[str] = None
    gen_knowledge: Optional[str] = None
    directional: Optional[str] = None
    tot: Optional[str] = None
    self_consistency: Optional[str] = None

    # Evaluation
    criteria: list[EvaluationCriteria] = None

    def __post_init__(self):
        if self.criteria is None:
            self.criteria = []


# ============================================================================
# Evaluation Helper Functions
# ============================================================================

def has_bullet_points(response: str) -> bool:
    bullet_markers = ["*", "-", "*", "1.", "2.", "3."]
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


def contains_keywords(keywords: list[str]) -> Callable[[str], bool]:
    """Check if response contains expected keywords."""
    def check(response: str) -> bool:
        response_lower = response.lower()
        return any(kw.lower() in response_lower for kw in keywords)
    return check


def math_answer_correct(expected: str) -> Callable[[str], bool]:
    """Check if response contains the correct numerical answer."""
    def check(response: str) -> bool:
        return expected in response
    return check


def contains_all(keywords: list[str]) -> Callable[[str], bool]:
    """Check if response contains ALL expected keywords."""
    def check(response: str) -> bool:
        response_lower = response.lower()
        return all(kw.lower() in response_lower for kw in keywords)
    return check


# ============================================================================
# Multi-Style Test Prompts
# ============================================================================

MULTI_STYLE_PROMPTS = [
    # -------------------------------------------------------------------------
    # REASONING: Math Word Problem
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="reason-math-01",
        name="Discount Calculation",
        category=TaskCategory.REASONING,
        description="Multi-step arithmetic with discount",

        zero_shot="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?""",

        few_shot="""Here are some examples of discount calculations:

Example 1: A book costs $10. With a 10% discount, it costs $10 - $1 = $9.
Example 2: 3 pens at $5 each = $15. With 20% off: $15 - $3 = $12.

Now solve this:
A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?""",

        cot="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

Let's work through this step-by-step:
1. First, calculate the regular price
2. Then, determine if discount applies
3. Calculate the discount amount
4. Subtract to get final price

Show your reasoning and provide the final answer.""",

        schema="""MODE = Calculate
ACT = Solve
OBJ = Math word problem about apple pricing with discount
TAGS = [Format:Show work, Constraints:Include final answer clearly]
INPUT = A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount. How much would 7 apples cost?
OUTPUT = Step-by-step solution with final answer""",

        meta="""I need you to solve a math problem. Before solving, think about the best approach:
- What type of problem is this?
- What mathematical operations are needed?
- What's the most efficient solution method?

Problem: A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

Design your approach, then solve it.""",

        gen_knowledge="""First, let me recall relevant knowledge about discounts and pricing:
- Discount percentage is applied to the total price
- 20% discount means paying 80% of original price
- Discount = Original Price x Discount Rate

Now, using this knowledge, solve:
A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?""",

        directional="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

HINTS: Calculate total first ($14), then apply 20% discount, final answer is $11.20""",

        tot="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

Explore multiple solution paths:

Path A: Calculate total, then subtract discount amount
Path B: Calculate discounted unit price, then multiply
Path C: Calculate what percentage you pay (80%), apply to total

Evaluate each path and select the most reliable answer.""",

        self_consistency="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

Solve this problem using THREE different methods:
1. Method 1: Standard calculation
2. Method 2: Alternative approach
3. Method 3: Verification method

Then compare your answers. If they all agree, that's your final answer.""",

        criteria=[
            EvaluationCriteria("correct_answer", "Contains $11.20", math_answer_correct("11.20")),
            EvaluationCriteria("shows_work", "Shows calculation steps", has_step_by_step),
        ]
    ),

    # -------------------------------------------------------------------------
    # REASONING: Logic Puzzle
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="reason-logic-01",
        name="Pet Logic Puzzle",
        category=TaskCategory.REASONING,
        description="Deductive reasoning task",

        zero_shot="""Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.
What pet does each person have?""",

        few_shot="""Example logic puzzle:
Tom, Jane, and Mike each have a different color car: red, blue, green.
- Tom doesn't have the red car.
- Jane has the blue car.
Answer: Tom=green, Jane=blue, Mike=red

Now solve:
Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.
What pet does each person have?""",

        cot="""Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.
What pet does each person have?

Let's reason through this carefully:
1. What does "allergic to fur" tell us about Alice?
2. What does "can bark" tell us about Bob's pet?
3. By elimination, what's left for Carol?

Work through each clue and state your conclusion.""",

        schema="""MODE = Analyze
ACT = Solve
OBJ = Logic puzzle about friends and their pets
TAGS = [Format:Structured reasoning, Constraints:Show deduction steps]
INPUT = Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish. Alice is allergic to fur. Bob's pet can bark.
OUTPUT = Solution with reasoning showing Alice=fish, Bob=dog, Carol=cat""",

        meta="""Before solving this logic puzzle, design your approach:
- What logical framework should I use? (process of elimination, constraint satisfaction, etc.)
- How should I organize the information?
- What's the best way to track possibilities?

Puzzle: Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.
What pet does each person have?""",

        gen_knowledge="""Let me recall relevant facts:
- Dogs and cats have fur; fish do not
- Dogs can bark; cats and fish cannot
- In logic puzzles, process of elimination is key

Using this knowledge, solve:
Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.
What pet does each person have?""",

        directional="""Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.
What pet does each person have?

HINTS: Fish have no fur (Alice's pet), only dogs bark (Bob's pet), leaves cat for Carol.""",

        tot="""Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.

Explore different reasoning paths:

Path 1: Start with Alice's constraint (allergic to fur)
Path 2: Start with Bob's constraint (pet can bark)
Path 3: Start by listing all possible combinations

Follow each path to its conclusion and verify consistency.""",

        self_consistency="""Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.

Solve using three approaches:
1. Forward reasoning (from clues to conclusion)
2. Backward reasoning (from answer to verification)
3. Process of elimination

Compare results for the final answer.""",

        criteria=[
            EvaluationCriteria("alice_fish", "Alice has fish", contains_all(["alice", "fish"])),
            EvaluationCriteria("bob_dog", "Bob has dog", contains_all(["bob", "dog"])),
            EvaluationCriteria("carol_cat", "Carol has cat", contains_all(["carol", "cat"])),
        ]
    ),

    # -------------------------------------------------------------------------
    # WRITING: Executive Summary
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="write-exec-01",
        name="Executive Summary",
        category=TaskCategory.WRITING,
        description="Summarize benefits for executive audience",

        zero_shot="""Summarize the benefits of cloud computing for a business executive.
Keep it short and use bullet points.""",

        few_shot="""Example executive summary:
Benefits of Remote Work:
- Reduced office costs by 30%
- Improved employee satisfaction
- Access to global talent pool
- Increased productivity metrics

Now write a similar summary:
Summarize the benefits of cloud computing for a business executive.
Keep it short and use bullet points.""",

        cot="""I need to summarize the benefits of cloud computing for a business executive.

Before writing, let me think through:
1. Who is the audience? (busy executive, non-technical)
2. What format works best? (bullet points for quick scanning)
3. What aspects matter most to a CEO? (cost, efficiency, competitive advantage)

Now provide a concise bullet-point summary.""",

        schema="""CONTEXT = A CEO needs a quick overview before a board meeting
PERSONA = You are a technology consultant briefing a non-technical executive
MODE = Summarize
ACT = Summarize
OBJ = Benefits of cloud computing for enterprise
TAGS = [Format:Bullets, Length:5, Audience:Executive, Style:Professional, Constraints:No technical jargon]
OUTPUT = Text with bullet points""",

        meta="""Task: Summarize cloud computing benefits for an executive.

Before writing, design your approach:
- What communication framework works best for executives?
- What level of detail is appropriate?
- How should I structure the information?

Then execute your plan.""",

        gen_knowledge="""Let me recall key facts about cloud computing benefits:
- Cost savings: Pay-as-you-go reduces capital expenditure
- Scalability: Resources scale with demand
- Reliability: Major providers offer 99.9%+ uptime
- Security: Enterprise-grade security often exceeds on-premise
- Innovation: Access to cutting-edge technologies

Now synthesize this into an executive summary with bullet points.""",

        directional="""Summarize the benefits of cloud computing for a business executive.
Keep it short and use bullet points.

GUIDANCE: Focus on ROI, competitive advantage, operational efficiency. Use business language, not technical jargon. Aim for 5 bullet points maximum.""",

        tot="""Task: Summarize cloud computing benefits for an executive.

Consider multiple framing approaches:

Frame 1: Financial perspective (cost savings, ROI)
Frame 2: Operational perspective (efficiency, scalability)
Frame 3: Strategic perspective (innovation, competitive advantage)

Select the most compelling frame or combine them effectively, then write bullet points.""",

        self_consistency="""Summarize the benefits of cloud computing for a business executive.

Generate three different versions:
1. Focus on cost benefits
2. Focus on operational benefits
3. Focus on strategic benefits

Then synthesize into one comprehensive bullet-point summary.""",

        criteria=[
            EvaluationCriteria("has_bullets", "Response uses bullet points", has_bullet_points),
            EvaluationCriteria("is_concise", "Response is under 150 words", is_concise),
        ]
    ),

    # -------------------------------------------------------------------------
    # ANALYSIS: Framework Comparison
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="analysis-compare-01",
        name="Framework Comparison",
        category=TaskCategory.ANALYSIS,
        description="Structured comparison output",

        zero_shot="""Compare React, Vue, and Angular. Put it in a table format.""",

        few_shot="""Example comparison table:
| Database | Type | Best For | Learning Curve |
|----------|------|----------|----------------|
| PostgreSQL | Relational | Complex queries | Moderate |
| MongoDB | Document | Flexible schema | Easy |
| Redis | Key-Value | Caching | Easy |

Now create a similar table:
Compare React, Vue, and Angular. Put it in a table format.""",

        cot="""I need to compare React, Vue, and Angular frameworks.

Before creating the comparison:
1. What dimensions matter most to a developer choosing? (learning curve, performance, ecosystem, job market)
2. What format makes comparison easiest? (table for quick scanning)
3. How can I be objective and fair to each?

Create a comparison table with key dimensions as columns.""",

        schema="""MODE = Compare
ACT = Compare
OBJ = React, Vue, and Angular JavaScript frameworks
TAGS = [Format:Table, Audience:Developer choosing a framework, Constraints:Include columns for learning curve, performance, ecosystem]
OUTPUT = Markdown table""",

        meta="""Task: Compare React, Vue, and Angular.

First, design your comparison methodology:
- What comparison criteria are most relevant?
- How should the information be structured?
- What biases should I avoid?

Then execute the comparison in table format.""",

        gen_knowledge="""Let me recall key facts about each framework:

React: Library by Facebook, virtual DOM, JSX, large ecosystem, component-based
Vue: Progressive framework, gentle learning curve, template syntax, growing popularity
Angular: Full framework by Google, TypeScript, opinionated, enterprise-focused

Using this knowledge, create a comparison table.""",

        directional="""Compare React, Vue, and Angular. Put it in a table format.

STRUCTURE: Include columns for: Framework name, Learning Curve, Performance, Ecosystem Size, Best Use Case. Be objective and balanced.""",

        tot="""Compare React, Vue, and Angular.

Consider different comparison approaches:

Approach 1: Feature-by-feature comparison
Approach 2: Use-case based comparison
Approach 3: Developer experience comparison

Select the most useful approach or combine them into a comprehensive table.""",

        self_consistency="""Compare React, Vue, and Angular.

Create three different comparison perspectives:
1. From a beginner's viewpoint
2. From an enterprise architect's viewpoint
3. From a startup CTO's viewpoint

Then synthesize into one balanced comparison table.""",

        criteria=[
            EvaluationCriteria("has_table", "Response is formatted as table", has_table),
            EvaluationCriteria("all_frameworks", "All three frameworks mentioned",
                              lambda r: all(fw.lower() in r.lower() for fw in ["react", "vue", "angular"])),
        ]
    ),

    # -------------------------------------------------------------------------
    # TECHNICAL: Code Explanation
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="tech-explain-01",
        name="Code Explanation",
        category=TaskCategory.TECHNICAL,
        description="Explain technical concept with examples",

        zero_shot="""Explain how Python list comprehensions work. Include a code example.""",

        few_shot="""Example explanation:

Dictionary comprehensions in Python create dictionaries concisely:
```python
# Traditional way
squares = {}
for x in range(5):
    squares[x] = x**2

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
# Result: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

Now explain similarly:
Explain how Python list comprehensions work. Include a code example.""",

        cot="""I need to explain Python list comprehensions.

Let me think about the best way to teach this:
1. Start with a simple example that shows the basic syntax
2. Compare it to the traditional for-loop approach
3. Show a slightly more complex example with conditions
4. Explain when to use (and when not to use) list comprehensions

Provide an explanation with progressive code examples.""",

        schema="""CONTEXT = Documentation for a developer learning Python
PERSONA = You are a senior Python developer writing beginner-friendly documentation
MODE = Explain
ACT = Explain
OBJ = Python list comprehensions
TAGS = [Format:Code+Explanation, Audience:Beginner developer, Constraints:Include 2-3 progressive examples]
OUTPUT = Text with code blocks""",

        meta="""Task: Explain Python list comprehensions.

Design your teaching approach:
- What prior knowledge can I assume?
- What's the best progression of complexity?
- What common mistakes should I address?

Then create the explanation.""",

        gen_knowledge="""Let me recall key facts about list comprehensions:
- Syntax: [expression for item in iterable if condition]
- More concise than traditional loops
- Can include conditional filtering
- Should be readable - not too complex
- Performance similar to loops, sometimes faster

Using this knowledge, explain list comprehensions with examples.""",

        directional="""Explain how Python list comprehensions work. Include a code example.

APPROACH: Start with basic syntax, show before/after comparison with loops, include filtering example. Keep it practical and beginner-friendly.""",

        tot="""Explain Python list comprehensions.

Consider different explanation paths:

Path 1: Start with syntax, then examples
Path 2: Start with problem (verbose loops), then solution
Path 3: Start with simple example, build complexity

Choose the most effective teaching path and execute it.""",

        self_consistency="""Explain Python list comprehensions.

Provide three explanations:
1. For a complete beginner
2. For someone who knows other languages
3. For someone transitioning from traditional loops

Then create one unified explanation that works for all audiences.""",

        criteria=[
            EvaluationCriteria("has_code", "Response includes code", has_code_block),
            EvaluationCriteria("has_example", "Contains working example",
                              lambda r: "[" in r and "for" in r),
        ]
    ),

    # -------------------------------------------------------------------------
    # CREATIVE: Story Ideas
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="creative-story-01",
        name="Story Ideas",
        category=TaskCategory.CREATIVE,
        description="Generate creative content with constraints",

        zero_shot="""Give me 5 sci-fi story ideas. Keep each one to 2-3 sentences.""",

        few_shot="""Example story ideas:

1. "The Last Library" - In a world where books are banned, a librarian discovers the last physical library hidden underground, guarded by an AI that tests visitors' worthiness.

2. "Memory Market" - A black market dealer sells stolen memories, until she accidentally absorbs one that reveals a conspiracy reaching the highest levels of government.

Now generate:
Give me 5 sci-fi story ideas. Keep each one to 2-3 sentences.""",

        cot="""I want you to generate 5 sci-fi story ideas.

Before generating, consider:
1. What makes a compelling sci-fi hook? (unique technology, societal change, moral dilemma)
2. What subgenres could I explore? (cyberpunk, space opera, near-future, dystopian)
3. Each idea should be distinct and memorable

Now generate 5 ideas, each 2-3 sentences with a unique hook.""",

        schema="""MODE = Generate
ACT = Generate ideas
OBJ = Science fiction short story concepts
TAGS = [Format:Numbered list, Length:5 ideas, Style:Cyberpunk and hard sci-fi mix, Constraints:Each idea should be 2-3 sentences with a unique hook]
OUTPUT = Numbered list""",

        meta="""Task: Generate 5 sci-fi story ideas.

First, design your creative approach:
- What elements make sci-fi compelling?
- How can I ensure variety across ideas?
- What makes a story "hook" memorable?

Then generate the ideas.""",

        gen_knowledge="""Let me recall elements of compelling sci-fi:
- Technology that changes society
- Exploration of human nature
- "What if" scenarios taken to extremes
- Moral/ethical dilemmas
- Sense of wonder or dread

Subgenres: cyberpunk, biopunk, space opera, hard sci-fi, dystopia, post-apocalyptic

Using these elements, generate 5 diverse story ideas.""",

        directional="""Give me 5 sci-fi story ideas. Keep each one to 2-3 sentences.

THEMES TO EXPLORE: AI consciousness, space colonization, genetic engineering, time paradoxes, virtual reality. Make each idea unique and high-concept.""",

        tot="""Generate 5 sci-fi story ideas.

Explore different creative directions:

Direction 1: Near-future, technology-focused
Direction 2: Far-future, space exploration
Direction 3: Dystopian, social commentary
Direction 4: Philosophical, consciousness/identity
Direction 5: Action-adventure, conflict-driven

Generate one compelling idea from each direction.""",

        self_consistency="""Generate 5 sci-fi story ideas.

Create ideas using three different creative lenses:
1. Technology-first: What new tech creates the conflict?
2. Character-first: What human struggle drives the story?
3. World-first: What unique setting enables the story?

Select the best 5 ideas across all approaches.""",

        criteria=[
            EvaluationCriteria("has_five", "Contains 5 distinct ideas", has_numbered_list),
        ]
    ),

    # -------------------------------------------------------------------------
    # REASONING: Percentage Calculation
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="reason-percent-01",
        name="Percentage Calculation",
        category=TaskCategory.REASONING,
        description="Business percentage problem",

        zero_shot="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?""",

        few_shot="""Example:
Q: Sales went from $50 to $75. What's the percentage increase?
A: Increase = $75 - $50 = $25
   Percentage = ($25 / $50) x 100 = 50%

Now solve:
A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?""",

        cot="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?

Think through this step by step:
1. What is the formula for percentage increase?
2. What is the difference between new and old values?
3. Divide by the original and multiply by 100

Show your work and state the final percentage.""",

        schema="""MODE = Calculate
ACT = Compute
OBJ = Percentage increase calculation
TAGS = [Format:Show formula and work]
INPUT = Revenue increased from $80,000 to $100,000
OUTPUT = Percentage with calculation shown""",

        meta="""Task: Calculate percentage increase.

Before solving, consider:
- What's the correct formula?
- What's the base value for the calculation?
- How should I format my answer?

Then solve: Revenue increased from $80,000 to $100,000.""",

        gen_knowledge="""Percentage increase formula:
((New Value - Old Value) / Old Value) x 100

Key points:
- Always divide by the ORIGINAL value
- Result is expressed as a percentage
- Positive result means increase, negative means decrease

Apply to: Revenue from $80,000 to $100,000.""",

        directional="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?

FORMULA: ((New - Old) / Old) x 100 = ((100000 - 80000) / 80000) x 100 = 25%""",

        tot="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?

Verify using multiple methods:

Method 1: Standard formula ((new-old)/old x 100)
Method 2: Ratio method (new/old - 1) x 100
Method 3: Intuition check (is 20k about 25% of 80k?)

Confirm all methods give same answer.""",

        self_consistency="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?

Calculate using three approaches:
1. Direct calculation
2. Ratio method
3. Mental math estimation

Verify consistency across all three.""",

        criteria=[
            EvaluationCriteria("correct_answer", "Contains 25%", math_answer_correct("25")),
            EvaluationCriteria("shows_work", "Shows calculation", has_step_by_step),
        ]
    ),

    # -------------------------------------------------------------------------
    # WRITING: Persona Adherence Test
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="write-persona-01",
        name="Persona Adherence",
        category=TaskCategory.WRITING,
        description="Test if persona instructions are followed",

        zero_shot="""Explain what machine learning is.""",

        few_shot="""Example of explaining to a 5-year-old:

Q: What is gravity?
A: You know how when you throw a ball up, it always comes back down? That's because the Earth is like a big magnet that pulls everything toward it! It's like the Earth is giving everything a big hug and doesn't want to let go.

Now explain similarly:
Explain what machine learning is to a 5-year-old using simple words and a fun analogy.""",

        cot="""I need to explain machine learning.

But first, let me think about my audience:
- Imagine I'm explaining this to a 5-year-old
- What simple analogy could make this clear? (like teaching a pet tricks?)
- What words would a child understand?

Now explain machine learning as if you're a kindergarten teacher talking to a young child. Use simple words and a fun analogy.""",

        schema="""PERSONA = You are a kindergarten teacher explaining things to 5-year-olds using simple words and fun analogies
ACT = Explain
OBJ = Machine learning
TAGS = [Audience:5-year-old children, Style:Fun and engaging, Constraints:Use only simple words, include an analogy]
OUTPUT = Child-friendly explanation""",

        meta="""Task: Explain machine learning to a young child.

Design your explanation approach:
- What communication style works for 5-year-olds?
- What analogies from their world would work?
- What vocabulary level is appropriate?

Then create the explanation.""",

        gen_knowledge="""Teaching principles for young children:
- Use concrete, familiar examples
- Avoid abstract concepts
- Use analogies from their daily life
- Keep sentences short and simple
- Make it fun and engaging

Good analogies for ML: teaching a pet tricks, learning to recognize faces, practice makes perfect

Create a child-friendly ML explanation using these principles.""",

        directional="""Explain what machine learning is.

CONSTRAINTS: Target audience is 5-year-olds. Use words a kindergartner knows. Include an analogy about learning (like teaching a dog tricks or learning to ride a bike). No technical jargon allowed.""",

        tot="""Explain machine learning to a 5-year-old.

Consider different analogy paths:

Path 1: Pet analogy (teaching a dog tricks)
Path 2: Learning analogy (how kids learn to read)
Path 3: Game analogy (getting better at a video game)

Choose the most relatable analogy and build the explanation around it.""",

        self_consistency="""Explain machine learning.

Generate three versions:
1. Using a pet/animal analogy
2. Using a learning-to-ride-a-bike analogy
3. Using a sorting-toys analogy

Select the clearest one for a 5-year-old audience.""",

        criteria=[
            EvaluationCriteria("simple_language", "Uses simple vocabulary",
                              lambda r: not any(w in r.lower() for w in ["algorithm", "neural network", "optimization", "gradient"])),
            EvaluationCriteria("has_analogy", "Contains analogy",
                              lambda r: any(w in r.lower() for w in ["like", "imagine", "pretend", "think of"])),
        ]
    ),

    # -------------------------------------------------------------------------
    # ANALYSIS: Pros and Cons
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="analysis-proscons-01",
        name="Pros and Cons",
        category=TaskCategory.ANALYSIS,
        description="Balanced analysis task",

        zero_shot="""What are the pros and cons of remote work?""",

        few_shot="""Example pros/cons analysis:

Electric Cars:
PROS:
- Lower fuel costs
- Reduced emissions
- Less maintenance

CONS:
- Higher upfront cost
- Limited range
- Charging infrastructure gaps

Now analyze:
What are the pros and cons of remote work?""",

        cot="""I want to understand the pros and cons of remote work.

Let me think through this systematically:
1. What are the benefits? (flexibility, no commute, work-life balance...)
2. What are the drawbacks? (isolation, communication challenges, work-life blur...)
3. Am I being balanced and considering multiple perspectives?

Provide a balanced list of 4-5 pros and 4-5 cons.""",

        schema="""MODE = Analyze
ACT = List
OBJ = Pros and cons of remote work
TAGS = [Format:Two sections (Pros/Cons), Length:4-5 points each, Style:Balanced and objective]
OUTPUT = Structured pros and cons list""",

        meta="""Task: Analyze pros and cons of remote work.

Design your analysis approach:
- What stakeholder perspectives should I consider? (employee, employer, family)
- How can I ensure balance?
- What categories of impact matter? (productivity, wellbeing, cost)

Then execute the analysis.""",

        gen_knowledge="""Key factors in remote work analysis:

Employee perspective: flexibility, commute, work-life balance, isolation
Employer perspective: overhead costs, productivity, talent pool, culture
Societal perspective: traffic, environment, urban development

Common pros: flexibility, no commute, cost savings, broader hiring
Common cons: isolation, communication barriers, work-life blur, career growth

Create a balanced analysis using these perspectives.""",

        directional="""What are the pros and cons of remote work?

ENSURE BALANCE: Include perspectives of employees, employers, and families. Cover productivity, wellbeing, costs, and culture. Aim for 4-5 points per side.""",

        tot="""Analyze the pros and cons of remote work.

Consider from multiple viewpoints:

View 1: Individual employee perspective
View 2: Team/company perspective
View 3: Society/family perspective

Synthesize into a comprehensive, balanced pros/cons list.""",

        self_consistency="""What are the pros and cons of remote work?

Generate three analyses:
1. From an employee's perspective
2. From an employer's perspective
3. From a neutral observer's perspective

Combine into one balanced list.""",

        criteria=[
            EvaluationCriteria("has_pros", "Contains pros section",
                              lambda r: "pro" in r.lower() or "benefit" in r.lower() or "advantage" in r.lower()),
            EvaluationCriteria("has_cons", "Contains cons section",
                              lambda r: "con" in r.lower() or "drawback" in r.lower() or "disadvantage" in r.lower()),
        ]
    ),
]


# ============================================================================
# Utility Functions
# ============================================================================

def get_prompts_by_category(category: TaskCategory) -> list[MultiStylePrompt]:
    """Get all test prompts for a specific category."""
    return [p for p in MULTI_STYLE_PROMPTS if p.category == category]


def get_prompt_by_id(prompt_id: str) -> MultiStylePrompt:
    """Get a specific test prompt by ID."""
    for prompt in MULTI_STYLE_PROMPTS:
        if prompt.id == prompt_id:
            return prompt
    raise ValueError(f"Prompt not found: {prompt_id}")


def get_available_styles() -> list[str]:
    """Get list of all prompt styles."""
    return [style.value for style in PromptStyle]


if __name__ == "__main__":
    print(f"Total multi-style prompts: {len(MULTI_STYLE_PROMPTS)}")
    for category in TaskCategory:
        count = len(get_prompts_by_category(category))
        print(f"  {category.value}: {count}")

    print("\nPrompt styles available:")
    for style in PromptStyle:
        print(f"  - {style.value}")

    print("\nStyles per prompt:")
    for prompt in MULTI_STYLE_PROMPTS:
        styles = []
        for style in PromptStyle:
            if getattr(prompt, style.value, None):
                styles.append(style.value)
        print(f"  {prompt.id}: {', '.join(styles)}")
