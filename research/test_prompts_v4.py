"""
Prompting Styles Test Suite v4

Refactored with improved evaluation schema:
- Clear ground truth for each test
- Multiple valid answers supported
- Evaluation type specified per test
- Answer leakage removed from directional prompts
- Numeric tolerance for math problems

Prompting techniques tested:
- zero_shot: Plain natural language
- few_shot: With examples
- cot: Chain-of-Thought step-by-step reasoning
- schema: Structured format (Ailo)
- meta: LLM designs its own approach
- gen_knowledge: Generate facts first, then answer
- directional: Hints/keywords to guide (NO answer leakage)
- tot: Tree of Thoughts - explore multiple paths
- self_consistency: Multiple approaches, reconcile
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum

from evaluation import EvalType, TestCase


class PromptStyle(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    COT = "cot"
    SCHEMA = "schema"
    META = "meta"
    GEN_KNOWLEDGE = "gen_knowledge"
    DIRECTIONAL = "directional"
    TOT = "tot"
    SELF_CONSISTENCY = "self_consistency"


class TaskCategory(Enum):
    MATH = "math"
    LOGIC = "logic"
    WRITING = "writing"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    TECHNICAL = "technical"
    CODE = "code"


@dataclass
class MultiStylePrompt:
    """
    A test case with multiple prompting style variants.
    Includes evaluation configuration.
    """
    id: str
    name: str
    category: TaskCategory
    description: str

    # Evaluation config
    eval_type: EvalType
    expected_answers: list[str] = field(default_factory=list)
    expected_number: Optional[float] = None
    expected_keywords: list[str] = field(default_factory=list)
    numeric_tolerance: float = 0.01

    # Different prompt styles for the same task
    zero_shot: str = ""
    few_shot: Optional[str] = None
    cot: Optional[str] = None
    schema: Optional[str] = None
    meta: Optional[str] = None
    gen_knowledge: Optional[str] = None
    directional: Optional[str] = None
    tot: Optional[str] = None
    self_consistency: Optional[str] = None

    def to_test_case(self) -> TestCase:
        """Convert to TestCase for evaluation."""
        return TestCase(
            id=self.id,
            name=self.name,
            category=self.category.value,
            task=self.zero_shot,
            eval_type=self.eval_type,
            expected_answers=self.expected_answers,
            expected_number=self.expected_number,
            expected_keywords=self.expected_keywords,
            numeric_tolerance=self.numeric_tolerance,
        )


# =============================================================================
# Test Prompts
# =============================================================================

MULTI_STYLE_PROMPTS = [
    # -------------------------------------------------------------------------
    # MATH: Discount Calculation
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="math-discount-01",
        name="Discount Calculation",
        category=TaskCategory.MATH,
        description="Multi-step arithmetic with percentage discount",

        # Evaluation config
        eval_type=EvalType.NUMERIC,
        expected_number=11.20,
        expected_answers=["11.20", "$11.20", "11.2", "$11.2"],

        zero_shot="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost? Give the final price.""",

        few_shot="""Here are some examples of discount calculations:

Example 1: A book costs $10. With a 10% discount, it costs $10 - $1 = $9.
Example 2: 3 pens at $5 each = $15. With 20% off: $15 - $3 = $12.

Now solve this:
A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost? Give the final price.""",

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
OUTPUT = Step-by-step solution with final numerical answer""",

        meta="""I need you to solve a math problem. Before solving, think about the best approach:
- What type of problem is this?
- What mathematical operations are needed?
- What's the most efficient solution method?

Problem: A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

Design your approach, then solve it. Provide the final price.""",

        gen_knowledge="""First, recall relevant knowledge about discounts and pricing:
- Discount percentage is applied to the total price
- 20% discount means paying 80% of original price
- Discount = Original Price × Discount Rate

Now, using this knowledge, solve:
A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost? Give the final price.""",

        # FIXED: No answer leakage - only hints about approach
        directional="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

HINTS:
- First multiply quantity by unit price for total
- Check if discount threshold is met
- Apply percentage discount to total
- Give the final price as a dollar amount""",

        tot="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

Explore multiple solution paths:

Path A: Calculate total, then subtract discount amount
Path B: Calculate discounted unit price, then multiply
Path C: Calculate what percentage you pay (80%), apply to total

Evaluate each path and select the most reliable answer. Provide the final price.""",

        self_consistency="""A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount.
How much would 7 apples cost?

Solve this problem using THREE different methods:
1. Method 1: Standard calculation
2. Method 2: Alternative approach
3. Method 3: Verification method

Then compare your answers. State the final price.""",
    ),

    # -------------------------------------------------------------------------
    # MATH: Percentage Increase
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="math-percent-01",
        name="Percentage Calculation",
        category=TaskCategory.MATH,
        description="Business percentage increase problem",

        eval_type=EvalType.NUMERIC,
        expected_number=25.0,
        expected_answers=["25%", "25 percent", "25"],

        zero_shot="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?""",

        few_shot="""Example:
Q: Sales went from $50 to $75. What's the percentage increase?
A: Increase = $75 - $50 = $25
   Percentage = ($25 / $50) × 100 = 50%

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

Then solve: Revenue increased from $80,000 to $100,000. What is the percentage increase?""",

        gen_knowledge="""Percentage increase formula:
((New Value - Old Value) / Old Value) × 100

Key points:
- Always divide by the ORIGINAL value
- Result is expressed as a percentage

Apply to: Revenue from $80,000 to $100,000. What is the percentage increase?""",

        # FIXED: Removed the answer
        directional="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?

HINTS:
- Use the formula: ((New - Old) / Old) × 100
- The denominator should be the original value
- Express your answer as a percentage""",

        tot="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?

Verify using multiple methods:

Method 1: Standard formula ((new-old)/old × 100)
Method 2: Ratio method (new/old - 1) × 100
Method 3: Intuition check - estimate the fraction

Confirm all methods give same answer.""",

        self_consistency="""A company's revenue increased from $80,000 to $100,000.
What is the percentage increase?

Calculate using three approaches:
1. Direct calculation
2. Ratio method
3. Mental math estimation

Verify consistency across all three and state the final percentage.""",
    ),

    # -------------------------------------------------------------------------
    # LOGIC: Pet Puzzle
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="logic-pets-01",
        name="Pet Logic Puzzle",
        category=TaskCategory.LOGIC,
        description="Deductive reasoning task",

        eval_type=EvalType.KEYWORDS,
        expected_keywords=["alice", "fish", "bob", "dog", "carol", "cat"],
        expected_answers=["Alice has fish, Bob has dog, Carol has cat"],

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
OUTPUT = Solution showing which person has which pet""",

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

        # FIXED: Removed the answers
        directional="""Three friends - Alice, Bob, and Carol - each have a different pet: a cat, a dog, and a fish.
- Alice is allergic to fur.
- Bob's pet can bark.
What pet does each person have?

HINTS:
- Consider which animals have fur
- Consider which animals can bark
- Use process of elimination for the remaining person""",

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
    ),

    # -------------------------------------------------------------------------
    # WRITING: Executive Summary
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="writing-exec-01",
        name="Executive Summary",
        category=TaskCategory.WRITING,
        description="Summarize benefits for executive audience",

        eval_type=EvalType.KEYWORDS,
        expected_keywords=["cost", "scal", "flex", "secur", "effic"],  # Partial matches
        expected_answers=[],

        zero_shot="""Summarize the benefits of cloud computing for a business executive.
Keep it short and use bullet points. Provide 4-5 key benefits.""",

        few_shot="""Example executive summary:
Benefits of Remote Work:
- Reduced office costs by 30%
- Improved employee satisfaction
- Access to global talent pool
- Increased productivity metrics

Now write a similar summary:
Summarize the benefits of cloud computing for a business executive.
Keep it short and use bullet points. Provide 4-5 key benefits.""",

        cot="""I need to summarize the benefits of cloud computing for a business executive.

Before writing, let me think through:
1. Who is the audience? (busy executive, non-technical)
2. What format works best? (bullet points for quick scanning)
3. What aspects matter most? (cost, efficiency, competitive advantage)

Now provide a concise bullet-point summary with 4-5 benefits.""",

        schema="""CONTEXT = A CEO needs a quick overview before a board meeting
PERSONA = You are a technology consultant briefing a non-technical executive
MODE = Summarize
ACT = Summarize
OBJ = Benefits of cloud computing for enterprise
TAGS = [Format:Bullets, Length:4-5 points, Audience:Executive, Style:Professional, Constraints:No technical jargon]
OUTPUT = Text with bullet points""",

        meta="""Task: Summarize cloud computing benefits for an executive.

Before writing, design your approach:
- What communication framework works best for executives?
- What level of detail is appropriate?
- How should I structure the information?

Then execute your plan with 4-5 bullet points.""",

        gen_knowledge="""Let me recall key facts about cloud computing benefits:
- Cost savings: Pay-as-you-go reduces capital expenditure
- Scalability: Resources scale with demand
- Reliability: Major providers offer 99.9%+ uptime
- Security: Enterprise-grade security often exceeds on-premise
- Innovation: Access to cutting-edge technologies

Now synthesize this into an executive summary with 4-5 bullet points.""",

        directional="""Summarize the benefits of cloud computing for a business executive.
Keep it short and use bullet points.

GUIDANCE:
- Focus on business value, not technical details
- Include ROI and operational efficiency points
- Use executive-friendly language
- Aim for 4-5 bullet points maximum""",

        tot="""Task: Summarize cloud computing benefits for an executive.

Consider multiple framing approaches:

Frame 1: Financial perspective (cost savings, ROI)
Frame 2: Operational perspective (efficiency, scalability)
Frame 3: Strategic perspective (innovation, competitive advantage)

Select the most compelling frame or combine them effectively, then write 4-5 bullet points.""",

        self_consistency="""Summarize the benefits of cloud computing for a business executive.

Generate three different versions:
1. Focus on cost benefits
2. Focus on operational benefits
3. Focus on strategic benefits

Then synthesize into one comprehensive bullet-point summary with 4-5 points.""",
    ),

    # -------------------------------------------------------------------------
    # ANALYSIS: Framework Comparison
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="analysis-compare-01",
        name="Framework Comparison",
        category=TaskCategory.ANALYSIS,
        description="Structured comparison output",

        eval_type=EvalType.KEYWORDS,
        expected_keywords=["react", "vue", "angular", "|"],  # Must have table
        expected_answers=[],

        zero_shot="""Compare React, Vue, and Angular. Put it in a table format with columns for learning curve, performance, and ecosystem.""",

        few_shot="""Example comparison table:
| Database | Type | Best For | Learning Curve |
|----------|------|----------|----------------|
| PostgreSQL | Relational | Complex queries | Moderate |
| MongoDB | Document | Flexible schema | Easy |
| Redis | Key-Value | Caching | Easy |

Now create a similar table:
Compare React, Vue, and Angular. Include columns for learning curve, performance, and ecosystem.""",

        cot="""I need to compare React, Vue, and Angular frameworks.

Before creating the comparison:
1. What dimensions matter most to a developer choosing? (learning curve, performance, ecosystem)
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

Then execute the comparison in table format with at least 3 columns.""",

        gen_knowledge="""Let me recall key facts about each framework:

React: Library by Facebook, virtual DOM, JSX, large ecosystem, component-based
Vue: Progressive framework, gentle learning curve, template syntax, growing popularity
Angular: Full framework by Google, TypeScript, opinionated, enterprise-focused

Using this knowledge, create a comparison table with columns for learning curve, performance, and ecosystem.""",

        directional="""Compare React, Vue, and Angular. Put it in a table format.

STRUCTURE:
- Include columns for: Framework name, Learning Curve, Performance, Ecosystem Size
- Be objective and balanced
- Use a markdown table format""",

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
    ),

    # -------------------------------------------------------------------------
    # TECHNICAL: Code Explanation
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="tech-explain-01",
        name="Code Explanation",
        category=TaskCategory.TECHNICAL,
        description="Explain technical concept with examples",

        eval_type=EvalType.KEYWORDS,
        expected_keywords=["for", "[", "]", "list"],  # Must show list comprehension syntax
        expected_answers=[],

        zero_shot="""Explain how Python list comprehensions work. Include a code example showing the syntax.""",

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
Explain how Python list comprehensions work. Include a code example showing the syntax.""",

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

Then create the explanation with code examples.""",

        gen_knowledge="""Let me recall key facts about list comprehensions:
- Syntax: [expression for item in iterable if condition]
- More concise than traditional loops
- Can include conditional filtering
- Should be readable - not too complex

Using this knowledge, explain list comprehensions with code examples.""",

        directional="""Explain how Python list comprehensions work. Include a code example.

APPROACH:
- Start with basic syntax pattern: [expr for item in iterable]
- Show before/after comparison with traditional loops
- Include a filtering example with 'if'
- Keep it practical and beginner-friendly""",

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
    ),

    # -------------------------------------------------------------------------
    # CREATIVE: Story Ideas
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="creative-story-01",
        name="Story Ideas",
        category=TaskCategory.CREATIVE,
        description="Generate creative content with constraints",

        eval_type=EvalType.KEYWORDS,
        expected_keywords=["1", "2", "3", "4", "5"],  # Must have numbered list
        expected_answers=[],

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
TAGS = [Format:Numbered list, Length:5 ideas, Style:Mix of subgenres, Constraints:Each idea 2-3 sentences with unique hook]
OUTPUT = Numbered list of 5 story ideas""",

        meta="""Task: Generate 5 sci-fi story ideas.

First, design your creative approach:
- What elements make sci-fi compelling?
- How can I ensure variety across ideas?
- What makes a story "hook" memorable?

Then generate the 5 ideas, each 2-3 sentences.""",

        gen_knowledge="""Let me recall elements of compelling sci-fi:
- Technology that changes society
- Exploration of human nature
- "What if" scenarios taken to extremes
- Moral/ethical dilemmas

Subgenres: cyberpunk, biopunk, space opera, hard sci-fi, dystopia

Using these elements, generate 5 diverse story ideas, each 2-3 sentences.""",

        directional="""Give me 5 sci-fi story ideas. Keep each one to 2-3 sentences.

THEMES TO EXPLORE:
- AI consciousness
- Space colonization
- Genetic engineering
- Time paradoxes
- Virtual reality

Make each idea unique and high-concept.""",

        tot="""Generate 5 sci-fi story ideas.

Explore different creative directions:

Direction 1: Near-future, technology-focused
Direction 2: Far-future, space exploration
Direction 3: Dystopian, social commentary
Direction 4: Philosophical, consciousness/identity
Direction 5: Action-adventure, conflict-driven

Generate one compelling idea from each direction, 2-3 sentences each.""",

        self_consistency="""Generate 5 sci-fi story ideas.

Create ideas using three different creative lenses:
1. Technology-first: What new tech creates the conflict?
2. Character-first: What human struggle drives the story?
3. World-first: What unique setting enables the story?

Select the best 5 ideas across all approaches, 2-3 sentences each.""",
    ),

    # -------------------------------------------------------------------------
    # ANALYSIS: Pros and Cons
    # -------------------------------------------------------------------------
    MultiStylePrompt(
        id="analysis-proscons-01",
        name="Pros and Cons",
        category=TaskCategory.ANALYSIS,
        description="Balanced analysis task",

        eval_type=EvalType.KEYWORDS,
        expected_keywords=["pro", "con", "advantage", "disadvantage", "benefit", "drawback"],
        expected_answers=[],

        zero_shot="""What are the pros and cons of remote work? List 3-4 points for each.""",

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
What are the pros and cons of remote work? List 3-4 points for each.""",

        cot="""I want to understand the pros and cons of remote work.

Let me think through this systematically:
1. What are the benefits? (flexibility, no commute, work-life balance...)
2. What are the drawbacks? (isolation, communication challenges, work-life blur...)
3. Am I being balanced and considering multiple perspectives?

Provide a balanced list of 3-4 pros and 3-4 cons.""",

        schema="""MODE = Analyze
ACT = List
OBJ = Pros and cons of remote work
TAGS = [Format:Two sections (Pros/Cons), Length:3-4 points each, Style:Balanced and objective]
OUTPUT = Structured pros and cons list""",

        meta="""Task: Analyze pros and cons of remote work.

Design your analysis approach:
- What stakeholder perspectives should I consider? (employee, employer, family)
- How can I ensure balance?
- What categories of impact matter? (productivity, wellbeing, cost)

Then execute the analysis with 3-4 points per side.""",

        gen_knowledge="""Key factors in remote work analysis:

Employee perspective: flexibility, commute, work-life balance, isolation
Employer perspective: overhead costs, productivity, talent pool, culture
Societal perspective: traffic, environment, urban development

Create a balanced analysis with 3-4 pros and 3-4 cons.""",

        directional="""What are the pros and cons of remote work?

ENSURE BALANCE:
- Include perspectives of employees and employers
- Cover productivity, wellbeing, and costs
- Aim for 3-4 points per side
- Be objective""",

        tot="""Analyze the pros and cons of remote work.

Consider from multiple viewpoints:

View 1: Individual employee perspective
View 2: Team/company perspective
View 3: Society/family perspective

Synthesize into a comprehensive, balanced pros/cons list with 3-4 points each.""",

        self_consistency="""What are the pros and cons of remote work?

Generate three analyses:
1. From an employee's perspective
2. From an employer's perspective
3. From a neutral observer's perspective

Combine into one balanced list with 3-4 pros and 3-4 cons.""",
    ),
]


# =============================================================================
# Utility Functions
# =============================================================================

def get_prompts_by_category(category: TaskCategory) -> list[MultiStylePrompt]:
    """Get all test prompts for a specific category."""
    return [p for p in MULTI_STYLE_PROMPTS if p.category == category]


def get_prompt_by_id(prompt_id: str) -> MultiStylePrompt:
    """Get a specific test prompt by ID."""
    for prompt in MULTI_STYLE_PROMPTS:
        if prompt.id == prompt_id:
            return prompt
    raise ValueError(f"Prompt not found: {prompt_id}")


def get_all_test_cases() -> list[TestCase]:
    """Convert all prompts to TestCase objects for evaluation."""
    return [p.to_test_case() for p in MULTI_STYLE_PROMPTS]


def get_available_styles() -> list[str]:
    """Get list of all prompt styles."""
    return [style.value for style in PromptStyle]


if __name__ == "__main__":
    print(f"Total multi-style prompts: {len(MULTI_STYLE_PROMPTS)}")

    for category in TaskCategory:
        count = len(get_prompts_by_category(category))
        if count > 0:
            print(f"  {category.value}: {count}")

    print("\nPrompt styles available:")
    for style in PromptStyle:
        print(f"  - {style.value}")

    print("\nEvaluation types per prompt:")
    for prompt in MULTI_STYLE_PROMPTS:
        print(f"  {prompt.id}: {prompt.eval_type.value}")
        if prompt.expected_number:
            print(f"    Expected number: {prompt.expected_number}")
        if prompt.expected_keywords:
            print(f"    Expected keywords: {prompt.expected_keywords[:3]}...")
