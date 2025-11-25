"""
Test prompts comparing plain English vs Ailo-structured prompts.

Each test case has:
- A plain English prompt (control)
- An Ailo-structured prompt (test)
- Evaluation criteria to measure the difference
"""

from dataclasses import dataclass
from typing import Callable, Optional
from enum import Enum


class TaskCategory(Enum):
    WRITING = "writing"
    RESEARCH = "research"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    BUSINESS = "business"
    ANALYSIS = "analysis"


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating responses."""
    name: str
    description: str
    check: Callable[[str], bool]  # Function to check if criteria is met


@dataclass
class TestPrompt:
    """A test case comparing plain vs Ailo prompts."""
    id: str
    name: str
    category: TaskCategory
    description: str

    # The prompts
    plain_prompt: str
    ailo_prompt: str

    # Optional: expected characteristics
    expected_format: Optional[str] = None  # "bullets", "table", "code", etc.
    expected_length: Optional[str] = None  # "short", "medium", "long"

    # Evaluation criteria
    criteria: list[EvaluationCriteria] = None

    def __post_init__(self):
        if self.criteria is None:
            self.criteria = []


# ============================================================================
# Evaluation Helper Functions
# ============================================================================

def has_bullet_points(response: str) -> bool:
    """Check if response contains bullet points."""
    bullet_markers = ["•", "-", "*", "1.", "2.", "3."]
    lines = response.strip().split("\n")
    bullet_count = sum(1 for line in lines if any(line.strip().startswith(m) for m in bullet_markers))
    return bullet_count >= 3


def has_numbered_list(response: str) -> bool:
    """Check if response contains a numbered list."""
    import re
    pattern = r'^\s*\d+[\.\)]\s'
    lines = response.strip().split("\n")
    numbered_count = sum(1 for line in lines if re.match(pattern, line))
    return numbered_count >= 3


def has_table(response: str) -> bool:
    """Check if response contains a markdown table."""
    return "|" in response and "---" in response


def has_code_block(response: str) -> bool:
    """Check if response contains a code block."""
    return "```" in response or response.count("    ") >= 3


def word_count_range(min_words: int, max_words: int) -> Callable[[str], bool]:
    """Return a function that checks if word count is within range."""
    def check(response: str) -> bool:
        words = len(response.split())
        return min_words <= words <= max_words
    return check


def contains_sections(section_names: list[str]) -> Callable[[str], bool]:
    """Return a function that checks if response contains named sections."""
    def check(response: str) -> bool:
        response_lower = response.lower()
        found = sum(1 for name in section_names if name.lower() in response_lower)
        return found >= len(section_names) * 0.7  # 70% of sections found
    return check


def is_formal_tone(response: str) -> bool:
    """Check if response uses formal language (heuristic)."""
    informal_markers = ["gonna", "wanna", "gotta", "kinda", "sorta", "yeah", "nope", "cool", "awesome", "!!", "lol", "btw"]
    response_lower = response.lower()
    informal_count = sum(1 for marker in informal_markers if marker in response_lower)
    return informal_count == 0


def is_concise(response: str, max_words: int = 150) -> bool:
    """Check if response is concise."""
    return len(response.split()) <= max_words


# ============================================================================
# Test Prompts
# ============================================================================

TEST_PROMPTS = [
    # -------------------------------------------------------------------------
    # WRITING TASKS
    # -------------------------------------------------------------------------
    TestPrompt(
        id="write-01",
        name="Executive Summary",
        category=TaskCategory.WRITING,
        description="Test format compliance and conciseness for executive audience",
        plain_prompt="""Summarize the benefits of cloud computing for a business executive.
Keep it short and professional, maybe use bullet points.""",
        ailo_prompt="""CONTEXT = A CEO needs a quick overview before a board meeting
PERSONA = You are a technology consultant briefing a non-technical executive
MODE = Summarize
ACT = Summarize
OBJ = Benefits of cloud computing for enterprise
TAGS = [Format:Bullets, Length:5, Audience:Executive, Style:Professional, Constraints:No technical jargon]
OUTPUT = Text with bullet points""",
        expected_format="bullets",
        expected_length="short",
        criteria=[
            EvaluationCriteria("has_bullets", "Response uses bullet points", has_bullet_points),
            EvaluationCriteria("is_concise", "Response is under 150 words", is_concise),
            EvaluationCriteria("is_formal", "Response uses formal tone", is_formal_tone),
        ]
    ),

    TestPrompt(
        id="write-02",
        name="Technical Documentation",
        category=TaskCategory.TECHNICAL,
        description="Test structured technical writing with code examples",
        plain_prompt="""Explain how to use Python list comprehensions.
Include some code examples.""",
        ailo_prompt="""CONTEXT = Documentation for a developer learning Python
PERSONA = You are a senior Python developer writing beginner-friendly documentation
MODE = Explain
ACT = Explain
OBJ = Python list comprehensions
TAGS = [Format:Code+Explanation, Audience:Beginner developer, Length:Medium, Constraints:Include 3 progressive examples from simple to complex]
OUTPUT = Text with code blocks""",
        expected_format="code",
        criteria=[
            EvaluationCriteria("has_code", "Response includes code blocks", has_code_block),
            EvaluationCriteria("has_multiple_examples", "Response has multiple examples",
                              lambda r: r.count("```") >= 4 or r.count(">>>") >= 3),
        ]
    ),

    TestPrompt(
        id="write-03",
        name="Comparison Table",
        category=TaskCategory.ANALYSIS,
        description="Test ability to produce structured tabular output",
        plain_prompt="""Compare React, Vue, and Angular frameworks.
Maybe put it in a table format.""",
        ailo_prompt="""MODE = Compare
ACT = Compare
OBJ = React, Vue, and Angular JavaScript frameworks
TAGS = [Format:Table, Audience:Developer choosing a framework, Constraints:Include columns for learning curve, performance, ecosystem, and best use case]
OUTPUT = Markdown table""",
        expected_format="table",
        criteria=[
            EvaluationCriteria("has_table", "Response is formatted as table", has_table),
            EvaluationCriteria("all_frameworks", "All three frameworks mentioned",
                              lambda r: all(fw.lower() in r.lower() for fw in ["react", "vue", "angular"])),
        ]
    ),

    # -------------------------------------------------------------------------
    # RESEARCH TASKS
    # -------------------------------------------------------------------------
    TestPrompt(
        id="research-01",
        name="Market Research",
        category=TaskCategory.RESEARCH,
        description="Test structured research output with specific sections",
        plain_prompt="""Research the electric vehicle market.
What are the trends and who are the major players?""",
        ailo_prompt="""CONTEXT = Preparing an investment thesis for a venture capital firm
PERSONA = You are a market research analyst specializing in automotive technology
MODE = Research
ACT = Research
OBJ = Electric vehicle market landscape
TAGS = [Format:Structured report, Audience:Investors, Timeframe:2023-2025, Constraints:Include market size, growth rate, key players, and emerging trends]
OUTPUT = Structured text with sections""",
        criteria=[
            EvaluationCriteria("has_sections", "Response has clear sections",
                              contains_sections(["market", "players", "trends", "growth"])),
            EvaluationCriteria("is_formal", "Uses professional tone", is_formal_tone),
        ]
    ),

    TestPrompt(
        id="research-02",
        name="Competitive Analysis",
        category=TaskCategory.BUSINESS,
        description="Test structured business analysis",
        plain_prompt="""Do a competitive analysis of Slack vs Microsoft Teams vs Discord for business use.""",
        ailo_prompt="""CONTEXT = A startup is choosing a communication platform for a 50-person team
PERSONA = You are an IT consultant evaluating enterprise software
MODE = Analyze
ACT = Compare
OBJ = Slack, Microsoft Teams, and Discord for business communication
TAGS = [Format:Structured analysis, Audience:Startup CTO, Constraints:Focus on pricing, integrations, security, and scalability; Include a recommendation]
OUTPUT = Text with sections and final recommendation""",
        criteria=[
            EvaluationCriteria("all_platforms", "All platforms compared",
                              lambda r: all(p.lower() in r.lower() for p in ["slack", "teams", "discord"])),
            EvaluationCriteria("has_recommendation", "Includes a recommendation",
                              lambda r: any(word in r.lower() for word in ["recommend", "suggestion", "best choice", "winner"])),
        ]
    ),

    # -------------------------------------------------------------------------
    # CREATIVE TASKS
    # -------------------------------------------------------------------------
    TestPrompt(
        id="creative-01",
        name="Story Ideas",
        category=TaskCategory.CREATIVE,
        description="Test creative generation with specific constraints",
        plain_prompt="""Give me some sci-fi story ideas. Maybe 5 of them, and keep them short.""",
        ailo_prompt="""MODE = Generate
ACT = Generate ideas
OBJ = Science fiction short story concepts
TAGS = [Format:Numbered list, Length:5 ideas, Style:Cyberpunk and hard sci-fi mix, Constraints:Each idea should be 2-3 sentences with a unique hook]
OUTPUT = Numbered list""",
        expected_format="numbered_list",
        criteria=[
            EvaluationCriteria("has_five_ideas", "Exactly 5 ideas provided",
                              lambda r: 4 <= sum(1 for c in r if c.isdigit() and int(c) <= 5) <= 6),
            EvaluationCriteria("has_numbered_list", "Uses numbered format", has_numbered_list),
        ]
    ),

    TestPrompt(
        id="creative-02",
        name="Marketing Copy",
        category=TaskCategory.CREATIVE,
        description="Test persuasive writing with specific tone",
        plain_prompt="""Write some marketing copy for a new fitness app. Make it catchy.""",
        ailo_prompt="""CONTEXT = Launching a mobile fitness app targeting busy professionals aged 25-40
PERSONA = You are a copywriter at a top digital marketing agency
MODE = Generate
ACT = Write
OBJ = Marketing copy for fitness app launch
TAGS = [Format:Multiple variants, Length:3 versions, Style:Energetic and motivating, Audience:Busy professionals, Constraints:Include a headline, tagline, and 2-sentence description for each variant]
OUTPUT = Three distinct marketing copy variants""",
        criteria=[
            EvaluationCriteria("multiple_variants", "Multiple versions provided",
                              lambda r: any(marker in r.lower() for marker in ["version", "variant", "option", "1.", "2.", "3."])),
        ]
    ),

    # -------------------------------------------------------------------------
    # TECHNICAL/ANALYSIS TASKS
    # -------------------------------------------------------------------------
    TestPrompt(
        id="tech-01",
        name="Code Review",
        category=TaskCategory.TECHNICAL,
        description="Test structured code analysis",
        plain_prompt="""Review this Python code and tell me what's wrong:

def get_user(id):
    users = load_users()
    for user in users:
        if user['id'] == id:
            return user
    return None""",
        ailo_prompt="""CONTEXT = Code review for a junior developer's pull request
PERSONA = You are a senior software engineer conducting a thorough code review
MODE = Evaluate
ACT = Review
OBJ = Python function for user lookup
TAGS = [Format:Structured feedback, Constraints:Address performance, error handling, type hints, and naming conventions; Provide specific improvement suggestions]
INPUT = ```python
def get_user(id):
    users = load_users()
    for user in users:
        if user['id'] == id:
            return user
    return None
```
OUTPUT = Structured code review with categories and suggestions""",
        criteria=[
            EvaluationCriteria("has_code", "Includes code suggestions", has_code_block),
            EvaluationCriteria("mentions_issues", "Identifies multiple issues",
                              lambda r: sum(1 for issue in ["performance", "type", "naming", "error", "shadow", "id"]
                                           if issue in r.lower()) >= 2),
        ]
    ),

    TestPrompt(
        id="tech-02",
        name="Architecture Decision",
        category=TaskCategory.TECHNICAL,
        description="Test structured technical decision-making",
        plain_prompt="""Should I use a monolithic or microservices architecture for a new e-commerce platform?""",
        ailo_prompt="""CONTEXT = Technical architect at a startup building a new e-commerce platform expecting 10K users in year 1, scaling to 1M by year 3
PERSONA = You are a solutions architect with experience scaling e-commerce systems
MODE = Evaluate
ACT = Recommend
OBJ = Architecture choice between monolith and microservices
TAGS = [Format:Decision matrix, Audience:Technical stakeholders, Constraints:Consider team size (5 developers), timeline (6 months to MVP), and scaling requirements; Provide a clear recommendation with reasoning]
OUTPUT = Structured analysis with recommendation""",
        criteria=[
            EvaluationCriteria("has_recommendation", "Clear recommendation provided",
                              lambda r: any(word in r.lower() for word in ["recommend", "suggest", "should", "best", "choice"])),
            EvaluationCriteria("considers_tradeoffs", "Discusses tradeoffs",
                              lambda r: any(word in r.lower() for word in ["tradeoff", "trade-off", "however", "but", "downside", "drawback"])),
        ]
    ),

    # -------------------------------------------------------------------------
    # EDGE CASES - Testing specific Ailo features
    # -------------------------------------------------------------------------
    TestPrompt(
        id="edge-01",
        name="Persona Impact Test",
        category=TaskCategory.WRITING,
        description="Test if PERSONA changes the response style significantly",
        plain_prompt="""Explain what machine learning is.""",
        ailo_prompt="""PERSONA = You are a kindergarten teacher explaining things to 5-year-olds using simple words, fun analogies, and enthusiasm
ACT = Explain
OBJ = Machine learning
TAGS = [Audience:5-year-old children, Style:Fun and engaging, Constraints:Use only simple words, include a fun analogy with toys or animals]
OUTPUT = Child-friendly explanation""",
        criteria=[
            EvaluationCriteria("simple_language", "Uses simple vocabulary",
                              lambda r: not any(word in r.lower() for word in ["algorithm", "neural network", "optimization", "gradient", "parameter"])),
            EvaluationCriteria("has_analogy", "Contains analogy",
                              lambda r: any(word in r.lower() for word in ["like", "imagine", "pretend", "think of", "similar to"])),
        ]
    ),

    TestPrompt(
        id="edge-02",
        name="Length Constraint Test",
        category=TaskCategory.WRITING,
        description="Test if LENGTH tag is respected",
        plain_prompt="""List the benefits of meditation. Give me exactly 3 points.""",
        ailo_prompt="""ACT = List
OBJ = Benefits of meditation
TAGS = [Format:Bullets, Length:Exactly 3 points, Constraints:One sentence per point]
OUTPUT = 3 bullet points""",
        expected_length="exactly_3",
        criteria=[
            EvaluationCriteria("exactly_three", "Exactly 3 points provided",
                              lambda r: 2 <= sum(1 for line in r.split("\n") if line.strip().startswith(("-", "•", "*", "1", "2", "3"))) <= 4),
        ]
    ),
]


def get_prompts_by_category(category: TaskCategory) -> list[TestPrompt]:
    """Get all test prompts for a specific category."""
    return [p for p in TEST_PROMPTS if p.category == category]


def get_prompt_by_id(prompt_id: str) -> TestPrompt:
    """Get a specific test prompt by ID."""
    for prompt in TEST_PROMPTS:
        if prompt.id == prompt_id:
            return prompt
    raise ValueError(f"Prompt not found: {prompt_id}")


if __name__ == "__main__":
    print(f"Total test prompts: {len(TEST_PROMPTS)}")
    for category in TaskCategory:
        count = len(get_prompts_by_category(category))
        print(f"  {category.value}: {count}")
