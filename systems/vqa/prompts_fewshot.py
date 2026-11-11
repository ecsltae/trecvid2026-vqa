"""
Few-shot prompts for VQA MC task to improve answer selection.

Includes:
- Example questions with reasoning
- Visual reasoning patterns
- Common video understanding strategies
"""

# Few-shot examples for MC prompting
FEWSHOT_EXAMPLES = """
Example 1:
Question: What activity is the person doing in the video?
Options:
A) Cooking in a kitchen
B) Playing basketball
C) Reading a book
D) Dancing to music

Visual Analysis: I can see a person standing in front of a stove with pots on the burners. There are ingredients on the counter and the person is holding a spatula. The setting is clearly a kitchen with cooking equipment visible.

Evaluation:
A) Cooking in a kitchen - MATCHES all visual evidence (kitchen, stove, spatula, ingredients)
B) Playing basketball - No basketball or court visible
C) Reading a book - No book visible, person is actively moving with cooking tools
D) Dancing to music - While the person may be moving, the primary activity is clearly cooking given the context

RANKING: A,D,C,B

Example 2:
Question: What type of animal appears in the video?
Options:
A) A domestic cat
B) A wild tiger
C) A small dog
D) A bird

Visual Analysis: I can see a four-legged animal with fur, a tail, and pointed ears. The animal is small to medium-sized, appears to be indoors on a carpet, and has a collar. The fur pattern shows stripes or spots typical of domestic cats.

Evaluation:
A) A domestic cat - STRONG MATCH: indoor setting, collar, size, fur pattern, behavior
B) A wild tiger - Size too small, indoor setting inappropriate, has collar
C) A small dog - Could fit some visual cues (collar, indoor) but body shape and movement suggest feline
D) A bird - Completely wrong: has four legs, fur, no wings

RANKING: A,C,B,D

Example 3:
Question: What is the weather condition shown?
Options:
A) Clear and sunny
B) Heavy rain
C) Snowy conditions
D) Foggy morning

Visual Analysis: The video shows bright lighting, strong shadows on the ground, blue sky visible, and people wearing light clothing like t-shirts and shorts. No precipitation or clouds visible. High contrast in the image suggests strong sunlight.

Evaluation:
A) Clear and sunny - PERFECT MATCH: bright light, shadows, blue sky, light clothing
B) Heavy rain - No rain visible, people not wearing rain gear, dry ground
C) Snowy conditions - No snow, people in summer clothes, would see white ground
D) Foggy morning - No fog, clear visibility, strong shadows indicate direct sunlight

RANKING: A,D,B,C
"""

MC_FEWSHOT_SYSTEM = (
    "You are a video understanding assistant. Analyze the video frames carefully, "
    "describe what you observe, evaluate each answer option based on visual evidence, "
    "then provide a ranking from most to least likely correct."
)

MC_FEWSHOT_TMPL = f"""\
{FEWSHOT_EXAMPLES}

Now analyze this question:

Question: {{question}}

Options:
A) {{opt_a}}
B) {{opt_b}}
C) {{opt_c}}
D) {{opt_d}}

Follow the same format:
1. Visual Analysis: Describe key observations from the video frames
2. Evaluation: Assess each option (A-D) against the visual evidence
3. RANKING: Provide the four letters A-D ordered from best to worst match

Your response:
"""
