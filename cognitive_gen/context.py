"""Cognitive context model representing the subterranean mental state of a persona."""

from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class CognitiveContext:
    """
    Represents the rich cognitive state underlying human text production.

    This models the "iceberg" beneath written text - the goals, anxieties,
    hidden motivations, and self-perceptions that subtly influence word choice,
    tone, and structure even when not explicitly mentioned.
    """

    # What the person consciously wants to achieve with this text
    explicit_goals: list[str] = field(default_factory=list)

    # Desires they may not fully admit to themselves or others
    hidden_motivations: list[str] = field(default_factory=list)

    # Active worries that color their communication
    anxieties: list[str] = field(default_factory=list)

    # How they see themselves (may not match reality)
    self_image: str = ""

    # Vulnerabilities that may leak through in subtle ways
    insecurities: list[str] = field(default_factory=list)

    # Current emotional state / mood
    emotional_state: str = ""

    # Understanding of the situation and audience
    situational_awareness: str = ""

    # How they want to be perceived by the reader
    social_positioning: str = ""

    # Competing internal desires that create tension
    internal_conflicts: list[str] = field(default_factory=list)

    # Recent experiences coloring their current state
    recent_experiences: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert the cognitive context to a prompt injection for the LLM."""
        sections = []

        sections.append("=== COGNITIVE CONTEXT (DO NOT MENTION DIRECTLY) ===")
        sections.append("You are writing AS this person. Let these elements subtly influence")
        sections.append("your word choice, tone, and what you leave unsaid - but do NOT")
        sections.append("explicitly reference them in the text.\n")

        if self.explicit_goals:
            sections.append(f"EXPLICIT GOALS (what you consciously want):")
            for goal in self.explicit_goals:
                sections.append(f"  - {goal}")

        if self.hidden_motivations:
            sections.append(f"\nHIDDEN MOTIVATIONS (desires you don't fully admit):")
            for motivation in self.hidden_motivations:
                sections.append(f"  - {motivation}")

        if self.anxieties:
            sections.append(f"\nACTIVE ANXIETIES (worries coloring your communication):")
            for anxiety in self.anxieties:
                sections.append(f"  - {anxiety}")

        if self.self_image:
            sections.append(f"\nSELF-IMAGE: {self.self_image}")

        if self.insecurities:
            sections.append(f"\nINSECURITIES (vulnerabilities that may leak through):")
            for insecurity in self.insecurities:
                sections.append(f"  - {insecurity}")

        if self.emotional_state:
            sections.append(f"\nCURRENT EMOTIONAL STATE: {self.emotional_state}")

        if self.situational_awareness:
            sections.append(f"\nSITUATIONAL AWARENESS: {self.situational_awareness}")

        if self.social_positioning:
            sections.append(f"\nSOCIAL POSITIONING (how you want to be seen): {self.social_positioning}")

        if self.internal_conflicts:
            sections.append(f"\nINTERNAL CONFLICTS (competing desires creating tension):")
            for conflict in self.internal_conflicts:
                sections.append(f"  - {conflict}")

        if self.recent_experiences:
            sections.append(f"\nRECENT EXPERIENCES (coloring current state):")
            for exp in self.recent_experiences:
                sections.append(f"  - {exp}")

        sections.append("\n=== END COGNITIVE CONTEXT ===")

        return "\n".join(sections)


# Pools of realistic context elements for random generation
GOAL_POOLS = {
    "personal_essay": [
        "Express something meaningful that I've been carrying",
        "Make sense of an experience that still confuses me",
        "Connect with whoever reads this",
        "Prove to myself that I've grown",
        "Document this before I forget the details",
        "Process grief or loss through writing",
        "Celebrate something without seeming boastful",
    ],
    "creative_fiction": [
        "Write something that would impress my favorite author",
        "Explore a feeling I can't express directly",
        "Create a character more interesting than myself",
        "Work through a scenario I'm afraid of",
        "Prove I have creative talent",
        "Escape into another world for a while",
    ],
    "email": [
        "Reconnect without seeming desperate",
        "Apologize without losing face",
        "Get information I need",
        "Maintain the relationship",
        "Clear the air about something",
        "Make plans without overcommitting",
    ],
    "message": [
        "Smooth things over quickly",
        "Express how I feel without drama",
        "Get a response",
        "Not seem too needy or too cold",
        "Keep it light but meaningful",
        "Test if they're still upset",
    ],
}

HIDDEN_MOTIVATIONS = [
    "Wanting validation and approval",
    "Fear of being forgotten or irrelevant",
    "Desire to seem more put-together than I am",
    "Hoping this will change how they see me",
    "Proving something to an imaginary critic",
    "Seeking connection I'm afraid to ask for directly",
    "Testing if this relationship is worth maintaining",
    "Avoiding a harder conversation",
    "Wanting to be seen as deep or thoughtful",
    "Fear of vulnerability driving performance",
    "Competing with a version of myself that doesn't exist",
    "Hoping to manufacture closure",
]

ANXIETIES = [
    "What if this comes across wrong?",
    "They might think I'm being fake",
    "I'm probably overthinking this",
    "What if they don't respond?",
    "Am I being too much? Or not enough?",
    "They'll see through me",
    "This might ruin something",
    "I don't actually know what I want to say",
    "Time is passing and I haven't done this",
    "Other people do this so much better",
    "What if I regret sending this?",
    "I'm not sure I mean what I'm about to write",
]

SELF_IMAGES = [
    "Someone who's thoughtful but sometimes overthinks",
    "A person who cares deeply but struggles to show it",
    "Someone who's been through things and come out wiser",
    "A creative person who doesn't create enough",
    "Someone better at listening than talking",
    "A person who's trying to be more open",
    "Someone who's changed a lot recently",
    "A person who values authenticity above all",
    "Someone who's harder on themselves than others",
    "A person still figuring things out",
]

INSECURITIES = [
    "My writing isn't as good as I think it is",
    "I take too long to do simple things",
    "People probably don't think about me as much as I think about them",
    "I might be more forgettable than I realize",
    "My emotions are sometimes too visible",
    "I come across as trying too hard",
    "I'm not as interesting as the image I project",
    "My self-awareness is actually self-absorption",
    "I use big words to hide shallow thoughts",
    "I'm afraid of direct communication",
]

EMOTIONAL_STATES = [
    "Slightly anxious but trying to project calm",
    "Nostalgic and a bit melancholy",
    "Cautiously hopeful",
    "Tired but pushing through",
    "Irritated but suppressing it",
    "Genuinely warm but self-conscious about it",
    "Detached, observing myself from outside",
    "Nervous energy looking for an outlet",
    "Quiet contentment with an edge of worry",
    "Feeling exposed and trying to compensate",
]

INTERNAL_CONFLICTS = [
    "Want to be honest vs. want to protect myself",
    "Want to connect vs. fear of rejection",
    "Want to move on vs. want acknowledgment first",
    "Want to seem casual vs. want to be taken seriously",
    "Want to express feelings vs. fear of seeming dramatic",
    "Want closure vs. want to leave the door open",
    "Want to be vulnerable vs. want to stay in control",
    "Want to help vs. fear of overstepping",
    "Want to be authentic vs. want to be liked",
]

RECENT_EXPERIENCES = [
    "Had a conversation that didn't go how I expected",
    "Saw something that reminded me of the past",
    "Received news that's still sinking in",
    "Made a small mistake I'm still thinking about",
    "Had a moment of unexpected clarity",
    "Felt disconnected from friends lately",
    "Accomplished something but it felt hollow",
    "Had a dream that stuck with me",
    "Noticed a pattern in my behavior",
    "Felt time moving faster than usual",
]


def generate_random_context(text_type: str = "personal_essay") -> CognitiveContext:
    """
    Generate a realistic random cognitive context for a persona.

    Args:
        text_type: One of "personal_essay", "creative_fiction", "email", "message"

    Returns:
        A CognitiveContext with randomly selected but coherent elements
    """
    goals = GOAL_POOLS.get(text_type, GOAL_POOLS["personal_essay"])

    return CognitiveContext(
        explicit_goals=random.sample(goals, k=random.randint(1, 3)),
        hidden_motivations=random.sample(HIDDEN_MOTIVATIONS, k=random.randint(1, 3)),
        anxieties=random.sample(ANXIETIES, k=random.randint(2, 4)),
        self_image=random.choice(SELF_IMAGES),
        insecurities=random.sample(INSECURITIES, k=random.randint(1, 3)),
        emotional_state=random.choice(EMOTIONAL_STATES),
        situational_awareness=_generate_situational_awareness(text_type),
        social_positioning=_generate_social_positioning(text_type),
        internal_conflicts=random.sample(INTERNAL_CONFLICTS, k=random.randint(1, 2)),
        recent_experiences=random.sample(RECENT_EXPERIENCES, k=random.randint(1, 2)),
    )


def _generate_situational_awareness(text_type: str) -> str:
    """Generate situational awareness appropriate to the text type."""
    awareness_map = {
        "personal_essay": [
            "Writing this late at night when thoughts feel more honest",
            "Taking time to finally put this into words",
            "Not sure who will read this, if anyone",
            "This might sit in drafts forever",
        ],
        "creative_fiction": [
            "Trying to write something meaningful in stolen moments",
            "This story has been in my head for a while",
            "Not sure where this is going yet",
            "Writing to see what comes out",
        ],
        "email": [
            "It's been too long since we talked",
            "I've drafted versions of this before",
            "They might be busy and not respond quickly",
            "Our last conversation ended awkwardly",
        ],
        "message": [
            "They're probably looking at their phone",
            "I should have sent this earlier",
            "This might interrupt something",
            "I don't know what mood they're in",
        ],
    }
    options = awareness_map.get(text_type, awareness_map["personal_essay"])
    return random.choice(options)


def _generate_social_positioning(text_type: str) -> str:
    """Generate social positioning appropriate to the text type."""
    positioning_map = {
        "personal_essay": [
            "Want to seem reflective without being self-indulgent",
            "Want to be seen as someone who's learned from experience",
            "Want to come across as honest but not oversharing",
            "Want to seem insightful without being pretentious",
        ],
        "creative_fiction": [
            "Want to seem like a real writer, not just someone who writes",
            "Want to show depth without being heavy-handed",
            "Want to be original but still accessible",
            "Want to create something worth reading",
        ],
        "email": [
            "Want to seem warm but not desperate for connection",
            "Want to come across as having my life together",
            "Want to seem like I've been meaning to reach out",
            "Want to appear casual but genuine",
        ],
        "message": [
            "Want to seem like this isn't a big deal",
            "Want to come across as mature about it",
            "Want to seem relaxed even if I'm not",
            "Want to leave room for them to respond how they want",
        ],
    }
    options = positioning_map.get(text_type, positioning_map["personal_essay"])
    return random.choice(options)
