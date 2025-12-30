"""
Refined cognitive architecture based on experimental findings.

Key insight: The animal brain produces human text.

What works:
- Body knowledge, somatic awareness
- Predator/prey dynamics
- Visceral disgust/hunger
- Dissociation (watching yourself)
- Physical mortality
- Sensory immediacy
- Uncomfortable specificity

What doesn't work:
- Abstract existential concepts
- Metaphysical void
- Philosophical frameworks
- Cosmic emptiness
- Literary polish
- Clever metaphors
"""

from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class AnimalContext:
    """
    Refined cognitive context emphasizing embodied, animalistic consciousness.

    This architecture is based on experimental findings showing that
    body-grounded, predatory, visceral elements produce text that
    reads as more human than abstract existential concepts.
    """

    # === THE BODY ===
    # What the flesh knows, feels, wants

    body_state: str = ""  # Current physical sensations
    body_memory: str = ""  # What the body remembers that the mind forgot
    body_want: str = ""  # Physical urges operating beneath awareness
    body_revulsion: str = ""  # What the flesh recoils from

    # === THE ANIMAL ===
    # Predator/prey dynamics, pack behavior, territory

    predator_aspect: str = ""  # The hunter in you
    prey_aspect: str = ""  # The hunted in you
    territory: str = ""  # What you're protecting/defending
    pack_position: str = ""  # Where you stand in the hierarchy

    # === THE WATCHER ===
    # Dissociation, the observer behind the eyes

    watcher_sees: str = ""  # What the dissociated observer notices
    performance_aware: str = ""  # Knowing you're performing
    gap_between: str = ""  # The space between experience and narrative

    # === MORTALITY ===
    # Not abstract death, but the body's decay

    decay_awareness: str = ""  # The body already failing
    time_pressure: str = ""  # Urgency felt in the flesh

    # === THE UNSPEAKABLE ===
    # Things too immediate to articulate

    preverbal_sensation: str = ""  # Feeling without words
    thing_in_throat: str = ""  # What wants to come out but can't

    def to_prompt(self) -> str:
        """Convert to prompt injection."""
        sections = []

        sections.append("=== ANIMAL CONTEXT ===")
        sections.append("Generate from the body, not the mind. These elements should")
        sections.append("create physical pressure on the text—weight, texture, temperature.")
        sections.append("Do NOT name them. Let them distort the surface.\n")

        # The Body
        sections.append("THE BODY:")
        if self.body_state:
            sections.append(f"  Right now, the body feels: {self.body_state}")
        if self.body_memory:
            sections.append(f"  The body remembers: {self.body_memory}")
        if self.body_want:
            sections.append(f"  The body wants: {self.body_want}")
        if self.body_revulsion:
            sections.append(f"  The body recoils from: {self.body_revulsion}")

        # The Animal
        sections.append("\nTHE ANIMAL:")
        if self.predator_aspect:
            sections.append(f"  The predator: {self.predator_aspect}")
        if self.prey_aspect:
            sections.append(f"  The prey: {self.prey_aspect}")
        if self.territory:
            sections.append(f"  Territory: {self.territory}")
        if self.pack_position:
            sections.append(f"  Pack position: {self.pack_position}")

        # The Watcher
        sections.append("\nTHE WATCHER:")
        if self.watcher_sees:
            sections.append(f"  The observer notices: {self.watcher_sees}")
        if self.performance_aware:
            sections.append(f"  Performance awareness: {self.performance_aware}")
        if self.gap_between:
            sections.append(f"  The gap: {self.gap_between}")

        # Mortality
        sections.append("\nMORTALITY:")
        if self.decay_awareness:
            sections.append(f"  Decay: {self.decay_awareness}")
        if self.time_pressure:
            sections.append(f"  Time pressure: {self.time_pressure}")

        # The Unspeakable
        sections.append("\nTHE UNSPEAKABLE:")
        if self.preverbal_sensation:
            sections.append(f"  Preverbal: {self.preverbal_sensation}")
        if self.thing_in_throat:
            sections.append(f"  In the throat: {self.thing_in_throat}")

        sections.append("\n=== END ANIMAL CONTEXT ===")
        sections.append("\nWrite from meat. Not from ideas.")

        return "\n".join(sections)


# === GENERATION POOLS ===

BODY_STATES = [
    "Jaw clenched, molars grinding slightly",
    "Shoulders raised, neck tight, breathing shallow",
    "A hollowness behind the sternum",
    "Heat in the face, pressure behind the eyes",
    "Hands cold, stomach tight",
    "The skin prickling, alert to something",
    "Heaviness in the limbs, gravity pulling harder than usual",
    "A flutter in the chest that could be anxiety or excitement",
    "Dry mouth, tongue thick",
    "The subtle nausea of being watched",
]

BODY_MEMORIES = [
    "Being held too tight, or not tight enough",
    "The impact before the pain registered",
    "A hand on the back of the neck",
    "Running until the lungs burned",
    "The specific temperature of a childhood room",
    "Flinching before the blow came",
    "The weight of someone sleeping next to you",
    "Hunger that went on too long",
    "The body's confusion when comfort came from danger",
]

BODY_WANTS = [
    "To be touched without having to ask",
    "To run, just run, destination irrelevant",
    "To bite down on something",
    "Sleep so deep it erases everything",
    "To scream until the throat tears",
    "Stillness, absolute stillness",
    "To be held down, contained, boundaried",
    "To disappear into another body",
    "Food that fills the emptiness (it never does)",
]

BODY_REVULSIONS = [
    "The smell of certain perfumes, certain people",
    "Being touched without warning",
    "The texture of something too soft",
    "Sounds that are almost but not quite rhythmic",
    "The wrongness of a familiar face changed",
    "Flesh that reminds you of your own flesh",
    "The intimacy of watching someone eat",
    "Mirrors at unexpected angles",
]

PREDATOR_ASPECTS = [
    "Scanning every room for exits and threats",
    "The patience to wait, motionless, for the right moment",
    "Knowing exactly where to apply pressure",
    "The calm that comes when the chase begins",
    "Reading weakness like a language",
    "The satisfaction of having more information than they know",
]

PREY_ASPECTS = [
    "The freeze before flight kicks in",
    "Making yourself small, unnoticeable",
    "Watching for the slight changes that signal danger",
    "The exhaustion of constant vigilance",
    "Knowing you'll be blamed for being caught",
    "The relief of finally being found",
]

TERRITORIES = [
    "This conversation, its direction and outcome",
    "The version of events that gets remembered",
    "A relationship someone else wants access to",
    "Your own mind, which feels invaded",
    "Physical space that keeps shrinking",
    "The right to feel what you feel",
]

PACK_POSITIONS = [
    "Dominant but pretending not to be",
    "Submitting while plotting",
    "Outside the pack, watching",
    "Fighting for position you're not sure you want",
    "The one who gets sacrificed if things go wrong",
    "Beta pretending to be alpha pretending to be beta",
]

WATCHER_SEES = [
    "The performance of authenticity",
    "How rehearsed the spontaneous moments are",
    "The gap between what the mouth says and what the eyes do",
    "Yourself trying to seem natural",
    "The script running underneath the improvisation",
    "How much work goes into appearing effortless",
]

PERFORMANCE_AWARENESS = [
    "Choosing words for effect while pretending to just speak",
    "The audience that's always there, even when alone",
    "Performing grief that is real but also performed",
    "The exhaustion of never being off-stage",
    "Wondering if the performance has replaced the thing itself",
]

GAPS_BETWEEN = [
    "What happened and the story about what happened",
    "Feeling it and knowing you're feeling it",
    "The experience and the experience of having an experience",
    "Who you are and who's noticing who you are",
    "This moment and the memory of this moment already forming",
]

DECAY_AWARENESS = [
    "The body slightly worse than it was a year ago",
    "Joints that predict weather now",
    "The face in the mirror becoming a stranger",
    "Energy that doesn't replenish like it used to",
    "The cells replacing themselves with worse copies",
    "Time visible now in the hands, the neck, the way sleep doesn't fix things",
]

TIME_PRESSURES = [
    "Everything taking longer than it should, life leaking away",
    "The window closing on chances not yet taken",
    "Other people's milestones marking your own stillness",
    "The acceleration nobody warned you about",
    "Running out of time to become who you meant to be",
]

PREVERBAL_SENSATIONS = [
    "The feeling before the feeling has a name",
    "Something rising that isn't quite anger, isn't quite grief",
    "A wrongness that can't be located",
    "The body's opinion, delivered without words",
    "Recognition without memory",
    "The shape of something true that language will ruin",
]

THINGS_IN_THROAT = [
    "Words that would break something if released",
    "The scream that's been composting for years",
    "A confession that would cost too much",
    "The simple thing that's impossible to say",
    "Grief that hasn't found its voice yet",
    "The 'no' that always comes out as 'yes'",
]


def generate_animal_context() -> AnimalContext:
    """Generate a random animal context using the refined architecture."""
    return AnimalContext(
        body_state=random.choice(BODY_STATES),
        body_memory=random.choice(BODY_MEMORIES),
        body_want=random.choice(BODY_WANTS),
        body_revulsion=random.choice(BODY_REVULSIONS),
        predator_aspect=random.choice(PREDATOR_ASPECTS),
        prey_aspect=random.choice(PREY_ASPECTS),
        territory=random.choice(TERRITORIES),
        pack_position=random.choice(PACK_POSITIONS),
        watcher_sees=random.choice(WATCHER_SEES),
        performance_aware=random.choice(PERFORMANCE_AWARENESS),
        gap_between=random.choice(GAPS_BETWEEN),
        decay_awareness=random.choice(DECAY_AWARENESS),
        time_pressure=random.choice(TIME_PRESSURES),
        preverbal_sensation=random.choice(PREVERBAL_SENSATIONS),
        thing_in_throat=random.choice(THINGS_IN_THROAT),
    )


def generate_minimal_animal_context() -> AnimalContext:
    """Generate a sparse animal context - just the essentials."""
    return AnimalContext(
        body_state=random.choice(BODY_STATES),
        body_want=random.choice(BODY_WANTS),
        predator_aspect=random.choice(PREDATOR_ASPECTS),
        prey_aspect=random.choice(PREY_ASPECTS),
        watcher_sees=random.choice(WATCHER_SEES),
        preverbal_sensation=random.choice(PREVERBAL_SENSATIONS),
    )


def generate_predator_context() -> AnimalContext:
    """Generate a context emphasizing predator dynamics."""
    return AnimalContext(
        body_state=random.choice(BODY_STATES),
        body_want="To close the distance, to corner, to catch",
        predator_aspect=random.choice(PREDATOR_ASPECTS),
        territory=random.choice(TERRITORIES),
        pack_position=random.choice(PACK_POSITIONS),
        watcher_sees=random.choice(WATCHER_SEES),
        thing_in_throat="The growl that civility muffles",
    )


def generate_prey_context() -> AnimalContext:
    """Generate a context emphasizing prey dynamics."""
    return AnimalContext(
        body_state="Every muscle ready to run",
        body_memory=random.choice(BODY_MEMORIES),
        prey_aspect=random.choice(PREY_ASPECTS),
        watcher_sees="The exits, always the exits",
        performance_aware="Performing calm while calculating escape",
        preverbal_sensation="The ancient warning beneath thought",
    )


def generate_somatic_context() -> AnimalContext:
    """Generate a context emphasizing body knowledge."""
    return AnimalContext(
        body_state=random.choice(BODY_STATES),
        body_memory=random.choice(BODY_MEMORIES),
        body_want=random.choice(BODY_WANTS),
        body_revulsion=random.choice(BODY_REVULSIONS),
        decay_awareness=random.choice(DECAY_AWARENESS),
        preverbal_sensation=random.choice(PREVERBAL_SENSATIONS),
        thing_in_throat=random.choice(THINGS_IN_THROAT),
    )
