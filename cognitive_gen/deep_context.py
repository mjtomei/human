"""Deep cognitive context - the subconscious layers we can't fully articulate."""

from dataclasses import dataclass, field
from typing import Optional
import random

from .context import CognitiveContext


@dataclass
class DeepCognitiveContext(CognitiveContext):
    """
    Extended cognitive context reaching into primal, pre-verbal layers.

    These are the drives and fears that operate beneath conscious awareness -
    the animal brain, the death awareness, the desperate need for connection
    that we dress up in acceptable language.
    """

    # Primal drives we rationalize but don't control
    primal_drives: list[str] = field(default_factory=list)

    # The awareness of mortality that colors everything
    death_awareness: str = ""

    # The desperate, animal need for connection/belonging
    belonging_hunger: str = ""

    # Shame so deep it has no narrative, just sensation
    preverbal_shame: str = ""

    # The part that watches yourself perform being a person
    dissociated_observer: str = ""

    # What the body knows that the mind denies
    somatic_knowledge: str = ""

    # The void - the formless dread beneath constructed meaning
    existential_substrate: str = ""

    # Memories that aren't memories - inherited fears, collective weight
    ancestral_residue: str = ""

    # The thing you're always running from without knowing it
    unnamed_pursuer: str = ""

    # What you'd be if you stopped performing
    feral_self: str = ""

    def to_prompt(self) -> str:
        """Convert deep context to prompt, including subconscious layers."""
        sections = []

        sections.append("=== DEEP COGNITIVE CONTEXT ===")
        sections.append("You are writing FROM this person's complete experience - including")
        sections.append("the layers they cannot articulate or admit to themselves. These")
        sections.append("subconscious elements should create pressure on the surface text,")
        sections.append("distorting it in subtle ways, without ever being named.\n")

        # Surface layer (inherited from parent)
        sections.append("-- CONSCIOUS LAYER --")
        if self.explicit_goals:
            sections.append(f"Goals: {', '.join(self.explicit_goals)}")
        if self.emotional_state:
            sections.append(f"Emotional state: {self.emotional_state}")
        if self.self_image:
            sections.append(f"Self-image: {self.self_image}")

        # Liminal layer
        sections.append("\n-- LIMINAL LAYER (half-known) --")
        if self.hidden_motivations:
            sections.append(f"Hidden motivations: {', '.join(self.hidden_motivations)}")
        if self.anxieties:
            sections.append(f"Active anxieties: {', '.join(self.anxieties)}")
        if self.internal_conflicts:
            sections.append(f"Internal conflicts: {', '.join(self.internal_conflicts)}")
        if self.insecurities:
            sections.append(f"Insecurities: {', '.join(self.insecurities)}")

        # Deep layer
        sections.append("\n-- SUBCONSCIOUS LAYER (unknown to self) --")
        if self.primal_drives:
            sections.append(f"Primal drives operating beneath awareness:")
            for drive in self.primal_drives:
                sections.append(f"  - {drive}")

        if self.death_awareness:
            sections.append(f"\nDeath awareness: {self.death_awareness}")

        if self.belonging_hunger:
            sections.append(f"\nBelonging hunger: {self.belonging_hunger}")

        if self.preverbal_shame:
            sections.append(f"\nPreverbal shame: {self.preverbal_shame}")

        if self.dissociated_observer:
            sections.append(f"\nDissociated observer: {self.dissociated_observer}")

        if self.somatic_knowledge:
            sections.append(f"\nSomatic knowledge: {self.somatic_knowledge}")

        # Abyss layer
        sections.append("\n-- THE ABYSS (the unknowable) --")
        if self.existential_substrate:
            sections.append(f"Existential substrate: {self.existential_substrate}")

        if self.ancestral_residue:
            sections.append(f"\nAncestral residue: {self.ancestral_residue}")

        if self.unnamed_pursuer:
            sections.append(f"\nThe unnamed pursuer: {self.unnamed_pursuer}")

        if self.feral_self:
            sections.append(f"\nThe feral self: {self.feral_self}")

        sections.append("\n=== END DEEP CONTEXT ===")
        sections.append("\nLet all of this create PRESSURE on the text without surfacing.")
        sections.append("The words should feel heavy with what they're not saying.")

        return "\n".join(sections)


# Pools for deep context generation

PRIMAL_DRIVES = [
    "The need to be seen as special, chosen, different from the mass",
    "Territorial aggression dressed up as principle",
    "Sexual undercurrents in all intimacy, even when inappropriate",
    "The urge to dominate or submit that organizes all relationships",
    "Disgust at bodies, including your own, beneath performed acceptance",
    "The compulsion to hoard - attention, resources, love",
    "Envy so constant it feels like atmosphere",
    "The predator's watchfulness, scanning for weakness",
    "The prey's hypervigilance, waiting for the attack",
    "Rage at being contained in a single, mortal body",
    "The drive to merge with another and dissolve the self",
    "Revulsion at dependency, even while craving it",
]

DEATH_AWARENESS = [
    "Every interaction shadowed by the knowledge this person will die",
    "The body already beginning its slow betrayal",
    "Time passing felt as physical pressure, a weight on the chest",
    "The impossibility of non-existence making all meaning feel thin",
    "Others' bodies as reminders - they too will become nothing",
    "The future collapsing into the present, urgency without outlet",
    "Legacy anxiety - the desperate need to leave a mark",
    "The suspicion that consciousness is already mostly gone, only the tail end remaining",
]

BELONGING_HUNGER = [
    "The infant's terror of abandonment, never fully outgrown",
    "Performing connection while feeling fundamentally alone",
    "The suspicion that others have a secret warmth you were denied",
    "Love as a demand disguised as offering",
    "The exhaustion of maintaining bonds that feel like work",
    "Fantasies of being truly known warring with terror of exposure",
    "Jealousy of others' ease with each other",
    "The pack animal's fear of exile, dressed in adult language",
]

PREVERBAL_SHAME = [
    "A wrongness that predates memory, located in the body",
    "The sense of having taken up space that wasn't yours",
    "Contamination that can't be washed away",
    "The original wound before the story you tell about it",
    "Existing as an imposition, an inconvenience to others",
    "The body as evidence of something shameful",
    "A core deficiency that all achievement is trying to hide",
]

DISSOCIATED_OBSERVER = [
    "Watching yourself speak as if operating a puppet",
    "The one who notices you noticing yourself",
    "Performing emotions rather than having them",
    "The gap between experience and the narrative about experience",
    "Wondering if others have this watcher too, or if you're broken",
    "The suspicion that the 'real you' died long ago and this is just momentum",
]

SOMATIC_KNOWLEDGE = [
    "The stomach knows before the mind admits",
    "Tension held in shoulders for decades",
    "The body bracing for impact that already happened",
    "Nausea as moral compass",
    "The jaw clenched against words that want to escape",
    "Arousal and anxiety as the same sensation, differently labeled",
    "The body keeping score of everything the mind forgot",
]

EXISTENTIAL_SUBSTRATE = [
    "The void that meaning is painted over",
    "The suspicion that consciousness is an accident, not a gift",
    "The vertigo of infinite space, even in small rooms",
    "The constructed nature of self, threatening to dissolve",
    "Time as a wound rather than a medium",
    "The absurdity of caring about anything, felt but not livable",
    "The silence beneath all internal monologue",
]

ANCESTRAL_RESIDUE = [
    "Fears that don't match this life - water, falling, being eaten",
    "Grief for losses you never experienced",
    "The weight of generations of survival, pressing down",
    "Patterns repeated without knowing their origin",
    "The dead continuing to speak through your gestures",
    "Inherited trauma wearing the mask of personality",
]

UNNAMED_PURSUER = [
    "The thing you're always one step ahead of, without knowing what it is",
    "The collapse that feels always imminent",
    "What would happen if you stopped moving",
    "The judgment that's always coming",
    "The exposure that feels inevitable",
    "The abandonment you're preemptively grieving",
]

FERAL_SELF = [
    "The one who would bite, given permission",
    "What remains when performance stops",
    "The animal that wants to run, just run",
    "The child before it learned to hide itself",
    "The howl that civilization muffled",
    "Appetites that have no acceptable form",
]


def generate_deep_context(text_type: str = "personal_essay") -> DeepCognitiveContext:
    """Generate a deep cognitive context with primal/subconscious layers."""

    # Import parent context generation for surface layer
    from .context import (
        GOAL_POOLS, HIDDEN_MOTIVATIONS, ANXIETIES, SELF_IMAGES,
        INSECURITIES, EMOTIONAL_STATES, INTERNAL_CONFLICTS,
        _generate_situational_awareness, _generate_social_positioning
    )

    goals = GOAL_POOLS.get(text_type, GOAL_POOLS["personal_essay"])

    return DeepCognitiveContext(
        # Surface layer
        explicit_goals=random.sample(goals, k=random.randint(1, 2)),
        emotional_state=random.choice(EMOTIONAL_STATES),
        self_image=random.choice(SELF_IMAGES),
        situational_awareness=_generate_situational_awareness(text_type),
        social_positioning=_generate_social_positioning(text_type),

        # Liminal layer
        hidden_motivations=random.sample(HIDDEN_MOTIVATIONS, k=random.randint(1, 2)),
        anxieties=random.sample(ANXIETIES, k=random.randint(1, 2)),
        internal_conflicts=random.sample(INTERNAL_CONFLICTS, k=1),
        insecurities=random.sample(INSECURITIES, k=random.randint(1, 2)),

        # Subconscious layer
        primal_drives=random.sample(PRIMAL_DRIVES, k=random.randint(2, 3)),
        death_awareness=random.choice(DEATH_AWARENESS),
        belonging_hunger=random.choice(BELONGING_HUNGER),
        preverbal_shame=random.choice(PREVERBAL_SHAME),
        dissociated_observer=random.choice(DISSOCIATED_OBSERVER),
        somatic_knowledge=random.choice(SOMATIC_KNOWLEDGE),

        # Abyss layer
        existential_substrate=random.choice(EXISTENTIAL_SUBSTRATE),
        ancestral_residue=random.choice(ANCESTRAL_RESIDUE),
        unnamed_pursuer=random.choice(UNNAMED_PURSUER),
        feral_self=random.choice(FERAL_SELF),
    )


def generate_minimal_deep_context(text_type: str = "personal_essay") -> DeepCognitiveContext:
    """Generate a minimal deep context - just the abyss, no surface."""

    return DeepCognitiveContext(
        # Skip surface entirely
        explicit_goals=[],
        emotional_state="",
        self_image="",

        # Minimal liminal
        hidden_motivations=[],
        anxieties=[],

        # Core subconscious only
        primal_drives=random.sample(PRIMAL_DRIVES, k=2),
        death_awareness=random.choice(DEATH_AWARENESS),
        belonging_hunger=random.choice(BELONGING_HUNGER),
        preverbal_shame=random.choice(PREVERBAL_SHAME),
        somatic_knowledge=random.choice(SOMATIC_KNOWLEDGE),

        # Full abyss
        existential_substrate=random.choice(EXISTENTIAL_SUBSTRATE),
        ancestral_residue=random.choice(ANCESTRAL_RESIDUE),
        unnamed_pursuer=random.choice(UNNAMED_PURSUER),
        feral_self=random.choice(FERAL_SELF),
    )
