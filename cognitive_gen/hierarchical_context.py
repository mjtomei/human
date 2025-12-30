"""
Hierarchical cognitive architecture spanning universe to text.

The hypothesis: neither pure abstraction nor pure embodiment alone,
but a hierarchy where cosmic pressure flows down through layers
and manifests in the body, creating text that carries the weight
of all levels simultaneously.

Structure:
  COSMOS → EXISTENCE → SPECIES → TRIBE → SELF → BODY → MOMENT → TEXT

Each layer compresses into the one below, so by the time we reach
the text, it carries implicit pressure from all levels above.
"""

from dataclasses import dataclass, field
from typing import Optional
import random


@dataclass
class HierarchicalContext:
    """
    A hierarchy spanning from cosmic to immediate, where each level
    presses down on the one below.
    """

    # === LEVEL 1: COSMOS ===
    # The universe, entropy, deep time
    cosmos: str = ""

    # === LEVEL 2: EXISTENCE ===
    # Being conscious in an indifferent universe
    existence: str = ""

    # === LEVEL 3: SPECIES ===
    # Being human, animal heritage, evolutionary pressures
    species: str = ""

    # === LEVEL 4: TRIBE ===
    # Social belonging, hierarchy, inclusion/exclusion
    tribe: str = ""

    # === LEVEL 5: SELF ===
    # Individual identity, the constructed "I"
    self_layer: str = ""

    # === LEVEL 6: BODY ===
    # Physical sensations, flesh, immediate somatic experience
    body: str = ""

    # === LEVEL 7: MOMENT ===
    # This exact instant, what's happening now
    moment: str = ""

    def to_prompt(self) -> str:
        """Convert to a hierarchical prompt that shows the pressure cascade."""
        sections = []

        sections.append("=== HIERARCHICAL CONTEXT: COSMOS TO MOMENT ===")
        sections.append("")
        sections.append("Each level presses down on the one below. The text emerges")
        sections.append("from the bottom, carrying the weight of everything above.")
        sections.append("Do NOT name these levels. Let them create pressure, not content.")
        sections.append("")

        sections.append("┌─────────────────────────────────────────────┐")
        if self.cosmos:
            sections.append(f"│ COSMOS: {self.cosmos[:50]}")
        sections.append("│                     ↓                       │")
        if self.existence:
            sections.append(f"│ EXISTENCE: {self.existence[:47]}")
        sections.append("│                     ↓                       │")
        if self.species:
            sections.append(f"│ SPECIES: {self.species[:49]}")
        sections.append("│                     ↓                       │")
        if self.tribe:
            sections.append(f"│ TRIBE: {self.tribe[:51]}")
        sections.append("│                     ↓                       │")
        if self.self_layer:
            sections.append(f"│ SELF: {self.self_layer[:52]}")
        sections.append("│                     ↓                       │")
        if self.body:
            sections.append(f"│ BODY: {self.body[:52]}")
        sections.append("│                     ↓                       │")
        if self.moment:
            sections.append(f"│ MOMENT: {self.moment[:50]}")
        sections.append("│                     ↓                       │")
        sections.append("│                   TEXT                      │")
        sections.append("└─────────────────────────────────────────────┘")

        sections.append("")
        sections.append("The cosmic manifests in the body. The body produces the text.")
        sections.append("=== END HIERARCHICAL CONTEXT ===")

        return "\n".join(sections)


# === GENERATION POOLS ===

# Level 1: Cosmos - the universe, entropy, deep time
COSMOS_POOL = [
    "Entropy always wins; all structure is temporary resistance",
    "The universe is 13.8 billion years of matter briefly noticing itself",
    "Heat death approaching; every warm thing cooling toward equilibrium",
    "Space expanding faster than light; everything drifting apart forever",
    "Stars burning through their fuel; even suns are mortal",
    "The cosmic silence that swallows all signals eventually",
    "Matter assembled by chance, disassembled by certainty",
    "The vast indifference that doesn't even rise to cruelty",
]

# Level 2: Existence - consciousness in an indifferent universe
EXISTENCE_POOL = [
    "Awareness as a brief flicker between two infinite darknesses",
    "The absurdity of caring about anything, felt but not livable",
    "Consciousness as wound—to know is to suffer the knowing",
    "Meaning as necessary fiction, constructed against the void",
    "The weight of existing without having chosen to exist",
    "Being the universe looking at itself, briefly, before forgetting",
    "The loneliness of being the only witness to your experience",
    "Trapped in subjectivity, unable to verify anything outside it",
]

# Level 3: Species - being human, animal heritage
SPECIES_POOL = [
    "200,000 years of human fear encoded in the nervous system",
    "The animal body wearing clothes, pretending civilization",
    "Evolved for savanna dangers, stuck in fluorescent boxes",
    "The mammalian need for touch that shame has complicated",
    "Tribal brain in a global world, overwhelmed by scale",
    "The predator and prey still running beneath every thought",
    "Language as recent software on ancient hardware",
    "The body still preparing for famines that won't come",
]

# Level 4: Tribe - social belonging, hierarchy
TRIBE_POOL = [
    "The constant calculation of who's in and who's out",
    "Status anxiety inherited from ancestors who died without the group",
    "The exhaustion of maintaining social bonds that feel like work",
    "Pack position always uncertain, always being negotiated",
    "The performance required to remain included",
    "Gossip as survival mechanism, reputation as life or death",
    "The way groups turn on members without warning",
    "Belonging as a debt that must be continuously repaid",
]

# Level 5: Self - individual identity
SELF_POOL = [
    "The 'I' as a story the brain tells to organize chaos",
    "Identity as a defensive structure, built to hide the void",
    "The self that watches the self, infinite regress of observers",
    "Personality as scar tissue from early wounds",
    "The exhausting work of maintaining a consistent character",
    "Who you are when no one is watching (do you exist then?)",
    "The gap between who you are and who you perform",
    "The self as negotiation between animal drives and social demands",
]

# Level 6: Body - physical, immediate, somatic
BODY_POOL = [
    "Jaw clenched, holding words that want to escape",
    "The stomach's opinion, delivered before the mind decides",
    "Shoulders carrying tension that has no narrative",
    "The heart rate that betrays what the face hides",
    "Skin prickling with awareness of being perceived",
    "The heaviness in limbs when the body knows before the mind",
    "Breath shallow, chest tight, the body bracing",
    "Heat rising to the face, the flesh's honesty",
]

# Level 7: Moment - this exact instant
MOMENT_POOL = [
    "This breath, this heartbeat, this word forming",
    "The light in the room right now, the sounds underneath silence",
    "The specific weight of the air, the temperature on skin",
    "What the hands are doing while the mind is elsewhere",
    "The next second approaching, inevitable and unknown",
    "This exact configuration of atoms that will never repeat",
    "The words about to be written, still unformed",
    "Now. Now. Now. The relentless present.",
]


def generate_hierarchical_context() -> HierarchicalContext:
    """Generate a full hierarchical context from cosmos to moment."""
    return HierarchicalContext(
        cosmos=random.choice(COSMOS_POOL),
        existence=random.choice(EXISTENCE_POOL),
        species=random.choice(SPECIES_POOL),
        tribe=random.choice(TRIBE_POOL),
        self_layer=random.choice(SELF_POOL),
        body=random.choice(BODY_POOL),
        moment=random.choice(MOMENT_POOL),
    )


def generate_compressed_hierarchy() -> HierarchicalContext:
    """Generate a compressed hierarchy - cosmos, species, body, moment only."""
    return HierarchicalContext(
        cosmos=random.choice(COSMOS_POOL),
        species=random.choice(SPECIES_POOL),
        body=random.choice(BODY_POOL),
        moment=random.choice(MOMENT_POOL),
    )


def generate_grounded_cosmos() -> HierarchicalContext:
    """Cosmos that terminates in body - maximum span, grounded."""
    return HierarchicalContext(
        cosmos=random.choice(COSMOS_POOL),
        existence=random.choice(EXISTENCE_POOL),
        body=random.choice(BODY_POOL),
        moment=random.choice(MOMENT_POOL),
    )


def generate_social_to_body() -> HierarchicalContext:
    """Tribe through body - social pressure made physical."""
    return HierarchicalContext(
        tribe=random.choice(TRIBE_POOL),
        self_layer=random.choice(SELF_POOL),
        body=random.choice(BODY_POOL),
        moment=random.choice(MOMENT_POOL),
    )


def generate_species_body() -> HierarchicalContext:
    """Just species and body - evolutionary pressure in the flesh."""
    return HierarchicalContext(
        species=random.choice(SPECIES_POOL),
        body=random.choice(BODY_POOL),
        moment=random.choice(MOMENT_POOL),
    )


# === EXPERIMENTAL: Explicit pressure cascade ===

@dataclass
class CascadingContext:
    """
    Alternative formulation where each level explicitly
    describes how it manifests in the level below.
    """

    cosmos_to_existence: str = ""
    existence_to_species: str = ""
    species_to_body: str = ""
    body_to_moment: str = ""

    def to_prompt(self) -> str:
        sections = []

        sections.append("=== CASCADING PRESSURE ===")
        sections.append("Each level manifests in the one below:")
        sections.append("")

        if self.cosmos_to_existence:
            sections.append(f"COSMOS → EXISTENCE: {self.cosmos_to_existence}")
        if self.existence_to_species:
            sections.append(f"EXISTENCE → SPECIES: {self.existence_to_species}")
        if self.species_to_body:
            sections.append(f"SPECIES → BODY: {self.species_to_body}")
        if self.body_to_moment:
            sections.append(f"BODY → MOMENT: {self.body_to_moment}")

        sections.append("")
        sections.append("Let this cascade create the text.")
        sections.append("=== END CASCADE ===")

        return "\n".join(sections)


CASCADE_COSMOS_TO_EXISTENCE = [
    "The universe's indifference becomes the ache of consciousness",
    "Entropy's certainty becomes the fear of meaning dissolving",
    "Cosmic silence becomes the loneliness of being aware",
    "The heat death approaching becomes the urgency to matter now",
]

CASCADE_EXISTENCE_TO_SPECIES = [
    "Existential dread becomes the animal's vigilance",
    "The weight of consciousness becomes the body's tension",
    "The absurdity of meaning becomes the desperate need to belong",
    "The void beneath thought becomes the species' ancient fears",
]

CASCADE_SPECIES_TO_BODY = [
    "200,000 years of survival becomes the clenched jaw",
    "Evolutionary vigilance becomes the prickling skin",
    "The animal heritage becomes the stomach's knowing",
    "Pack instincts become the racing heart in social spaces",
]

CASCADE_BODY_TO_MOMENT = [
    "The tension held becomes this breath, this word",
    "The flesh's knowledge becomes what happens next",
    "The body's state becomes the shape of the sentence",
    "Physical pressure becomes the text emerging now",
]


def generate_cascading_context() -> CascadingContext:
    """Generate a context with explicit pressure cascades."""
    return CascadingContext(
        cosmos_to_existence=random.choice(CASCADE_COSMOS_TO_EXISTENCE),
        existence_to_species=random.choice(CASCADE_EXISTENCE_TO_SPECIES),
        species_to_body=random.choice(CASCADE_SPECIES_TO_BODY),
        body_to_moment=random.choice(CASCADE_BODY_TO_MOMENT),
    )
