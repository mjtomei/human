"""
Large Dimension Pool spanning all levels of abstraction.

This pool contains ~50 candidate dimensions. The meta-search
will discover which subset is most predictive.
"""

# All candidate dimensions organized by category (for reference only - search ignores categories)
DIMENSION_POOL = {
    # ==========================================================================
    # EXISTENTIAL / COSMIC (most abstract, stable across lifetime)
    # ==========================================================================
    "relationship_to_existence": "How they relate to the fact of being",
    "cosmic_stance": "Their position toward the universe/reality",
    "mortality_awareness": "How death shapes their consciousness",
    "meaning_framework": "Where meaning comes from (or doesn't)",
    "relationship_to_time": "How they experience temporality",
    "ontological_position": "What they believe is fundamentally real",
    "relationship_to_infinity": "How they relate to the boundless/eternal",

    # ==========================================================================
    # PSYCHOLOGICAL / IDENTITY (stable across years)
    # ==========================================================================
    "core_belief": "Foundational belief about reality/life",
    "self_image": "How they see themselves",
    "shadow_self": "What they reject/deny in themselves",
    "identity_they_reject": "Who they refuse to be",
    "relationship_to_uncertainty": "How they handle not-knowing",
    "intellectual_stance": "How they approach knowing/truth",
    "moral_framework": "Their ethical orientation",
    "what_outrages_them": "What triggers moral anger",
    "deepest_fear": "What they most dread",
    "deepest_desire": "What they most want",

    # ==========================================================================
    # EMBODIED / SOMATIC (felt sense, body-level)
    # ==========================================================================
    "body_state": "Current embodied experience",
    "preverbal_feeling": "Feeling before words",
    "where_tension_lives": "Location of held tension",
    "somatic_memory": "Body's stored experiences",
    "felt_sense_of_world": "How the world feels (not thinks)",
    "breathing_pattern": "Quality of breath/life force",
    "posture_toward_existence": "Bodily stance toward being",

    # ==========================================================================
    # RELATIONAL / SOCIAL (how they relate to others)
    # ==========================================================================
    "stance_toward_reader": "Relationship to who reads them",
    "who_they_write_for": "Intended/imagined audience",
    "what_they_want_reader_to_feel": "Intended emotional effect",
    "audience_relationship": "Dynamic with their readers",
    "relationship_to_authority": "How they relate to power",
    "relationship_to_collective": "How they relate to groups/society",
    "what_they_offer": "What they give to others",

    # ==========================================================================
    # SHADOW / HIDDEN (what's unsaid, repressed, protected)
    # ==========================================================================
    "what_they_hide": "What they conceal from others",
    "what_they_cant_say_directly": "What must be said sideways",
    "what_they_protect": "What they guard/defend",
    "forbidden_knowledge": "What they know but shouldn't",
    "secret_shame": "Hidden source of shame",
    "unacknowledged_desire": "Want they won't admit",

    # ==========================================================================
    # TEMPORAL (relationship to past/present/future)
    # ==========================================================================
    "relationship_to_past": "How they carry history",
    "relationship_to_future": "How they anticipate what's coming",
    "relationship_to_present": "How they inhabit now",
    "what_haunts_them": "What returns unbidden",
    "what_they_anticipate": "What they expect/dread",

    # ==========================================================================
    # AESTHETIC / PERCEPTUAL (what they notice, find beautiful)
    # ==========================================================================
    "what_they_notice": "What draws their attention",
    "what_they_find_beautiful": "Their sense of beauty",
    "what_they_find_ugly": "What repels or disgusts them",
    "how_they_see_details": "Quality of their attention",
    "sensory_orientation": "Which senses dominate",

    # ==========================================================================
    # CREATIVE / THEMATIC (specific to their work)
    # ==========================================================================
    "active_preoccupation": "Current obsession/focus",
    "creative_mission": "What they're trying to accomplish",
    "life_circumstance": "Current situation shaping writing",
    "emotional_weather": "Prevailing emotional climate",
    "central_tension": "Core conflict they're working with",
    "sense_of_urgency": "Temporal pressure or lack thereof",

    # ==========================================================================
    # PSYCHODYNAMIC (core wounds and defenses)
    # ==========================================================================
    "the_wound": "The original injury that shaped them",
    "the_compensation": "How they defend against or make up for the wound",

    # ==========================================================================
    # SHADOW / DARK (malice, cruelty, what's denied or repressed)
    # ==========================================================================
    "the_cruelty_they_enjoy": "What forms of harm or dominance they secretly enjoy",
    "who_they_resent": "People or types they harbor resentment toward",
    "the_lie_they_tell_themselves": "The core self-deception that protects their self-image",
    "their_capacity_for_violence": "What violence (physical, emotional, social) they're capable of",
    "what_they_refuse_to_forgive": "Grudges, unforgiven wounds they carry",
    "what_they_want_to_destroy": "What they want to tear down, end, or eliminate",
    "the_envy_they_wont_admit": "Hidden jealousy of others' success, traits, or circumstances",
    "their_contempt": "Who or what they look down on, consider beneath them",
    "the_power_they_crave": "Hidden desires for control, dominance, or influence",
    "their_petty_satisfactions": "Small cruelties or mean pleasures that satisfy them",
    "the_mask_they_wear": "The false self they present to the world",
    "what_they_would_do_if_no_one_knew": "Actions they'd take without consequence or witness",

    # ==========================================================================
    # EPISTEMOLOGICAL (how they know)
    # ==========================================================================
    "how_they_know_things": "Their way of knowing",
    "relationship_to_truth": "How they relate to truth",
    "what_counts_as_evidence": "What convinces them",
    "relationship_to_language": "How they relate to words",

    # ==========================================================================
    # CLINICAL / PATHOLOGICAL (mental health dimensions, disorder-adjacent traits)
    # ==========================================================================
    # Dark Triad
    "narcissistic_supply": "What feeds their grandiosity and need for admiration",
    "machiavellian_calculation": "How they manipulate and strategize for personal gain",
    "psychopathic_detachment": "Capacity for callous disregard of others' wellbeing",

    # Schizotypal / Perception
    "magical_thinking": "Belief in supernatural connections, omens, or personal powers",
    "paranoid_ideation": "Suspicion that others have malicious intent toward them",
    "ideas_of_reference": "Belief that random events or messages are personally directed at them",
    "apophenia": "Tendency to perceive meaningful patterns in random or unrelated things",
    "perceptual_disturbance": "Unusual sensory experiences or mild hallucinations",

    # Dissociative / Identity
    "dissociative_tendency": "Capacity to detach from self, body, or reality",
    "identity_fragmentation": "Sense of multiple selves or unstable identity",
    "derealization": "Feeling that the world is unreal, dreamlike, or artificial",

    # Grandiosity / Delusion-adjacent
    "grandiose_fantasy": "Beliefs about special destiny, unique importance, or hidden powers",
    "persecution_complex": "Belief that forces are aligned against them specifically",
    "messianic_tendency": "Sense of being chosen or having a world-changing mission",

    # Obsessive / Compulsive
    "obsessive_fixation": "Thoughts they cannot escape or stop returning to",
    "compulsive_ritual": "Behaviors or patterns they must repeat",
    "intrusive_thoughts": "Unwanted thoughts that disturb or horrify them",

    # Attachment / Relational pathology
    "abandonment_terror": "Overwhelming fear of being left or forgotten",
    "engulfment_fear": "Fear of losing self in relationships or being consumed by others",
    "object_inconstancy": "Difficulty holding stable images of others when apart",

    # ==========================================================================
    # SEVEN DEADLY SINS (Christian vices)
    # ==========================================================================
    "pride": "Excessive belief in one's own abilities; the sin from which all others arise",
    "greed": "Desire for material wealth or gain beyond what one needs",
    "lust": "Intense or unbridled desire, especially of a sexual nature",
    "envy": "Desire for others' traits, status, abilities, or possessions",
    "gluttony": "Overindulgence and overconsumption to the point of waste",
    "wrath": "Uncontrolled feelings of anger, rage, and hatred",
    "sloth": "Avoidance of physical or spiritual work; apathy and inactivity",

    # ==========================================================================
    # FRUITS OF THE SPIRIT (Christian virtues - Galatians 5:22-23)
    # ==========================================================================
    "love_agape": "Selfless, unconditional love and care for others",
    "joy": "Deep-seated gladness not dependent on circumstances",
    "peace": "Tranquility and harmony; absence of inner conflict",
    "patience": "Ability to endure waiting, delay, or provocation without anger",
    "kindness": "Quality of being friendly, generous, and considerate",
    "goodness": "Moral excellence; virtue and righteousness in action",
    "faithfulness": "Steadfast loyalty and reliability; keeping commitments",
    "gentleness": "Mildness of manner; humility and considerateness",
    "self_control": "Ability to regulate one's emotions, desires, and behaviors",
}

# Flat list for easy access
ALL_DIMENSIONS = list(DIMENSION_POOL.keys())

# Number of dimensions
N_DIMENSIONS = len(ALL_DIMENSIONS)

# Dimension groups for linkage-aware crossover
# Dimensions in same group tend to be co-adapted and should be inherited together
DIMENSION_GROUPS = {
    "existential": [
        "relationship_to_existence", "cosmic_stance", "mortality_awareness",
        "meaning_framework", "relationship_to_time", "ontological_position",
        "relationship_to_infinity",
    ],
    "identity": [
        "core_belief", "self_image", "shadow_self", "identity_they_reject",
    ],
    "intellectual": [
        "intellectual_stance", "relationship_to_uncertainty",
        "how_they_know_things", "relationship_to_truth", "what_counts_as_evidence",
    ],
    "moral": [
        "moral_framework", "what_outrages_them", "what_they_protect",
    ],
    "embodied": [
        "body_state", "preverbal_feeling", "where_tension_lives",
        "somatic_memory", "felt_sense_of_world", "breathing_pattern",
        "posture_toward_existence",
    ],
    "relational": [
        "stance_toward_reader", "who_they_write_for", "what_they_want_reader_to_feel",
        "audience_relationship", "relationship_to_authority", "relationship_to_collective",
        "what_they_offer",
    ],
    "shadow": [
        "what_they_hide", "what_they_cant_say_directly", "forbidden_knowledge",
        "secret_shame", "unacknowledged_desire", "deepest_fear", "deepest_desire",
    ],
    "dark": [
        "the_cruelty_they_enjoy", "who_they_resent", "the_lie_they_tell_themselves",
        "their_capacity_for_violence", "what_they_refuse_to_forgive", "what_they_want_to_destroy",
        "the_envy_they_wont_admit", "their_contempt", "the_power_they_crave",
        "their_petty_satisfactions", "the_mask_they_wear", "what_they_would_do_if_no_one_knew",
    ],
    "clinical": [
        # Dark triad
        "narcissistic_supply", "machiavellian_calculation", "psychopathic_detachment",
        # Schizotypal
        "magical_thinking", "paranoid_ideation", "ideas_of_reference", "apophenia",
        "perceptual_disturbance",
        # Dissociative
        "dissociative_tendency", "identity_fragmentation", "derealization",
        # Grandiose
        "grandiose_fantasy", "persecution_complex", "messianic_tendency",
        # Obsessive
        "obsessive_fixation", "compulsive_ritual", "intrusive_thoughts",
        # Attachment
        "abandonment_terror", "engulfment_fear", "object_inconstancy",
    ],
    "temporal": [
        "relationship_to_past", "relationship_to_future", "relationship_to_present",
        "what_haunts_them", "what_they_anticipate", "sense_of_urgency",
    ],
    "aesthetic": [
        "what_they_notice", "what_they_find_beautiful", "what_they_find_ugly",
        "how_they_see_details", "sensory_orientation", "relationship_to_language",
    ],
    "creative": [
        "active_preoccupation", "creative_mission", "life_circumstance",
        "emotional_weather", "central_tension",
    ],
    "psychodynamic": [
        "the_wound", "the_compensation",
    ],
    "deadly_sins": [
        "pride", "greed", "lust", "envy", "gluttony", "wrath", "sloth",
    ],
    "fruits_of_spirit": [
        "love_agape", "joy", "peace", "patience", "kindness",
        "goodness", "faithfulness", "gentleness", "self_control",
    ],
}

# Map each dimension to its group
DIMENSION_TO_GROUP = {}
for group_name, dims in DIMENSION_GROUPS.items():
    for dim in dims:
        DIMENSION_TO_GROUP[dim] = group_name

# V1's exact dimensions and groups (proven to work well)
V1_DIMENSIONS = [
    "body_state", "preverbal_feeling",
    "core_belief", "intellectual_stance", "what_they_notice",
    "moral_framework", "what_outrages_them", "what_they_protect",
    "stance_toward_reader", "who_they_write_for", "what_they_want_reader_to_feel",
    "relationship_to_past", "relationship_to_future", "sense_of_urgency",
    "what_they_find_beautiful", "what_they_find_ugly", "relationship_to_language",
    "what_they_cant_say_directly", "the_wound", "the_compensation",
]

V1_LINKAGE_GROUPS = {
    "embodied": ["body_state", "preverbal_feeling"],
    "intellectual": ["core_belief", "intellectual_stance", "what_they_notice"],
    "moral": ["moral_framework", "what_outrages_them", "what_they_protect"],
    "relational": ["stance_toward_reader", "who_they_write_for", "what_they_want_reader_to_feel"],
    "temporal": ["relationship_to_past", "relationship_to_future", "sense_of_urgency"],
    "aesthetic": ["what_they_find_beautiful", "what_they_find_ugly", "relationship_to_language"],
    "shadow": ["what_they_cant_say_directly", "the_wound", "the_compensation"],
}

V1_DIMENSION_TO_GROUP = {}
for group_name, dims in V1_LINKAGE_GROUPS.items():
    for dim in dims:
        V1_DIMENSION_TO_GROUP[dim] = group_name


# V1.1: V1 dimensions plus shadow/dark dimensions (28 total)
V1_1_DIMENSIONS = V1_DIMENSIONS + [
    # Dark/shadow additions
    "the_cruelty_they_enjoy",
    "who_they_resent",
    "the_lie_they_tell_themselves",
    "their_capacity_for_violence",
    "what_they_refuse_to_forgive",
    "their_contempt",
    "the_power_they_crave",
    "what_they_would_do_if_no_one_knew",
]

V1_1_LINKAGE_GROUPS = {
    **V1_LINKAGE_GROUPS,
    "dark": [
        "the_cruelty_they_enjoy", "who_they_resent", "the_lie_they_tell_themselves",
        "their_capacity_for_violence", "what_they_refuse_to_forgive", "their_contempt",
        "the_power_they_crave", "what_they_would_do_if_no_one_knew",
    ],
}

V1_1_DIMENSION_TO_GROUP = {}
for group_name, dims in V1_1_LINKAGE_GROUPS.items():
    for dim in dims:
        V1_1_DIMENSION_TO_GROUP[dim] = group_name


def get_dimension_description(dim: str) -> str:
    """Get description for a dimension."""
    return DIMENSION_POOL.get(dim, dim)


def get_dimension_group(dim: str) -> str:
    """Get the group a dimension belongs to."""
    return DIMENSION_TO_GROUP.get(dim, "other")


def get_random_dimensions(n: int) -> list[str]:
    """Get n random dimensions from the pool."""
    import random
    return random.sample(ALL_DIMENSIONS, min(n, len(ALL_DIMENSIONS)))


def sample_dimensions_by_group(
    n_groups: int = 5,
    dims_per_group: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Sample dimensions while keeping group structure.

    Returns (dimensions, group_names) where dimensions from the same group
    are kept together to preserve potential co-adaptation.

    Args:
        n_groups: Number of dimension groups to sample
        dims_per_group: Max dimensions to sample from each group

    Returns:
        Tuple of (list of dimension names, list of group names used)
    """
    import random

    all_groups = list(DIMENSION_GROUPS.keys())

    # Purely random group selection - no bias toward any group
    selected_groups = random.sample(all_groups, min(n_groups, len(all_groups)))

    # Sample dimensions from each selected group
    dimensions = []
    for group in selected_groups:
        group_dims = DIMENSION_GROUPS[group]
        n_sample = min(dims_per_group, len(group_dims))
        dimensions.extend(random.sample(group_dims, n_sample))

    return dimensions, selected_groups


# Architecture templates representing different psychological theories
ARCHITECTURE_TEMPLATES = {
    "psychodynamic": ["embodied", "shadow", "psychodynamic"],      # Freudian - wounds, compensation, body
    "existential": ["existential", "temporal", "aesthetic"],       # Phenomenological - being, time, perception
    "pathological": ["dark", "clinical", "identity"],              # Clinical - disorders, dark traits
    "social": ["relational", "moral", "intellectual"],             # Social psych - relationships, ethics, cognition
    "spiritual": ["deadly_sins", "fruits_of_spirit", "existential"],  # Religious - vices, virtues, meaning
    "creative": ["creative", "aesthetic", "temporal"],             # Artistic - process, beauty, urgency
    "somatic": ["embodied", "temporal", "psychodynamic"],          # Body-centered - felt sense, wounds
    "shadow_work": ["shadow", "dark", "psychodynamic"],            # Depth psych - hidden, repressed, wounds
}


def sample_from_template(
    template_name: str,
    dims_per_group: int = 3,
) -> tuple[list[str], list[str]]:
    """
    Sample dimensions from a predefined architecture template.

    Args:
        template_name: Name of template from ARCHITECTURE_TEMPLATES
        dims_per_group: Max dimensions to sample from each group

    Returns:
        Tuple of (list of dimension names, list of group names used)
    """
    import random

    if template_name not in ARCHITECTURE_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")

    selected_groups = ARCHITECTURE_TEMPLATES[template_name]

    # Sample dimensions from each group in the template
    dimensions = []
    for group in selected_groups:
        group_dims = DIMENSION_GROUPS[group]
        n_sample = min(dims_per_group, len(group_dims))
        dimensions.extend(random.sample(group_dims, n_sample))

    return dimensions, selected_groups


def get_random_template_name() -> str:
    """Get a random template name."""
    import random
    return random.choice(list(ARCHITECTURE_TEMPLATES.keys()))


def get_dimension_prompt_section(dimensions: list[str]) -> str:
    """
    Generate a prompt section listing dimensions with descriptions.

    Args:
        dimensions: List of dimension names to include

    Returns:
        Formatted string for use in prompts
    """
    lines = []
    for dim in dimensions:
        desc = DIMENSION_POOL.get(dim, dim.replace("_", " "))
        lines.append(f"- {dim}: {desc}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Dimension Pool: {N_DIMENSIONS} dimensions")
    print("\nCategories:")

    categories = {}
    current_cat = None
    for line in open(__file__).readlines():
        if "# ====" in line and "===" in line:
            continue
        if line.strip().startswith("# ") and "(" in line:
            current_cat = line.strip("# \n").split("(")[0].strip()
            categories[current_cat] = 0
        elif '": "' in line and current_cat:
            categories[current_cat] += 1

    for cat, count in categories.items():
        print(f"  {cat}: {count}")
