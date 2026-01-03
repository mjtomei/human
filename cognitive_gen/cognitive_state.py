"""
Cognitive state architectures for hypothesis representation.

V1: Original 20-dimension architecture (primal + intellectual + moral)
V2: Expanded 30-dimension architecture (adds identity, power, existential, defensive)
"""

from dataclasses import dataclass, fields, field
from typing import Optional
import json


@dataclass
class CognitiveStateV1:
    """
    Original cognitive state architecture.
    20 dimensions covering embodied, intellectual, moral, relational,
    temporal, aesthetic, and hidden aspects.
    """
    # === EMBODIED ===
    body_state: str = ""
    preverbal_feeling: str = ""

    # === INTELLECTUAL ===
    core_belief: str = ""
    intellectual_stance: str = ""
    what_they_notice: str = ""

    # === MORAL/ETHICAL ===
    moral_framework: str = ""
    what_outrages_them: str = ""
    what_they_protect: str = ""

    # === RELATIONAL ===
    stance_toward_reader: str = ""
    who_they_write_for: str = ""
    what_they_want_reader_to_feel: str = ""

    # === TEMPORAL ===
    relationship_to_past: str = ""
    relationship_to_future: str = ""
    sense_of_urgency: str = ""

    # === AESTHETIC ===
    what_they_find_beautiful: str = ""
    what_they_find_ugly: str = ""
    relationship_to_language: str = ""

    # === HIDDEN ===
    what_they_cant_say_directly: str = ""
    the_wound: str = ""
    the_compensation: str = ""

    def to_prompt(self) -> str:
        sections = ["=== COGNITIVE STATE ===", "Write as if experiencing:", ""]
        for f in fields(self):
            value = getattr(self, f.name)
            if value:
                label = f.name.replace("_", " ").title()
                sections.append(f"{label}: {value}")
        sections.extend(["", "Let these shape the voice without naming them.", "=== END ==="])
        return "\n".join(sections)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name)}

    @classmethod
    def from_dict(cls, d: dict) -> "CognitiveStateV1":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def get_dimensions(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def get_dimension_groups(cls) -> dict[str, list[str]]:
        """Return semantically linked dimension groups."""
        return {
            "embodied": ["body_state", "preverbal_feeling"],
            "intellectual": ["core_belief", "intellectual_stance", "what_they_notice"],
            "moral": ["moral_framework", "what_outrages_them", "what_they_protect"],
            "relational": ["stance_toward_reader", "who_they_write_for", "what_they_want_reader_to_feel"],
            "temporal": ["relationship_to_past", "relationship_to_future", "sense_of_urgency"],
            "aesthetic": ["what_they_find_beautiful", "what_they_find_ugly", "relationship_to_language"],
            "shadow": ["what_they_cant_say_directly", "the_wound", "the_compensation"],
        }


@dataclass
class CognitiveStateV1_1:
    """
    V1.1: Original 20 dimensions plus 8 shadow/dark dimensions (28 total).
    Adds dimensions for malice, cruelty, resentment, and hidden darkness.
    """
    # === EMBODIED ===
    body_state: str = ""
    preverbal_feeling: str = ""

    # === INTELLECTUAL ===
    core_belief: str = ""
    intellectual_stance: str = ""
    what_they_notice: str = ""

    # === MORAL/ETHICAL ===
    moral_framework: str = ""
    what_outrages_them: str = ""
    what_they_protect: str = ""

    # === RELATIONAL ===
    stance_toward_reader: str = ""
    who_they_write_for: str = ""
    what_they_want_reader_to_feel: str = ""

    # === TEMPORAL ===
    relationship_to_past: str = ""
    relationship_to_future: str = ""
    sense_of_urgency: str = ""

    # === AESTHETIC ===
    what_they_find_beautiful: str = ""
    what_they_find_ugly: str = ""
    relationship_to_language: str = ""

    # === HIDDEN (from V1) ===
    what_they_cant_say_directly: str = ""
    the_wound: str = ""
    the_compensation: str = ""

    # === DARK / SHADOW (new in V1.1) ===
    the_cruelty_they_enjoy: str = ""
    who_they_resent: str = ""
    the_lie_they_tell_themselves: str = ""
    their_capacity_for_violence: str = ""
    what_they_refuse_to_forgive: str = ""
    their_contempt: str = ""
    the_power_they_crave: str = ""
    what_they_would_do_if_no_one_knew: str = ""

    def to_prompt(self) -> str:
        sections = ["=== COGNITIVE STATE ===", "Write as if experiencing:", ""]
        for f in fields(self):
            value = getattr(self, f.name)
            if value:
                label = f.name.replace("_", " ").title()
                sections.append(f"{label}: {value}")
        sections.extend(["", "Let these shape the voice without naming them.", "=== END ==="])
        return "\n".join(sections)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name)}

    @classmethod
    def from_dict(cls, d: dict) -> "CognitiveStateV1_1":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def get_dimensions(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def get_dimension_groups(cls) -> dict[str, list[str]]:
        """Return semantically linked dimension groups."""
        return {
            "embodied": ["body_state", "preverbal_feeling"],
            "intellectual": ["core_belief", "intellectual_stance", "what_they_notice"],
            "moral": ["moral_framework", "what_outrages_them", "what_they_protect"],
            "relational": ["stance_toward_reader", "who_they_write_for", "what_they_want_reader_to_feel"],
            "temporal": ["relationship_to_past", "relationship_to_future", "sense_of_urgency"],
            "aesthetic": ["what_they_find_beautiful", "what_they_find_ugly", "relationship_to_language"],
            "shadow": ["what_they_cant_say_directly", "the_wound", "the_compensation"],
            "dark": [
                "the_cruelty_they_enjoy", "who_they_resent", "the_lie_they_tell_themselves",
                "their_capacity_for_violence", "what_they_refuse_to_forgive", "their_contempt",
                "the_power_they_crave", "what_they_would_do_if_no_one_knew",
            ],
        }


@dataclass
class CognitiveStateV2:
    """
    Expanded cognitive state architecture.
    ~30 dimensions adding identity, power, existential, and defensive layers.
    """
    # === EMBODIED ===
    body_state: str = ""
    preverbal_feeling: str = ""
    where_tension_lives: str = ""
    somatic_memory: str = ""

    # === IDENTITY ===
    self_image: str = ""
    identity_they_reject: str = ""

    # === INTELLECTUAL ===
    core_belief: str = ""
    intellectual_stance: str = ""
    what_they_notice: str = ""
    how_they_know_things: str = ""
    relationship_to_uncertainty: str = ""

    # === MORAL ===
    moral_framework: str = ""
    what_outrages_them: str = ""
    what_they_protect: str = ""

    # === RELATIONAL ===
    stance_toward_reader: str = ""
    who_they_write_for: str = ""
    what_they_want_reader_to_feel: str = ""
    how_they_see_others: str = ""
    who_they_carry_with_them: str = ""

    # === POWER ===
    stance_toward_authority: str = ""
    where_they_feel_powerless: str = ""

    # === TEMPORAL ===
    relationship_to_past: str = ""
    relationship_to_future: str = ""
    sense_of_urgency: str = ""

    # === AESTHETIC ===
    what_they_find_beautiful: str = ""
    what_they_find_ugly: str = ""
    relationship_to_language: str = ""

    # === EXISTENTIAL ===
    relationship_to_meaning: str = ""
    hope_structure: str = ""

    # === DEFENSIVE ===
    primary_defense: str = ""
    what_threatens_them: str = ""
    how_they_hide: str = ""

    # === SHADOW ===
    what_they_cant_say_directly: str = ""
    the_wound: str = ""
    the_compensation: str = ""
    source_of_shame: str = ""
    central_tension: str = ""

    def to_prompt(self) -> str:
        sections = ["=== COGNITIVE STATE ===", "Write as if experiencing:", ""]
        for f in fields(self):
            value = getattr(self, f.name)
            if value:
                label = f.name.replace("_", " ").title()
                sections.append(f"{label}: {value}")
        sections.extend(["", "Let these shape the voice without naming them.", "=== END ==="])
        return "\n".join(sections)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name)}

    @classmethod
    def from_dict(cls, d: dict) -> "CognitiveStateV2":
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def get_dimensions(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def get_dimension_groups(cls) -> dict[str, list[str]]:
        """Return semantically linked dimension groups."""
        return {
            "embodied": ["body_state", "preverbal_feeling", "where_tension_lives", "somatic_memory"],
            "identity": ["self_image", "identity_they_reject"],
            "intellectual": ["core_belief", "intellectual_stance", "what_they_notice",
                           "how_they_know_things", "relationship_to_uncertainty"],
            "moral": ["moral_framework", "what_outrages_them", "what_they_protect"],
            "relational": ["stance_toward_reader", "who_they_write_for", "what_they_want_reader_to_feel",
                          "how_they_see_others", "who_they_carry_with_them"],
            "power": ["stance_toward_authority", "where_they_feel_powerless"],
            "temporal": ["relationship_to_past", "relationship_to_future", "sense_of_urgency"],
            "aesthetic": ["what_they_find_beautiful", "what_they_find_ugly", "relationship_to_language"],
            "existential": ["relationship_to_meaning", "hope_structure"],
            "defensive": ["primary_defense", "what_threatens_them", "how_they_hide"],
            "shadow": ["what_they_cant_say_directly", "the_wound", "the_compensation",
                      "source_of_shame", "central_tension"],
        }


# Type alias for any version
CognitiveState = CognitiveStateV1 | CognitiveStateV1_1 | CognitiveStateV2


def get_state_class(version: str = "v1"):
    """Get the appropriate state class by version."""
    if version.lower() == "v1":
        return CognitiveStateV1
    elif version.lower() == "v1.1" or version.lower() == "v1_1":
        return CognitiveStateV1_1
    elif version.lower() == "v2":
        return CognitiveStateV2
    else:
        raise ValueError(f"Unknown version: {version}. Use 'v1', 'v1.1', or 'v2'.")
