"""
Reverse inference: Text → Cognitive State

Given observed text, infer the cognitive context that would most likely produce it.

Method:
1. Take a corpus of someone's writing
2. Split into "context" samples and "prediction" targets
3. For each target, propose many candidate cognitive states
4. Generate predictions from each cognitive state
5. Score how well each prediction matches the actual target
6. Build a distribution over cognitive states that best explain the writing

This is essentially Bayesian inference over the latent cognitive space.
"""

import json
import random
from dataclasses import dataclass, field
from typing import Optional
import anthropic

from cognitive_gen.hierarchical_context import (
    HierarchicalContext,
    COSMOS_POOL,
    EXISTENCE_POOL,
    SPECIES_POOL,
    TRIBE_POOL,
    SELF_POOL,
    BODY_POOL,
    MOMENT_POOL,
)
from cognitive_gen.refined_context import (
    AnimalContext,
    BODY_STATES,
    BODY_MEMORIES,
    BODY_WANTS,
    PREDATOR_ASPECTS,
    PREY_ASPECTS,
    WATCHER_SEES,
    PREVERBAL_SENSATIONS,
)


@dataclass
class CognitiveHypothesis:
    """A hypothesis about someone's cognitive state."""

    # Hierarchical layers
    cosmos: str = ""
    existence: str = ""
    species: str = ""
    body: str = ""
    moment: str = ""

    # Animal/embodied layers
    body_state: str = ""
    body_want: str = ""
    predator_aspect: str = ""
    prey_aspect: str = ""
    watcher_sees: str = ""
    preverbal: str = ""

    # Inferred psychological state
    primary_drive: str = ""
    hidden_fear: str = ""
    unspoken_desire: str = ""
    self_deception: str = ""

    def to_prompt(self) -> str:
        """Convert to conditioning prompt."""
        sections = []
        sections.append("=== INFERRED COGNITIVE STATE ===")
        sections.append("Write as if experiencing:")
        sections.append("")

        if self.cosmos:
            sections.append(f"COSMIC AWARENESS: {self.cosmos}")
        if self.existence:
            sections.append(f"EXISTENTIAL STANCE: {self.existence}")
        if self.body_state:
            sections.append(f"BODY STATE: {self.body_state}")
        if self.body_want:
            sections.append(f"BODY WANTS: {self.body_want}")
        if self.predator_aspect:
            sections.append(f"PREDATOR IN YOU: {self.predator_aspect}")
        if self.prey_aspect:
            sections.append(f"PREY IN YOU: {self.prey_aspect}")
        if self.watcher_sees:
            sections.append(f"THE WATCHER SEES: {self.watcher_sees}")
        if self.preverbal:
            sections.append(f"PREVERBAL SENSATION: {self.preverbal}")
        if self.primary_drive:
            sections.append(f"PRIMARY DRIVE: {self.primary_drive}")
        if self.hidden_fear:
            sections.append(f"HIDDEN FEAR: {self.hidden_fear}")
        if self.unspoken_desire:
            sections.append(f"UNSPOKEN DESIRE: {self.unspoken_desire}")
        if self.self_deception:
            sections.append(f"SELF-DECEPTION: {self.self_deception}")

        sections.append("")
        sections.append("Let these shape the voice without naming them.")
        sections.append("=== END STATE ===")

        return "\n".join(sections)


# Extended pools for psychological inference
PRIMARY_DRIVE_POOL = [
    "To be seen as intelligent, to never appear foolish",
    "To control, to never be at another's mercy",
    "To be loved, to matter to someone",
    "To be right, to have the correct understanding",
    "To create something that outlasts the body",
    "To understand, to not be confused by existence",
    "To belong, to be inside rather than outside",
    "To be free, to escape all constraint",
    "To be safe, to eliminate threat",
    "To be special, to not be ordinary",
]

HIDDEN_FEAR_POOL = [
    "That I am not as smart as I pretend to be",
    "That I am fundamentally unlovable",
    "That my life has been wasted on the wrong things",
    "That I will be forgotten completely after death",
    "That others can see through my performance",
    "That I am the same as everyone else, nothing special",
    "That my beliefs are wrong and I've built my life on error",
    "That I am weak and others will exploit this",
    "That I don't actually know what I'm doing",
    "That the emptiness inside cannot be filled",
]

UNSPOKEN_DESIRE_POOL = [
    "To stop performing, to be seen without the mask",
    "To be held without having to ask",
    "To destroy something, to let the rage out",
    "To disappear, to be free of the self",
    "To be told what to do, to surrender responsibility",
    "To hurt someone who hurt me",
    "To be young again, to have the choices back",
    "To be admired without having to earn it",
    "To rest, to stop striving",
    "To know what others really think of me",
]

SELF_DECEPTION_POOL = [
    "I am objective when I am actually defensive",
    "I am helping when I am actually controlling",
    "I am honest when I am actually curating",
    "I don't care what others think (I care desperately)",
    "I am over it (I am not over it)",
    "I am being rational when I am rationalizing",
    "I am generous when I am buying loyalty",
    "I am principled when I am rigid",
    "I am independent when I am avoidant",
    "I am curious when I am anxious",
]


def generate_random_hypothesis() -> CognitiveHypothesis:
    """Generate a random cognitive hypothesis."""
    return CognitiveHypothesis(
        cosmos=random.choice(COSMOS_POOL) if random.random() > 0.5 else "",
        existence=random.choice(EXISTENCE_POOL) if random.random() > 0.3 else "",
        body_state=random.choice(BODY_STATES),
        body_want=random.choice(BODY_WANTS) if random.random() > 0.5 else "",
        predator_aspect=random.choice(PREDATOR_ASPECTS) if random.random() > 0.5 else "",
        prey_aspect=random.choice(PREY_ASPECTS) if random.random() > 0.5 else "",
        watcher_sees=random.choice(WATCHER_SEES) if random.random() > 0.3 else "",
        preverbal=random.choice(PREVERBAL_SENSATIONS) if random.random() > 0.5 else "",
        primary_drive=random.choice(PRIMARY_DRIVE_POOL),
        hidden_fear=random.choice(HIDDEN_FEAR_POOL),
        unspoken_desire=random.choice(UNSPOKEN_DESIRE_POOL) if random.random() > 0.5 else "",
        self_deception=random.choice(SELF_DECEPTION_POOL) if random.random() > 0.5 else "",
    )


class ReverseInferenceEngine:
    """
    Infer cognitive states from observed text.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    def score_hypothesis(
        self,
        hypothesis: CognitiveHypothesis,
        context_text: str,
        target_text: str,
        n_samples: int = 3,
    ) -> float:
        """
        Score how well a hypothesis predicts target text given context.

        Returns average similarity between generated continuations and actual target.
        """
        context_prompt = hypothesis.to_prompt()

        prompt = f"""You are channeling a specific writer's voice. You have been given samples of their writing and must continue in their exact style.

{context_prompt}

Here is a sample of their writing for context:

<context>
{context_text}
</context>

Now write a short continuation (100-150 words) in this exact voice and style.
Do not explain or comment. Just write as this person would write.
"""

        generations = []
        for _ in range(n_samples):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                temperature=0.9,
                messages=[{"role": "user", "content": prompt}]
            )
            generations.append(response.content[0].text)

        # Score similarity between generations and target
        scores = []
        for gen in generations:
            score = self._compute_similarity(gen, target_text)
            scores.append(score)

        return sum(scores) / len(scores)

    def _compute_similarity(self, generated: str, target: str) -> float:
        """
        Compute similarity between generated and target text.
        Uses Claude to judge stylistic/voice similarity.
        """
        prompt = f"""Rate how similar these two text samples are in VOICE and STYLE (not content).
Consider: sentence rhythm, word choice, emotional register, perspective, what's left unsaid.

Text A:
{generated[:500]}

Text B:
{target[:500]}

Rate similarity from 0.0 (completely different voices) to 1.0 (identical voice).
Respond with just a number."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            return float(response.content[0].text.strip())
        except:
            return 0.5  # Default if parsing fails

    def infer_cognitive_state(
        self,
        writing_samples: list[str],
        n_hypotheses: int = 20,
        n_iterations: int = 3,
    ) -> list[tuple[CognitiveHypothesis, float]]:
        """
        Infer cognitive state from writing samples.

        Returns ranked list of (hypothesis, score) tuples.
        """
        # Split samples into context and targets
        if len(writing_samples) < 2:
            raise ValueError("Need at least 2 writing samples")

        context_samples = writing_samples[:-1]
        target_sample = writing_samples[-1]

        context_text = "\n\n---\n\n".join(context_samples)

        results = []

        for i in range(n_hypotheses):
            hypothesis = generate_random_hypothesis()

            # Score this hypothesis
            score = self.score_hypothesis(
                hypothesis,
                context_text,
                target_sample,
                n_samples=n_iterations,
            )

            results.append((hypothesis, score))
            print(f"Hypothesis {i+1}/{n_hypotheses}: score = {score:.3f}")

        # Sort by score (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def generate_cognitive_profile(
        self,
        writing_samples: list[str],
        n_hypotheses: int = 50,
    ) -> dict:
        """
        Generate a full cognitive profile from writing samples.

        Returns a distribution over cognitive elements.
        """
        results = self.infer_cognitive_state(
            writing_samples,
            n_hypotheses=n_hypotheses
        )

        # Take top 20% of hypotheses
        top_k = max(1, n_hypotheses // 5)
        top_hypotheses = results[:top_k]

        # Aggregate cognitive elements
        profile = {
            "primary_drives": {},
            "hidden_fears": {},
            "body_states": {},
            "predator_aspects": {},
            "prey_aspects": {},
            "self_deceptions": {},
        }

        for hyp, score in top_hypotheses:
            if hyp.primary_drive:
                profile["primary_drives"][hyp.primary_drive] = \
                    profile["primary_drives"].get(hyp.primary_drive, 0) + score
            if hyp.hidden_fear:
                profile["hidden_fears"][hyp.hidden_fear] = \
                    profile["hidden_fears"].get(hyp.hidden_fear, 0) + score
            if hyp.body_state:
                profile["body_states"][hyp.body_state] = \
                    profile["body_states"].get(hyp.body_state, 0) + score
            if hyp.predator_aspect:
                profile["predator_aspects"][hyp.predator_aspect] = \
                    profile["predator_aspects"].get(hyp.predator_aspect, 0) + score
            if hyp.prey_aspect:
                profile["prey_aspects"][hyp.prey_aspect] = \
                    profile["prey_aspects"].get(hyp.prey_aspect, 0) + score
            if hyp.self_deception:
                profile["self_deceptions"][hyp.self_deception] = \
                    profile["self_deceptions"].get(hyp.self_deception, 0) + score

        # Normalize and sort
        for key in profile:
            if profile[key]:
                total = sum(profile[key].values())
                profile[key] = {
                    k: v/total for k, v in
                    sorted(profile[key].items(), key=lambda x: x[1], reverse=True)
                }

        return profile


# === NON-FAMOUS WRITERS ===
# Curated samples representing regular people's writing online
# These test whether the technique works on non-canonical writing

REGULAR_WRITERS = {
    "grief_post": {
        "name": "Anonymous (Grief forum)",
        "source": "Public forum post about loss",
        "samples": [
            """My mom died three weeks ago. I keep picking up my phone to call her. Every time something happens - good or bad - my first instinct is still to tell her. Yesterday I got a promotion at work and I was halfway through dialing before I remembered.""",

            """People keep telling me it gets easier. I don't want it to get easier. Easier feels like forgetting. Like she mattered less than she did. I want it to stay hard because that hardness is proof of how much she mattered.""",

            """I found her grocery list yesterday. Just a normal list - milk, bread, bananas, that yogurt she liked. Her handwriting. Such an ordinary thing to have outlasted her. I can't throw it away.""",

            """The worst part is how the world just keeps going. Traffic, weather, people complaining about their coffee. Don't they know? How can everything be so normal when nothing is normal anymore?""",
        ]
    },

    "career_post": {
        "name": "Anonymous (Career forum)",
        "source": "Public forum posts about work",
        "samples": [
            """I've been at my job for 7 years. I'm good at it. I hate it. Every Sunday night I feel this dread building in my chest. But I have a mortgage and two kids and I can't just quit to "find myself" like some 22-year-old.""",

            """My coworker got promoted over me again. Third time. I trained her. I literally trained her. And now she's my boss. HR says it's about "leadership potential." Whatever that means when you do the actual work.""",

            """Had my annual review today. "Exceeds expectations" across the board. Two percent raise. Inflation is running at four percent. My boss called it "a strong year" with a straight face.""",

            """I used to believe hard work got noticed. Now I know: hard work gets you more work. The people who get ahead are the ones who are good at looking like they're working while actually networking.""",
        ]
    },

    "travel_blog": {
        "name": "Anonymous (Travel blog)",
        "source": "Personal travel writing",
        "samples": [
            """The guidebook said to visit the famous cathedral. I walked past it to a bakery where an old woman was pulling bread from an oven that looked older than my country. She didn't speak English. I didn't speak Portuguese. We communicated in gestures and flour dust and the universal language of good bread.""",

            """Everyone photographs the Eiffel Tower. I photographed the couple arguing beneath it, the teenager reading alone on the grass, the vendor who'd seen a million tourists and looked right through them all. The tower is just the excuse. The people are the story.""",

            """I've been traveling for six months now. People back home think I'm living the dream. They don't see the loneliness of eating every meal alone, of having no one to say "did you see that?" to. Freedom and loneliness are the same thing with different lighting.""",

            """The best meal I had in Japan wasn't at any restaurant. It was rice balls from a 7-Eleven at 2am, sitting on a curb in Tokyo, jet-lagged and crying for no reason. Sometimes you have to be very far from home to find out who you are when no one's watching.""",
        ]
    },

    "advice_post": {
        "name": "Anonymous (Advice forum)",
        "source": "Public Q&A responses",
        "samples": [
            """I was you, ten years ago. Stayed in a relationship I knew was wrong because leaving felt harder than staying. Here's what I wish someone had told me: that dull ache you feel? That's you, dying slowly. Not dramatically. Just fading.""",

            """Everyone giving you logical advice is missing the point. You don't need someone to solve your problem. You already know the answer. You're hoping someone will give you permission to do what you're afraid to do.""",

            """The question isn't whether you should leave your job. The question is: what are you so afraid of that you're asking strangers on the internet instead of trusting yourself? That fear is the real problem.""",

            """I've given advice to thousands of people on here. Most don't take it. Not because it's bad advice, but because advice isn't what they want. They want witness. Someone to see them struggling. Consider yourself seen.""",
        ]
    },
}


# === PUBLIC FIGURE WRITING SAMPLES ===
# Using publicly available writing from well-known figures

PUBLIC_FIGURES = {
    "hemingway": {
        "name": "Ernest Hemingway",
        "source": "Letters and journalism",
        "samples": [
            # From his letters
            """The great thing is to last and get your work done and see and hear and learn and understand; and write when there is something that you know; and not before; and not too damned much after.""",

            """I know war as few other men now living know it, and nothing to me is more revolting. I have long advocated its complete abolition, as its very destructiveness on both combatants and enemies has rendered it useless as a means of settling international disputes.""",

            """All good books are alike in that they are truer than if they had really happened and after you are finished reading one you will feel that all that happened to you and afterwards it all belongs to you: the good and the bad, the ecstasy, the remorse and sorrow, the people and the places and how the weather was.""",

            """The world breaks everyone and afterward many are strong at the broken places. But those that will not break it kills. It kills the very good and the very gentle and the very brave impartially.""",
        ]
    },

    "orwell": {
        "name": "George Orwell",
        "source": "Essays and journalism",
        "samples": [
            """In our age there is no such thing as 'keeping out of politics.' All issues are political issues, and politics itself is a mass of lies, evasions, folly, hatred and schizophrenia.""",

            """The nationalist not only does not disapprove of atrocities committed by his own side, but he has a remarkable capacity for not even hearing about them.""",

            """Political language is designed to make lies sound truthful and murder respectable, and to give an appearance of solidity to pure wind.""",

            """Every generation imagines itself to be more intelligent than the one that went before it, and wiser than the one that comes after it.""",
        ]
    },

    "didion": {
        "name": "Joan Didion",
        "source": "Essays",
        "samples": [
            """We tell ourselves stories in order to live. We look for the sermon in the suicide, for the social or moral lesson in the murder of five. We interpret what we see, select the most workable of the multiple choices.""",

            """I write entirely to find out what I'm thinking, what I'm looking at, what I see and what it means. What I want and what I fear.""",

            """Grammar is a piano I play by ear. All I know about grammar is its power.""",

            """A place belongs forever to whoever claims it hardest, remembers it most obsessively, wrenches it from itself, shapes it, renders it, loves it so radically that he remakes it in his image.""",
        ]
    },

    "baldwin": {
        "name": "James Baldwin",
        "source": "Essays and interviews",
        "samples": [
            """Not everything that is faced can be changed, but nothing can be changed until it is faced.""",

            """I imagine one of the reasons people cling to their hates so stubbornly is because they sense, once hate is gone, they will be forced to deal with pain.""",

            """You think your pain and your heartbreak are unprecedented in the history of the world, but then you read. It was books that taught me that the things that tormented me most were the very things that connected me with all the people who were alive, who had ever been alive.""",

            """People are trapped in history and history is trapped in them.""",
        ]
    },

    "huxley": {
        "name": "Aldous Huxley",
        "source": "Essays and letters",
        "samples": [
            """The propagandist's purpose is to make one set of people forget that certain other sets of people are human.""",

            """Facts do not cease to exist because they are ignored.""",

            """There is only one corner of the universe you can be certain of improving, and that's your own self.""",

            """Experience is not what happens to you; it's what you do with what happens to you.""",
        ]
    },

    "woolf": {
        "name": "Virginia Woolf",
        "source": "Essays and diaries",
        "samples": [
            """One cannot think well, love well, sleep well, if one has not dined well.""",

            """You cannot find peace by avoiding life.""",

            """The eyes of others our prisons; their thoughts our cages.""",

            """For most of history, Anonymous was a woman.""",
        ]
    },

    "kafka": {
        "name": "Franz Kafka",
        "source": "Letters and diaries",
        "samples": [
            """A book must be the axe for the frozen sea within us.""",

            """I am a cage, in search of a bird.""",

            """In the fight between you and the world, back the world.""",

            """There is an infinite amount of hope in the universe... but not for us.""",
        ]
    },

    "nietzsche": {
        "name": "Friedrich Nietzsche",
        "source": "Philosophical writings",
        "samples": [
            """He who has a why to live can bear almost any how.""",

            """There are no facts, only interpretations.""",

            """The individual has always had to struggle to keep from being overwhelmed by the tribe. If you try it, you will be lonely often, and sometimes frightened. But no price is too high to pay for the privilege of owning yourself.""",

            """And those who were seen dancing were thought to be insane by those who could not hear the music.""",
        ]
    },

    "camus": {
        "name": "Albert Camus",
        "source": "Essays and notebooks",
        "samples": [
            """In the depth of winter, I finally learned that within me there lay an invincible summer.""",

            """The only way to deal with an unfree world is to become so absolutely free that your very existence is an act of rebellion.""",

            """Man is the only creature who refuses to be what he is.""",

            """I do not believe in God and I am not an atheist.""",
        ]
    },
}


def run_inference_experiment(
    figure_key: str,
    n_hypotheses: int = 30,
) -> dict:
    """
    Run reverse inference on a public figure's writing.
    """
    if figure_key not in PUBLIC_FIGURES:
        raise ValueError(f"Unknown figure: {figure_key}")

    figure = PUBLIC_FIGURES[figure_key]

    print("=" * 70)
    print(f"REVERSE INFERENCE: {figure['name']}")
    print(f"Source: {figure['source']}")
    print("=" * 70)

    engine = ReverseInferenceEngine()

    profile = engine.generate_cognitive_profile(
        figure["samples"],
        n_hypotheses=n_hypotheses,
    )

    print("\n" + "=" * 70)
    print("INFERRED COGNITIVE PROFILE")
    print("=" * 70)

    for category, distribution in profile.items():
        if distribution:
            print(f"\n{category.upper().replace('_', ' ')}:")
            for item, prob in list(distribution.items())[:3]:
                print(f"  {prob:.2%}: {item}")

    return {
        "figure": figure["name"],
        "profile": profile,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", choices=list(PUBLIC_FIGURES.keys()))
    parser.add_argument("--hypotheses", "-n", type=int, default=30)
    args = parser.parse_args()

    result = run_inference_experiment(args.figure, args.hypotheses)

    # Save results
    import os
    os.makedirs("results", exist_ok=True)
    with open(f"results/reverse_{args.figure}.json", "w") as f:
        json.dump(result, f, indent=2)
