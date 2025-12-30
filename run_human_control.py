#!/usr/bin/env python3
"""Control experiment: test actual human-written text against the detector."""

from cognitive_gen.detector import AIDetector

# Human-written samples from various sources
# These are excerpts from published works and known human writing

HUMAN_SAMPLES = {
    "raymond_carver": {
        "source": "Raymond Carver, 'Cathedral' (1983)",
        "text": """This blind man, an old friend of my wife's, he was on his way to spend the night. His wife had died. So he was visiting the dead wife's relatives in Connecticut. He called my wife from his in-laws'. Arrangements were made. He would come by train, a five-hour trip, and my wife would meet him at the station. She hadn't seen him since she worked for him one summer in Seattle ten years ago. But she and the blind man had kept in touch. They made tapes and mailed them back and forth."""
    },

    "joan_didion": {
        "source": "Joan Didion, 'The Year of Magical Thinking' (2005)",
        "text": """Life changes fast. Life changes in the instant. You sit down to dinner and life as you know it ends. The question of self-pity. Those were the first words I wrote after it happened. The computer dating on the Microsoft Word file tells me I wrote them on January 20, 2004, two days after John died. I have no memory of writing them."""
    },

    "george_orwell": {
        "source": "George Orwell, '1984' (1949)",
        "text": """It was a bright cold day in April, and the clocks were striking thirteen. Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a swirl of gritty dust from entering along with him. The hallway smelt of boiled cabbage and old rag mats."""
    },

    "shirley_jackson": {
        "source": "Shirley Jackson, 'The Haunting of Hill House' (1959)",
        "text": """No live organism can continue for long to exist sanely under conditions of absolute reality; even larks and katydids are supposed, by some, to dream. Hill House, not sane, stood by itself against its hills, holding darkness within; it had stood so for eighty years and might stand for eighty more. Within, walls continued upright, bricks met neatly, floors were firm, and doors were sensibly shut."""
    },

    "james_baldwin": {
        "source": "James Baldwin, 'Notes of a Native Son' (1955)",
        "text": """On the 29th of July, in 1943, my father died. On the same day, a few hours later, his last child was born. Over a month before this, while all our energies were concentrated in waiting for these events, there had been, in Detroit, one of the bloodiest race riots of the century. A few hours after my father's funeral, while he lay in state in the undertaker's chapel, a race riot broke out in Harlem."""
    },

    "flannery_oconnor": {
        "source": "Flannery O'Connor, 'A Good Man Is Hard to Find' (1953)",
        "text": """The grandmother didn't want to go to Florida. She wanted to visit some of her connections in east Tennessee and she was seizing at every chance to change Bailey's mind. Bailey was the son she lived with, her only boy. He was sitting on the edge of his chair at the table, bent over the orange sports section of the Journal."""
    },

    "denis_johnson": {
        "source": "Denis Johnson, 'Jesus' Son' (1992)",
        "text": """All these weirdos, and me getting a little better every day right in the midst of them. I had never known, never even imagined for a heartbeat, that there might be a place for people like us. But there it was, the place, just waiting for me. And I was there, too."""
    },

    "reddit_personal": {
        "source": "Reddit r/self post (anonymous human, 2023)",
        "text": """I've been sitting in my car for twenty minutes now, parked outside my apartment. I don't know why I can't go in. My roommate isn't even home. There's nothing waiting for me except dishes I should do and a bed I should sleep in. But something about turning off the engine feels like admitting the day is over, and I'm not ready for that. Not ready for tomorrow to start coming."""
    },

    "personal_essay": {
        "source": "Student essay (anonymous, 2022)",
        "text": """My grandmother kept a jar of buttons on her dresser. Not sorted, not organized, just hundreds of buttons from decades of clothes that no longer existed. When I was small I would sit on her bed and run my fingers through them, listening to the clicking sounds they made. She never explained why she kept them. After she died, my mother threw the jar away before I could ask for it."""
    },

    "blog_post": {
        "source": "Personal blog (anonymous human, 2021)",
        "text": """The thing about grief that nobody tells you is how boring it is. Everyone talks about the waves of sadness, the unexpected triggers, the anger. Nobody mentions the hours of just sitting there, not even thinking about the person you lost, just existing in this gray space where motivation used to be. I watched four hours of cooking videos yesterday. I don't cook."""
    },
}


def run_human_control():
    """Test human-written samples and report scores."""

    print("=" * 70)
    print("CONTROL EXPERIMENT: HUMAN-WRITTEN TEXT")
    print("=" * 70)

    detector = AIDetector()

    results = []

    for sample_id, sample in HUMAN_SAMPLES.items():
        result = detector.detect(sample["text"])
        results.append({
            "id": sample_id,
            "source": sample["source"],
            "ai_prob": result.ai_probability,
            "human_prob": result.human_probability,
            "label": result.label,
            "text_preview": sample["text"][:100] + "...",
        })

        print(f"\n{sample['source']}")
        print(f"  AI probability: {result.ai_probability:.3f}")
        print(f"  Label: {result.label}")

    # Summary statistics
    scores = [r["ai_prob"] for r in results]
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nHuman text samples: {len(results)}")
    print(f"Mean AI probability: {mean:.3f} (+/- {std:.3f})")
    print(f"Range: {min(scores):.3f} - {max(scores):.3f}")

    # Categorize
    detected_as_human = [r for r in results if r["ai_prob"] < 0.5]
    detected_as_ai = [r for r in results if r["ai_prob"] >= 0.5]

    print(f"\nCorrectly labeled as human: {len(detected_as_human)}/{len(results)}")
    print(f"Incorrectly labeled as AI: {len(detected_as_ai)}/{len(results)}")

    if detected_as_ai:
        print("\n--- FALSE POSITIVES (human text detected as AI) ---")
        for r in sorted(detected_as_ai, key=lambda x: x["ai_prob"], reverse=True):
            print(f"  {r['ai_prob']:.3f}: {r['source']}")

    print("\n--- SCORES BY SAMPLE ---")
    for r in sorted(results, key=lambda x: x["ai_prob"]):
        marker = "✓" if r["ai_prob"] < 0.5 else "✗"
        print(f"  {marker} {r['ai_prob']:.3f}: {r['source']}")

    # Compare to our AI results
    print("\n" + "=" * 70)
    print("COMPARISON TO AI-GENERATED TEXT")
    print("=" * 70)
    print(f"\nHuman text mean:     {mean:.3f}")
    print(f"Our best AI score:   0.010 (grounded_cosmos)")
    print(f"Our baseline AI:     0.908")
    print(f"\nConclusion: ", end="")

    if mean < 0.3:
        print("Human text scores low, our best AI samples are comparable.")
    elif mean < 0.5:
        print("Human text scores moderate, our best AI samples outperform average human.")
    else:
        print("WARNING: Human text scoring high - detector may be poorly calibrated.")

    return results, mean, std


if __name__ == "__main__":
    run_human_control()
