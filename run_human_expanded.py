#!/usr/bin/env python3
"""Expanded human text control - searching for patterns in what scores as human."""

from cognitive_gen.detector import AIDetector

# Expanded set organized by style/era
HUMAN_SAMPLES = {
    # === DIRECT/PLAIN STYLE (Orwell-like) ===
    "hemingway_old_man": {
        "source": "Ernest Hemingway, 'The Old Man and the Sea' (1952)",
        "category": "direct_prose",
        "text": """He was an old man who fished alone in a skiff in the Gulf Stream and he had gone eighty-four days now without taking a fish. In the first forty days a boy had been with him. But after forty days without a fish the boy's parents had told him that the old man was now definitely and finally salao, which is the worst form of unlucky."""
    },
    "hemingway_hills": {
        "source": "Ernest Hemingway, 'Hills Like White Elephants' (1927)",
        "category": "direct_prose",
        "text": """The hills across the valley of the Ebro were long and white. On this side there was no shade and no trees and the station was between two lines of rails in the sun. Close against the side of the station there was the warm shadow of the building and a curtain, made of strings of bamboo beads, hung across the open door into the bar, to keep out flies."""
    },
    "camus_stranger": {
        "source": "Albert Camus, 'The Stranger' (1942)",
        "category": "direct_prose",
        "text": """Maman died today. Or yesterday maybe, I don't know. I got a telegram from the home: "Mother deceased. Funeral tomorrow. Faithfully yours." That doesn't mean anything. Maybe it was yesterday. The old people's home is at Marengo, about eighty kilometers from Algiers. I'll take the two o'clock bus and get there in the afternoon."""
    },
    "orwell_animal_farm": {
        "source": "George Orwell, 'Animal Farm' (1945)",
        "category": "direct_prose",
        "text": """Mr. Jones, of the Manor Farm, had locked the hen-houses for the night, but was too drunk to remember to shut the pop-holes. With the ring of light from his lantern dancing from side to side, he lurched across the yard, kicked off his boots at the back door, drew himself a last glass of beer from the barrel in the scullery, and made his way up to bed."""
    },
    "orwell_shooting": {
        "source": "George Orwell, 'Shooting an Elephant' (1936)",
        "category": "direct_prose",
        "text": """In Moulmein, in lower Burma, I was hated by large numbers of people – the only time in my life that I have been important enough for this to happen to me. I was sub-divisional police officer of the town, and in an aimless, petty kind of way anti-European feeling was very bitter."""
    },
    "mccarthy_road": {
        "source": "Cormac McCarthy, 'The Road' (2006)",
        "category": "direct_prose",
        "text": """When he woke in the woods in the dark and the cold of the night he'd reach out to touch the child sleeping beside him. Nights dark beyond darkness and the days more gray each one than what had gone before. Like the onset of some cold glaucoma dimming away the world."""
    },
    "carver_what": {
        "source": "Raymond Carver, 'What We Talk About When We Talk About Love' (1981)",
        "category": "direct_prose",
        "text": """My friend Mel McGinnis was talking. Mel McGinnis is a cardiologist, and sometimes that gives him the right. The four of us were sitting around his kitchen table drinking gin. Sunlight filled the kitchen from the big window behind the sink. There were Mel and me and his second wife, Teresa—Terri, we called her—and my wife, Laura."""
    },

    # === LITERARY/ORNATE STYLE ===
    "nabokov_lolita": {
        "source": "Vladimir Nabokov, 'Lolita' (1955)",
        "category": "literary_ornate",
        "text": """Lolita, light of my life, fire of my loins. My sin, my soul. Lo-lee-ta: the tip of the tongue taking a trip of three steps down the palate to tap, at three, on the teeth. Lo. Lee. Ta. She was Lo, plain Lo, in the morning, standing four feet ten in one sock. She was Lola in slacks. She was Dolly at school."""
    },
    "woolf_dalloway": {
        "source": "Virginia Woolf, 'Mrs Dalloway' (1925)",
        "category": "literary_ornate",
        "text": """Mrs. Dalloway said she would buy the flowers herself. For Lucy had her work cut out for her. The doors would be taken off their hinges; Rumpelmayer's men were coming. And then, thought Clarissa Dalloway, what a morning—fresh as if issued to children on a beach."""
    },
    "faulkner_sound": {
        "source": "William Faulkner, 'The Sound and the Fury' (1929)",
        "category": "literary_ornate",
        "text": """Through the fence, between the curling flower spaces, I could see them hitting. They were coming toward where the flag was and I went along the fence. Luster was hunting in the grass by the flower tree. They took the flag out, and they were hitting. Then they put the flag back and they went to the table, and he hit and the other hit."""
    },
    "pynchon_gravity": {
        "source": "Thomas Pynchon, 'Gravity's Rainbow' (1973)",
        "category": "literary_ornate",
        "text": """A screaming comes across the sky. It has happened before, but there is nothing to compare it to now. It is too late. The Evacuation still proceeds, but it's all theatre. There are no lights inside the cars. No light anywhere. Above him lift girders old as an iron queen, and glass somewhere far above that would let the light of day through."""
    },
    "morrison_beloved": {
        "source": "Toni Morrison, 'Beloved' (1987)",
        "category": "literary_ornate",
        "text": """124 was spiteful. Full of a baby's venom. The women in the house knew it and so did the children. For years each put up with the spite in his own way, but by 1873 Sethe and her daughter Denver were its only victims. The grandmother, Baby Suggs, was dead, and the sons, Howard and Buglar, had run away by the time they were thirteen years old."""
    },

    # === GENRE FICTION ===
    "king_shining": {
        "source": "Stephen King, 'The Shining' (1977)",
        "category": "genre_fiction",
        "text": """Jack Torrance thought: Officious little prick. Ullman stood five-five, and when he moved, it was with the prissy speed that seems to be the exclusive domain of all small plump men. The toes of his feet were polished to a high sheen, and he expected a similar shine from his employees."""
    },
    "chandler_sleep": {
        "source": "Raymond Chandler, 'The Big Sleep' (1939)",
        "category": "genre_fiction",
        "text": """It was about eleven o'clock in the morning, mid October, with the sun not shining and a look of hard wet rain in the clearness of the foothills. I was wearing my powder-blue suit, with dark blue shirt, tie and display handkerchief, black brogues, black wool socks with dark blue clocks on them. I was neat, clean, shaved and sober, and I didn't care who knew it."""
    },
    "leguin_left_hand": {
        "source": "Ursula K. Le Guin, 'The Left Hand of Darkness' (1969)",
        "category": "genre_fiction",
        "text": """I'll make my report as if I told a story, for I was taught as a child on my homeworld that Truth is a matter of the imagination. The soundest fact may fail or prevail in the style of its telling: like that singular organic jewel of our seas, which grows brighter as one woman wears it and, worn by another, dulls and goes to dust."""
    },
    "dick_androids": {
        "source": "Philip K. Dick, 'Do Androids Dream of Electric Sheep?' (1968)",
        "category": "genre_fiction",
        "text": """A merry little surge of electricity piped by automatic alarm from the mood organ beside his bed awakened Rick Deckard. Surprised—it always surprised him to find himself awake without prior notice—he rose from the bed, stood up in his multicolored pajamas, and stretched."""
    },

    # === NON-FICTION/ESSAYS ===
    "didion_slouching": {
        "source": "Joan Didion, 'Slouching Towards Bethlehem' (1968)",
        "category": "nonfiction",
        "text": """The center was not holding. It was a country of bankruptcy notices and public-Loss of Credit notices. The divorce notices. The want-to-sell notices. The want-to-buy notices. It was a country in which families routinely disappeared, trailing bad checks and repossession papers."""
    },
    "orwell_politics": {
        "source": "George Orwell, 'Politics and the English Language' (1946)",
        "category": "nonfiction",
        "text": """Most people who bother with the matter at all would admit that the English language is in a bad way, but it is generally assumed that we cannot by conscious action do anything about it. Our civilization is decadent and our language—so the argument runs—must inevitably share in the general collapse."""
    },
    "white_once_more": {
        "source": "E.B. White, 'Once More to the Lake' (1941)",
        "category": "nonfiction",
        "text": """One summer, along about 1904, my father rented a camp on a lake in Maine and took us all there for the month of August. We all got ringworm from some kittens and had to rub Pond's Extract on our arms and legs night and morning, and my father rolled over in a canoe with all his clothes on."""
    },
    "baldwin_fire": {
        "source": "James Baldwin, 'The Fire Next Time' (1963)",
        "category": "nonfiction",
        "text": """I underwent, during the summer that I became fourteen, a prolonged religious crisis. I use the word "religious" in the common, and arbitrary, sense, meaning that I then discovered God, His saints and angels, and His blazing Hell. And since I had been born in a Christian nation, I accepted this Deity as the only one."""
    },

    # === CONTEMPORARY ===
    "smith_white_teeth": {
        "source": "Zadie Smith, 'White Teeth' (2000)",
        "category": "contemporary",
        "text": """Early in the morning, late in the century, Cricklewood Broadway. At 06.27 hours on 1 January 1975, Alfred Archibald Jones was dressed in corduroy and sat in a fume-filled Cavalier Musketeer Estate face down on the steering wheel, hoping the judgment would not be too heavy upon him."""
    },
    "eugenides_virgin": {
        "source": "Jeffrey Eugenides, 'The Virgin Suicides' (1993)",
        "category": "contemporary",
        "text": """On the morning the last Lisbon daughter took her turn at suicide—it was Mary this time, and target sleeping pills, as Therese, the first to go, had done—the two paramedics arrived at the house knowing exactly where the knife drawer was, and the gas oven, and the beam in the basement from which it was possible to tie a rope."""
    },
    "lahiri_namesake": {
        "source": "Jhumpa Lahiri, 'The Namesake' (2003)",
        "category": "contemporary",
        "text": """On a sticky August evening two weeks before her due date, Ashima Ganguli stands in the kitchen of a Central Square apartment, combining Rice Krispies and Planters peanuts and chopped red onion in a bowl. She adds salt, lemon juice, thin slices of green chili pepper, wishing there were mustard oil to pour into the mix."""
    },

    # === JOURNALISM/REPORTAGE ===
    "hersey_hiroshima": {
        "source": "John Hersey, 'Hiroshima' (1946)",
        "category": "journalism",
        "text": """At exactly fifteen minutes past eight in the morning, on August 6, 1945, Japanese time, at the moment when the atomic bomb flashed above Hiroshima, Miss Toshiko Sasaki, a clerk in the personnel department of the East Asia Tin Works, had just sat down at her place in the plant office and was turning her head to speak to the girl at the next desk."""
    },
    "capote_cold_blood": {
        "source": "Truman Capote, 'In Cold Blood' (1966)",
        "category": "journalism",
        "text": """The village of Holcomb stands on the high wheat plains of western Kansas, a lonesome area that other Kansans call "out there." Some seventy miles east of the Colorado border, the countryside, with its hard blue skies and desert-clear air, has an atmosphere that is rather more Far West than Middle West."""
    },
    "wolfe_right_stuff": {
        "source": "Tom Wolfe, 'The Right Stuff' (1979)",
        "category": "journalism",
        "text": """Within five minutes, or ten minutes, no more than that, three of the others had called her on the telephone to ask her if she had heard that something bad had happened out there. "Jane, this is Alice. Listen, I just got a call from Betty, and she said she heard something's happened out there. Have you heard anything?" That was the way they talked."""
    },
}


def run_expanded_control():
    """Test expanded human samples and look for patterns."""

    print("=" * 70)
    print("EXPANDED HUMAN TEXT CONTROL")
    print("Searching for patterns in what scores as human")
    print("=" * 70)

    detector = AIDetector()

    results = []
    by_category = {}

    for sample_id, sample in HUMAN_SAMPLES.items():
        result = detector.detect(sample["text"])

        data = {
            "id": sample_id,
            "source": sample["source"],
            "category": sample["category"],
            "ai_prob": result.ai_probability,
            "label": result.label,
        }
        results.append(data)

        # Track by category
        cat = sample["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(data)

    # Sort all results by score
    results.sort(key=lambda x: x["ai_prob"])

    print("\n" + "=" * 70)
    print("ALL RESULTS (sorted by AI probability)")
    print("=" * 70)

    for r in results:
        marker = "✓ HUMAN" if r["ai_prob"] < 0.5 else "✗ AI"
        print(f"{r['ai_prob']:.3f} [{marker:8}] {r['source']}")

    # Stats by category
    print("\n" + "=" * 70)
    print("BY CATEGORY")
    print("=" * 70)

    category_stats = {}
    for cat, samples in by_category.items():
        scores = [s["ai_prob"] for s in samples]
        mean = sum(scores) / len(scores)
        human_count = sum(1 for s in samples if s["ai_prob"] < 0.5)
        category_stats[cat] = {
            "mean": mean,
            "min": min(scores),
            "max": max(scores),
            "human_count": human_count,
            "total": len(samples),
        }

    for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]["mean"]):
        print(f"\n{cat.upper()}:")
        print(f"  Mean: {stats['mean']:.3f} | Range: {stats['min']:.3f}-{stats['max']:.3f}")
        print(f"  Classified as human: {stats['human_count']}/{stats['total']}")

    # Overall stats
    all_scores = [r["ai_prob"] for r in results]
    mean = sum(all_scores) / len(all_scores)
    human_count = sum(1 for r in results if r["ai_prob"] < 0.5)

    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"\nTotal samples: {len(results)}")
    print(f"Mean AI probability: {mean:.3f}")
    print(f"Range: {min(all_scores):.3f} - {max(all_scores):.3f}")
    print(f"Classified as human: {human_count}/{len(results)} ({100*human_count/len(results):.1f}%)")

    # Show the human-passing samples
    human_passing = [r for r in results if r["ai_prob"] < 0.5]
    print(f"\n" + "=" * 70)
    print(f"SAMPLES THAT PASSED AS HUMAN ({len(human_passing)})")
    print("=" * 70)

    for r in human_passing:
        print(f"\n{r['ai_prob']:.3f}: {r['source']}")
        print(f"       Category: {r['category']}")
        # Print the actual text
        text = HUMAN_SAMPLES[r['id']]['text'][:300]
        print(f"       Text: {text}...")

    return results, by_category, category_stats


if __name__ == "__main__":
    run_expanded_control()
