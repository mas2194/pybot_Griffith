"""Build a diverse, strictly leak-free module-I/O training & evaluation fixture.

Keeps ``examples/moduleio/calls.jsonl``, ``examples/moduleio/teacher_calls.jsonl.gz``,
and the held-out 18-call benchmark ``experiments/moduleio/model.pt.validation.jsonl.gz``
completely untouched, and verifies programmatically that none of the held-out
entities, group IDs, or exact inputs appear in the generated dataset.
"""
import json
from pathlib import Path
import random

from moduleio import codec
from moduleio.data import read_jsonl


ROOT = Path(__file__).resolve().parents[2]
HELD_OUT_PATH = ROOT / "experiments" / "moduleio" / "model.pt.validation.jsonl.gz"

FORBIDDEN_COLOURS = {"purple", "black"}
FORBIDDEN_AGES = {(11, 14), (14, 11)}
FORBIDDEN_NAMES = {"max", "kim"}

COLOURS = (
    "red", "blue", "green", "orange", "silver", "white", "pink", "yellow", "teal", "brown",
    "amber", "azure", "beige", "bronze", "cerise", "charcoal", "cobalt", "copper", "coral",
    "cream", "crimson", "cyan", "emerald", "gold", "grey", "gray", "indigo", "ivory", "jade",
    "khaki", "lavender", "lilac", "lime", "magenta", "maroon", "mint", "mustard", "navy",
    "ochre", "olive", "peach", "plum", "rose", "ruby", "rust", "sage", "salmon", "scarlet",
    "sepia", "slate", "tan", "taupe", "turquoise", "violet", "burgundy", "dark green",
    "sky blue", "navy blue", "mint green", "light blue",
)

FAV_THINGS = (
    "colour", "color", "food", "film", "book", "drink", "song", "sport",
    "animal", "fruit", "city", "band", "season", "game", "flower", "dish",
    "hobby", "dessert", "tree", "country", "subject", "paint", "gem", "tea",
)

OTHER_FAVOURITES = (
    ("food", "curry"), ("food", "pizza"), ("food", "sushi"), ("food", "pasta"),
    ("food", "tacos"), ("food", "ramen"), ("film", "alien"), ("film", "dune"),
    ("film", "jaws"), ("film", "matrix"), ("book", "hobbit"), ("book", "emma"),
    ("drink", "coffee"), ("drink", "tea"), ("drink", "lemonade"), ("drink", "water"),
    ("song", "yesterday"), ("sport", "tennis"), ("sport", "cycling"), ("sport", "swimming"),
    ("animal", "otter"), ("animal", "fox"), ("animal", "eagle"), ("fruit", "apple"),
    ("fruit", "mango"), ("fruit", "kiwi"), ("fruit", "grape"), ("city", "tokyo"),
    ("city", "paris"), ("city", "oslo"), ("city", "rome"), ("city", "berlin"),
    ("band", "blur"), ("band", "queen"), ("season", "autumn"), ("season", "winter"),
    ("game", "chess"), ("game", "poker"), ("flower", "daisy"), ("flower", "orchid"),
    ("dish", "risotto"), ("dish", "quiche"), ("dessert", "gelato"), ("tree", "oak"),
    ("country", "japan"), ("subject", "physics"),
)

AGE_GROUPS = (
    "kids", "children", "sons", "daughters", "boys", "girls",
    "parents", "brothers", "sisters", "twins", "cousins", "grandkids",
)

BASE_NAMES = (
    "Sam", "Kay", "Alex", "Jo", "Lee", "Robin", "Ada", "Ben", "Cara", "Dan",
    "Eli", "Fay", "Gus", "Hana", "Ian", "Jude", "Kate", "Leo", "Mia", "Noah",
    "Omar", "Paul", "Quinn", "Rosa", "Sara", "Tara", "Uma", "Vera", "Will", "Xena",
    "Yuri", "Zoe", "Arthur", "Bella", "Chloe", "David", "Elena", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Liam", "Maya", "Nora", "Owen", "Priya", "Ruby", "Simon", "Tessa",
    "Victor", "Wendy", "Zack", "Al", "Bo", "Cy", "Ed", "Fe", "Gi", "Mo",
    "Oz", "Ty", "Vi", "Yu", "Amara", "Callum", "Devon", "Freya", "Gideon", "Hazel",
    "Jonah", "Klara", "Nadia", "Oscar", "Petra", "Ronan", "Sienna", "Tobias", "Ulysses", "Wren",
)

CAKE_FAMILIES = (
    ("cake-coffee-filling", "what filling for a coffee sponge cake", "why is my cake dry"),
    ("cake-how-fill", "how do i fill a cake", "how long does cake keep"),
    ("cake-between-layers", "what goes between cake layers", "how do i cut my hair"),
    ("cake-victoria-cream", "what cream goes in a victoria sponge", "what is half of 46"),
    ("cake-lemon-middle", "what goes in the middle of a lemon cake", "why did my cake sink"),
    ("cake-put-between", "what can i put between the layers of a cake", "why is my sponge dense"),
    ("cake-layer-it", "how do i layer a sponge cake", "how many calories in a cake"),
    ("cake-split-half", "how do i cut a cake in half", "what size tin for a birthday cake"),
    ("cake-jam-sponge", "what jam goes in a victoria sponge", "why did my cake burn"),
    ("cake-sandwich-it", "how do i split a sandwich cake", "how do i ice a cake"),
    ("cake-chocolate-middle", "what filling goes in a chocolate cake", "can you freeze a sponge cake"),
    ("cake-two-layers", "what cream goes between two layers of a cake", "why is my cake raw in the middle"),
    ("cake-halve-sponge", "how do i halve a cake", "how to make a vegan cake"),
    ("cake-choc-sponge", "what cream goes in a chocolate sponge", "how many people does a cake feed"),
    ("cake-sponge-fill", "what filling goes in a sponge cake", "i cut myself a slice of cake"),
    ("cake-sponge-jam", "what jam for a sandwich cake", "my cake sank in the middle"),
)


def _synthetic_words(count, *, seed, min_len=3, max_len=9, forbidden=()):
    """Generate deterministic pronounceable alphabetic tokens to prevent lookup memorization."""
    rng = random.Random(seed)
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    words = []
    seen = set(w.lower() for w in forbidden)
    # Include all single letters in consonant starts to ensure full alphabet coverage.
    while len(words) < count:
        target_len = rng.randint(min_len, max_len)
        chars = []
        use_c = rng.random() < 0.75
        while len(chars) < target_len:
            if use_c:
                ch = consonants[len(words) % len(consonants)] if not chars else rng.choice(consonants)
                chars.append(ch)
                if len(chars) < target_len and rng.random() < 0.12:
                    chars.append(ch)  # doubled letters like pp/ll/bb
            else:
                chars.append(rng.choice(vowels))
            use_c = not use_c
        word = "".join(chars[:target_len])
        if word in seen or word in {"not", "now", "are", "was", "yes", "all", "new", "old", "the"}:
            continue
        seen.add(word)
        words.append(word)
    return words


def _synthetic_age_pairs(count, *, seed):
    rng = random.Random(seed)
    # Include ample same-tens-digit pairs so youngest/oldest generalizes when tens digits tie.
    pairs = [
        (1, 3), (2, 4), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (10, 13),
        (10, 15), (10, 19), (11, 16), (11, 18), (12, 15), (12, 17), (13, 19), (14, 17), (14, 19),
        (15, 11), (15, 19), (16, 12), (17, 13), (18, 13), (19, 11), (19, 15),
        (21, 28), (22, 27), (27, 22), (31, 39), (33, 38), (38, 32), (41, 47), (44, 49), (46, 42),
        (51, 58), (55, 61), (59, 53), (62, 68), (67, 61), (72, 79), (78, 71), (81, 88), (87, 82),
    ]
    seen = set(pairs) | FORBIDDEN_AGES
    while len(pairs) < count:
        if len(pairs) % 3 == 0:
            tens = rng.randint(1, 8) * 10
            a = tens + rng.randint(0, 9)
            b = tens + rng.randint(0, 9)
        else:
            a = rng.randint(1, 89)
            b = rng.randint(1, 89)
        if a < 1 or b < 1 or (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b))
    return pairs


def examples():
    def call(group, module, hook, text, memory):
        return {
            "conversation_id": group,
            "module": module,
            "hook": hook,
            "seed": 0,
            "arguments": {"text": text, "memory": memory},
        }

    # 1. Favourites: 60 real colours + 46 real pairs + 220 synthetic words = 326 distinct entities
    synth_fav_words = _synthetic_words(
        220, seed=101, min_len=3, max_len=9,
        forbidden=set(COLOURS) | {v for _, v in OTHER_FAVOURITES} | FORBIDDEN_COLOURS,
    )
    fav_items = [("colour", c) for c in COLOURS] + list(OTHER_FAVOURITES) + [
        (FAV_THINGS[idx % len(FAV_THINGS)], w) for idx, w in enumerate(synth_fav_words)
    ]
    for idx, (thing, value) in enumerate(fav_items):
        assert value.lower() not in FORBIDDEN_COLOURS
        group = f"div-fav-{idx}-{thing}-{value.replace(' ', '-')}"
        yield call(group, "favourites", "observe", f"my favourite {thing} is {value}", {})
        if thing == "colour":
            yield call(group, "favourites", "observe", f"my favorite color is {value}", {})
            yield call(group, "favourites", "respond", "what is my favorite color",
                       {"user": {"favourites": {"color": value}}})
        elif idx % 2 == 0:
            yield call(group, "favourites", "observe", f"my favorite {thing} is {value}", {})
        else:
            yield call(group, "favourites", "observe", f"my fav {thing} is {value}", {})
        if idx % 6 == 0:
            yield call(group, "favourites", "observe", f"my mum's favourite {thing} is {value}", {})
        if idx % 7 == 0:
            yield call(group, "favourites", "observe", "hello there", {})
        memory = {"user": {"favourites": {thing: value}}}
        yield call(group, "favourites", "respond", f"what is my favourite {thing}", memory)
        if idx % 3 == 0:
            yield call(group, "favourites", "respond", f"whats my favorite {thing}", memory)
        if idx % 2 == 0:
            inactive_line = ("how do i bake a cake", "hello", "what is the weather", "why is the sky blue")[idx % 4]
            yield call(group, "favourites", "respond", inactive_line, memory)
        if idx % 8 == 0:
            other_thing = "food" if thing != "food" else "colour"
            yield call(group, "favourites", "respond", f"what is my favourite {other_thing}", memory)

    # 2. Group ages: 200 diverse age pairs
    age_pairs = _synthetic_age_pairs(200, seed=202)
    for idx, (a, b) in enumerate(age_pairs):
        assert (a, b) not in FORBIDDEN_AGES
        word = AGE_GROUPS[idx % len(AGE_GROUPS)]
        alt_word = AGE_GROUPS[(idx + 1) % len(AGE_GROUPS)]
        group = f"div-ages-{idx}-{a}-{b}"
        said = f"my {word} are {a} and {b}"
        yield call(group, "group_ages", "observe", said, {})
        yield call(group, "group_ages", "observe", f"my {alt_word} are {a} and {b}", {})
        if idx % 6 == 0:
            yield call(group, "group_ages", "observe", f"i have {a} apples", {})
        memory = {"group_ages": {"word": word, "ages": [a, b], "line": said}}
        yield call(group, "group_ages", "respond", "how old are they", memory)
        yield call(group, "group_ages", "respond", "how old is my youngest", memory)
        if idx % 2 == 0:
            yield call(group, "group_ages", "respond", "how old is my oldest", memory)
        if idx % 5 == 0:
            yield call(group, "group_ages", "respond", "how do i bake a cake", memory)

    # 3. Name taken back: 80 real names + 240 synthetic names = 320 distinct names
    synth_names = _synthetic_words(
        240, seed=303, min_len=2, max_len=8,
        forbidden=set(BASE_NAMES) | FORBIDDEN_NAMES,
    )
    all_names = list(BASE_NAMES) + [w.capitalize() for w in synth_names]
    for idx, name in enumerate(all_names):
        assert name.lower() not in FORBIDDEN_NAMES
        other = all_names[(idx + 11) % len(all_names)]
        group = f"div-name-{idx}-{name.lower()}"
        mem = {"user": {"name": name}}
        yield call(group, "name_taken_back", "observe", "that is not my name", mem)
        yield call(group, "name_taken_back", "observe", "my name is wrong", mem)
        if idx % 2 == 0:
            yield call(group, "name_taken_back", "observe", "hello", mem)
        else:
            yield call(group, "name_taken_back", "observe", f"im not {name.lower()}", mem)
        if idx % 4 == 0:
            yield call(group, "name_taken_back", "observe", f"dont call me {name.lower()}", mem)
        if idx % 5 == 0:
            yield call(group, "name_taken_back", "observe", "thats not my name", {"user": {"name": name, "age": 20 + idx % 50}})
        if idx % 6 == 0:
            yield call(group, "name_taken_back", "observe", f"im not {other.lower()}", mem)
            yield call(group, "name_taken_back", "observe", "that is not my name", {})

    # 4. Cake between its layers (16 independent groups so validation covers cake too)
    for idx, (group, active_q, inactive_q) in enumerate(CAKE_FAMILIES):
        mem = {} if idx % 3 else {"user": {"name": "Sam"}}
        yield call(group, "a_cake_between_its_layers", "respond", active_q, mem)
        yield call(group, "a_cake_between_its_layers", "respond", inactive_q, mem)


def verify_no_leakage(rows):
    if not HELD_OUT_PATH.exists():
        return
    held_out = read_jsonl(HELD_OUT_PATH)
    held_groups = {row["conversation_id"] for row in held_out}
    held_inputs = set()
    for row in held_out:
        inp = codec.decode(row["input"])
        held_inputs.add((row["module"], row["hook"], inp["text"], codec.dumps(inp["memory"])))
    for row in rows:
        assert row["conversation_id"] not in held_groups, f"leaked group: {row['conversation_id']}"
        key = (row["module"], row["hook"], row["arguments"]["text"], codec.dumps(row["arguments"]["memory"]))
        assert key not in held_inputs, f"leaked exact input: {key}"


if __name__ == "__main__":
    rows = list(examples())
    verify_no_leakage(rows)
    path = Path(__file__).with_name("diverse_calls.jsonl")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} leak-free specs to {path}")
