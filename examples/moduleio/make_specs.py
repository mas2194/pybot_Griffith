"""Rebuild the small, explicitly bounded module-I/O experiment input fixtures.

These are independent calls, not a measure of whole-chat quality. Scenario
families share IDs so paraphrases and values are never split row-by-row. The
fixtures contain inputs only; actual PyBot hooks supply every target.
"""
import json
from pathlib import Path


def examples():
    def call(group, module, hook, text, memory):
        return {"conversation_id": group, "module": module, "hook": hook,
                "seed": 0, "arguments": {"text": text, "memory": memory}}

    for colour in ("red", "blue", "green", "purple", "orange", "silver",
                   "black", "white", "pink", "yellow", "teal", "brown"):
        group = "favourite-colour-" + colour
        yield call(group, "favourites", "observe", "my favourite colour is " + colour, {})
        yield call(group, "favourites", "observe", "my favorite color is " + colour, {})
        memory = {"user": {"favourites": {"colour": colour}}}
        yield call(group, "favourites", "respond", "what is my favourite colour", memory)
        yield call(group, "favourites", "respond", "how do i bake a cake", memory)
    for a, b in ((2, 4), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10),
                 (8, 11), (9, 12), (10, 13), (11, 14)):
        group = f"ages-{a}-{b}"
        said = f"my kids are {a} and {b}"
        yield call(group, "group_ages", "observe", said, {})
        yield call(group, "group_ages", "observe", f"my children are {a} and {b}", {})
        memory = {"group_ages": {"word": "kids", "ages": [a, b], "line": said}}
        yield call(group, "group_ages", "respond", "how old are they", memory)
        yield call(group, "group_ages", "respond", "how old is my youngest", memory)
    for name in ("Sam", "Kay", "Alex", "Jo", "Lee", "Max", "Robin", "Kim"):
        group = "name-correction-" + name.lower()
        yield call(group, "name_taken_back", "observe", "that is not my name", {"user": {"name": name}})
        yield call(group, "name_taken_back", "observe", "my name is wrong", {"user": {"name": name}})
        yield call(group, "name_taken_back", "observe", "hello", {"user": {"name": name}})
    for index, text in enumerate((
            "what filling for a coffee cake", "how do i fill a cake",
            "what goes between cake layers", "what cream goes in a victoria sponge",
            "why is my cake dry", "how long does cake keep",
            "how do i cut my hair", "what is half of 46")):
        # Keep all cake paraphrases together: one fixed response paragraph would
        # otherwise make a random row split look like broad generalization.
        yield call("cake-boundary", "a_cake_between_its_layers", "respond", text, {})


if __name__ == "__main__":
    path = Path(__file__).with_name("calls.jsonl")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in examples()),
                    encoding="utf-8")
    print(path)
