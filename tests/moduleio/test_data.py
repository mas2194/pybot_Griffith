import copy
import tempfile
import unittest
from pathlib import Path

from moduleio import codec
from moduleio.data import (input_for_call, prepare_records, read_jsonl,
                           record_to_example, split_examples, write_jsonl)


def record(group="family", text="hello", memory=None, after=None):
    before = {} if memory is None else memory
    return {"schema_version": 1, "conversation_id": group,
            "module": "example", "hook": "respond",
            "input": codec.encode({"text": text, "memory": before}),
            "output": codec.encode({"return": None, "memory": before if after is None else after,
                                    "exception": None}),
            "metadata": {"fixture": "isolated_seeded_call", "seed": 0,
                         "environment_date": "2026-11-21"}}


class DataContractTests(unittest.TestCase):
    def test_targets_never_affect_input_and_none_mutation_is_active(self):
        one = record(memory={"x": (1, 2)}, after={"x": (1, 3)})
        two = copy.deepcopy(one)
        two["output"] = codec.encode({"return": "different", "memory": {}, "exception": None})
        a, b = record_to_example(one), record_to_example(two)
        self.assertEqual(a["input_text"], b["input_text"])
        self.assertNotEqual(a["output_text"], b["output_text"])
        self.assertTrue(a["active"])
        self.assertEqual(codec.loads(a["output_text"])["memory"]["x"], (1, 3))

    def test_exact_bound_includes_eos_and_does_not_truncate(self):
        row = record()
        example = record_to_example(row)
        required = len(example["input_text"].encode()) + len(example["output_text"].encode()) + 3
        kept, coverage = prepare_records([row], max_seq_len=required)
        self.assertEqual(kept[0]["output_text"], example["output_text"])
        omitted, coverage = prepare_records([row], max_seq_len=required - 1)
        self.assertEqual(omitted, [])
        self.assertEqual(coverage["exclusion_reasons"], {"context_overflow": 1})
        self.assertEqual(coverage["excluded"][0]["required_tokens"], required)

    def test_groups_and_identical_inputs_never_leak_across_split(self):
        rows = [record("one", "same"), record("two", "same"),
                record("three", "different"), record("three", "third"), record("four", "fourth")]
        examples, _ = prepare_records(rows)
        train, heldout, _ = split_examples(examples, seed=3)
        self.assertFalse({e["group"] for e in train} & {e["group"] for e in heldout})
        self.assertFalse({e["input_text"] for e in train} & {e["input_text"] for e in heldout})
        self.assertTrue(train and heldout)

    def test_contradictory_targets_are_not_silently_averaged(self):
        rows = [record("one"), record("two", after={"hidden": True})]
        with self.assertRaisesRegex(ValueError, "contradictory"):
            prepare_records(rows)

    def test_natural_trace_inputs_include_full_pre_call_random_state(self):
        import random
        row = record()
        row["metadata"]["fixture"] = "runtime_conversation"
        row["metadata"]["rng_before"] = codec.encode(random.Random(7).getstate())
        prompt = codec.loads(record_to_example(row)["input_text"])
        self.assertEqual(prompt["environment"]["random_state"], random.Random(7).getstate())

    def test_gzip_records_roundtrip_without_losing_types(self):
        rows = [record(memory={"tuple": (1, True)})]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl.gz"
            write_jsonl(path, rows)
            self.assertEqual(read_jsonl(path), rows)

    def test_prompt_rejects_extra_labels_in_arguments(self):
        with self.assertRaises(ValueError):
            input_for_call("example", "respond", {"text": "hi", "memory": {}, "answer": "leak"})

    def test_compact_v2_record_is_shorter_and_roundtrips_without_loss(self):
        from moduleio import wire

        row = record(memory={"user": {"name": "Ada"}}, after={"user": {"name": "Ada"}})
        legacy = record_to_example(row, wire_format="legacy-v1")
        compact = record_to_example(row, wire_format="compact-v2")
        self.assertLess(len(compact["input_text"]), len(legacy["input_text"]))
        self.assertLess(len(compact["output_text"]), len(legacy["output_text"]))
        self.assertEqual(compact["wire_format"], "compact-v2")
        decoded = wire.decode_output_compact(
            compact["output_text"],
            "respond",
            codec.decode(row["input"]),
        )
        self.assertEqual(decoded, codec.decode(row["output"]))


if __name__ == "__main__":
    unittest.main()
