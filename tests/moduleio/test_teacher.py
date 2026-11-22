"""Check tracing against the repository's unmodified runtime control flow."""
import ast
import copy
import random
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from moduleio import codec, teacher


def fixture_runtime():
    """Use the real runtime function bodies with a small deterministic module set.

    Importing bot.runtime would load the entire large rule/knowledge tree.  Here
    only its unchanged dispatch functions are compiled with synthetic hooks;
    candidate isolation, exception handling, and arbitration are the real code.
    """
    path = Path(__file__).resolve().parents[2] / "bot" / "runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_copy", "_choose", "turn")]
    runtime = types.ModuleType("small_runtime")
    runtime.copy, runtime.FALLBACK = copy, "Hello!"
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
         runtime.__dict__)
    runtime.events, runtime.turn_ids = [], []

    def observe(text, memory):
        runtime.events.append("observe")
        memory.setdefault("turns", [])
        memory.setdefault("seen", []).append(text)
        memory["nested"] = {"count": 0, "tuple": (1, 2)}

    def loser(text, memory):
        runtime.events.append("loser")
        runtime.turn_ids.append(id(memory["turns"]))
        memory["nested"]["count"] = 999
        return "loser:" + str(random.randrange(100000))

    def passing(text, memory):
        runtime.events.append("passing")
        runtime.turn_ids.append(id(memory["turns"]))
        memory["not_selected"] = True
        memory["nested"]["count"] = 888
        return None

    def broken(text, memory):
        runtime.events.append("broken")
        memory["exception_side_effect"] = (text,)
        random.random()
        raise ValueError("teacher failure", (1, 2))

    def winner(text, memory):
        runtime.events.append("winner")
        runtime.turn_ids.append(id(memory["turns"]))
        # If a preceding candidate's state leaked, this assertion would fail.
        assert memory["nested"]["count"] == 0
        memory["nested"]["count"] = 1
        return "winner:" + str(random.randrange(100000))

    def choose(text, memory, candidates):
        runtime.events.append("choose")
        memory["arbiter_mutation"] = True
        candidates[0]["visited"] = True
        return candidates[-1]

    def after(text, reply, memory):
        runtime.events.append("after")
        memory["last"] = (text, reply)

    runtime._OBSERVERS = [("observer", 100, observe)]
    runtime._HANDLERS = [("loser", 90, loser), ("passing", 80, passing),
                         ("broken", 70, broken), ("winner", 60, winner)]
    runtime._AFTER = [("after", 0, after)]
    runtime._ARBITER = choose
    runtime._MODULES = [("arbiter", 0, types.SimpleNamespace(choose=choose))]
    return runtime


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.getstate()

    def tearDown(self):
        random.setstate(self.rng)

    def test_whole_trace_matches_real_runtime_and_restores_tables_and_rng(self):
        texts, seed = ["first", "second"], 164
        reference = fixture_runtime()
        random.seed(seed)
        memory = {}
        expected = [reference.turn(text, memory) for text in texts]
        expected_rng = random.getstate()
        runtime = fixture_runtime()
        originals = {name: getattr(runtime, name)
                     for name in ("_OBSERVERS", "_HANDLERS", "_AFTER", "_ARBITER")}
        caller_rng = random.getstate()
        with patch.object(teacher, "_load_runtime", return_value=runtime):
            records, summary = teacher.capture_conversation(
                texts, conversation_id="two-turns", seed=seed)
        self.assertEqual([codec.decode(r) for r in summary["turn_results"]], expected)
        self.assertEqual(codec.decode(summary["final_memory"]), memory)
        self.assertEqual(random.getstate(), caller_rng)
        self.assertEqual(codec.decode(records[-1]["metadata"]["rng_after"]), expected_rng)
        self.assertEqual(runtime.events, reference.events)
        self.assertEqual(len(records), 14)
        for name, original in originals.items():
            self.assertIs(getattr(runtime, name), original)
        # Actual runtime shares turns between candidates, but no other memory.
        self.assertEqual(len(set(runtime.turn_ids)), 1)
        self.assertNotIn("not_selected", memory)
        self.assertNotIn("exception_side_effect", memory)
        self.assertNotIn("arbiter_mutation", memory)
        self.assertEqual(memory["nested"]["count"], 1)

    def test_none_mutations_exception_mutations_and_unselected_candidates_survive(self):
        runtime = fixture_runtime()
        with patch.object(teacher, "_load_runtime", return_value=runtime):
            records, _ = teacher.capture_conversation(["line"], conversation_id="details", seed=8)
        by_hook = {record["module"]: record for record in records}
        passing = codec.decode(by_hook["passing"]["output"])
        self.assertIsNone(passing["return"])
        self.assertTrue(passing["memory"]["not_selected"])
        self.assertEqual(passing["memory"]["nested"]["count"], 888)
        broken = codec.decode(by_hook["broken"]["output"])
        self.assertEqual(broken["exception"],
                         {"type": "builtins.ValueError", "args": ("teacher failure", (1, 2))})
        self.assertEqual(broken["memory"]["exception_side_effect"], ("line",))
        self.assertEqual(codec.decode(by_hook["loser"]["output"])["memory"]["nested"]["count"], 999)
        self.assertEqual(codec.decode(by_hook["winner"]["input"])["memory"]["nested"]["count"], 0)
        arbiter = codec.decode(by_hook["arbiter"]["output"])
        self.assertIs(arbiter["return"], arbiter["candidates"][-1])
        self.assertTrue(arbiter["candidates"][0]["visited"])
        self.assertTrue(arbiter["memory"]["arbiter_mutation"])
        self.assertNotEqual(by_hook["broken"]["metadata"]["rng_before"],
                            by_hook["broken"]["metadata"]["rng_after"])

    def test_filtering_changes_only_records_and_discovery_is_runtime_based(self):
        runtime = fixture_runtime()
        with patch.object(teacher, "_load_runtime", return_value=runtime):
            inventory = teacher.discover_hooks()
            selected, selected_summary = teacher.capture_conversation(
                ["text"], conversation_id="selected", seed=19,
                selected_hooks=[("passing", "respond")])
        self.assertEqual([row["id"] for row in inventory],
                         ["observer.observe", "loser.respond", "passing.respond", "broken.respond",
                          "winner.respond", "arbiter.choose", "after.after"])
        full = fixture_runtime()
        with patch.object(teacher, "_load_runtime", return_value=full):
            _, full_summary = teacher.capture_conversation(["text"], conversation_id="full", seed=19)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["module"], "passing")
        self.assertEqual(selected_summary["turn_results"], full_summary["turn_results"])
        self.assertEqual(selected_summary["final_memory"], full_summary["final_memory"])

    def test_snapshot_errors_do_not_make_runtime_skip_successful_hooks(self):
        runtime = fixture_runtime()
        original = runtime._OBSERVERS
        called = []

        def unsupported(text, memory):
            called.append("observer")
            original[0][2](text, memory)
            memory["opaque"] = object()

        runtime._OBSERVERS = [("unsupported", 100, unsupported)]
        table = runtime._OBSERVERS
        before = random.getstate()
        with patch.object(teacher, "_load_runtime", return_value=runtime):
            with self.assertRaises(teacher.TeacherCaptureError) as caught:
                teacher.capture_conversation(["text"], conversation_id="unsupported")
        self.assertIn("winner", runtime.events)
        self.assertIn("choose", runtime.events)
        self.assertIn("after", runtime.events)
        self.assertTrue(caught.exception.errors)
        self.assertIs(runtime._OBSERVERS, table)
        self.assertEqual(random.getstate(), before)

    def test_tables_and_rng_restored_if_runtime_itself_raises(self):
        runtime = fixture_runtime()
        original = runtime._HANDLERS

        def fail(text, memory):
            random.random()
            raise RuntimeError("outside hook")

        runtime.turn = fail
        before = random.getstate()
        with patch.object(teacher, "_load_runtime", return_value=runtime):
            with self.assertRaisesRegex(RuntimeError, "outside hook"):
                teacher.capture_conversation(["text"], conversation_id="runtime-failure", seed=3)
        self.assertIs(runtime._HANDLERS, original)
        self.assertEqual(random.getstate(), before)

    def test_isolated_call_preserves_input_and_records_full_output_and_rng(self):
        def respond(text, memory):
            memory["values"].append(random.randrange(100000))
            return memory["values"]

        module = types.SimpleNamespace(respond=respond, PRIORITY=7)
        memory = {"values": [1]}
        arguments = {"text": "hello", "memory": memory}
        before = random.getstate()
        with patch.object(teacher, "_load_module", return_value=module):
            record = teacher.capture_call("example", "respond", arguments,
                                          conversation_id="isolated", seed=18)
        self.assertEqual(arguments, {"text": "hello", "memory": {"values": [1]}})
        self.assertEqual(codec.decode(record["input"]), arguments)
        output = codec.decode(record["output"])
        self.assertIs(output["return"], output["memory"]["values"])
        self.assertEqual(len(output["return"]), 2)
        self.assertEqual(record["metadata"]["fixture"], "isolated_seeded_call")
        self.assertEqual(record["metadata"]["seed"], 18)
        self.assertEqual(random.getstate(), before)
        expected = random.Random(18)
        self.assertEqual(output["return"][-1], expected.randrange(100000))

    def test_isolated_exception_is_data_and_unknown_hook_is_not(self):
        def observe(text, memory):
            memory["changed"] = True
            raise KeyError("missing")

        module = types.SimpleNamespace(observe=observe)
        with patch.object(teacher, "_load_module", return_value=module):
            record = teacher.capture_call("example", "observe", {"text": "x", "memory": {}},
                                          conversation_id="exception")
            with self.assertRaises(ValueError):
                teacher.capture_call("example", "respond", {"text": "x", "memory": {}},
                                     conversation_id="no-respond")
        output = codec.decode(record["output"])
        self.assertTrue(output["memory"]["changed"])
        self.assertEqual(output["exception"], {"type": "builtins.KeyError", "args": ("missing",)})
        with self.assertRaises(ValueError):
            teacher.capture_call("example", "invalid", {}, conversation_id="invalid")
        with self.assertRaises(ValueError):
            teacher.capture_call("example", "respond", {"text": "x"}, conversation_id="invalid")


if __name__ == "__main__":
    unittest.main()
