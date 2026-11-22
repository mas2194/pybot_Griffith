"""Failure-aware whole-call metrics and inference without an oracle fallback."""

import copy
import builtins
import random
import sys
import types
import unittest
from unittest import mock

from moduleio import codec
from moduleio.evaluation import (
    InvalidOutputError,
    always_none_baseline,
    compare_outputs,
    evaluate,
    exact_equal,
    validate_output,
)
from moduleio.predictor import ModuleIOPredictor, PredictionError, UnknownHookError


def example(memory_before=None, memory_after=None, answer=None, hook="respond", **extra):
    arguments = {"text": "an input, without a gold prefix", "memory": memory_before or {}}
    output = {"return": answer, "memory": memory_after if memory_after is not None else arguments["memory"], "exception": None}
    if hook == "choose":
        arguments["candidates"] = extra.pop("candidates", [])
        output["candidates"] = extra.pop("candidates_after", arguments["candidates"])
    if hook == "after":
        arguments["reply"] = "earlier selected answer"
    output.update(extra)
    return {
        "module": "test_module", "hook": hook, "group": "conversation-one",
        "input_text": "THE INPUT ONLY", "output_text": codec.dumps(output),
        "record": {"input": codec.encode(arguments), "output": codec.encode(output)},
    }


class FakeModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, input_text, **kwargs):
        self.calls.append((input_text, kwargs))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        if isinstance(output, dict) and "eos_generated" in output:
            return output
        return {"text": output, "eos_generated": True, "stop_reason": "eos", "generated_tokens": 10}


class CompleteOutputScoring(unittest.TestCase):
    def test_only_inputs_reach_generation(self):
        sample = example(answer="SECRET TEACHER ANSWER")
        model = FakeModel([sample["output_text"]])
        report = evaluate(model, [sample], max_new_tokens=500)
        self.assertEqual(model.calls, [(sample["input_text"], {"max_new_tokens": 500, "mode": "hard"})])
        self.assertEqual(report["rates"]["whole_call_exact"], 1.0)

    def test_correct_return_does_not_hide_wrong_memory(self):
        sample = example({"name": "Ada"}, {"name": "Emi"}, "Updated.")
        wrong = codec.loads(sample["output_text"])
        wrong["memory"] = {"name": "Ada"}
        report = evaluate(FakeModel([codec.dumps(wrong)]), [sample])
        self.assertEqual(report["rates"]["return_exact"], 1.0)
        self.assertEqual(report["rates"]["memory_exact"], 0.0)
        self.assertEqual(report["rates"]["whole_call_exact"], 0.0)

    def test_none_with_state_update_is_active_and_baseline_fails(self):
        sample = example({"a": 1}, {"a": 2}, None, hook="observe")
        report = always_none_baseline([sample])
        self.assertEqual(report["by_activity"]["active"]["counts"]["examples"], 1)
        self.assertEqual(report["rates"]["return_exact"], 1.0)
        self.assertEqual(report["rates"]["whole_call_exact"], 0.0)

    def test_tuple_and_list_are_different(self):
        sample = example({}, {"held": (1, 2)})
        wrong = codec.loads(sample["output_text"])
        wrong["memory"]["held"] = [1, 2]
        report = evaluate(FakeModel([codec.dumps(wrong)]), [sample])
        self.assertFalse(report["predictions"][0]["whole_call_exact"])
        self.assertFalse(exact_equal(True, 1))

    def test_choose_candidate_mutation_counts(self):
        sample = example(
            answer=0, hook="choose", candidates=[{"reply": "a"}], candidates_after=[{"reply": "b"}],
        )
        wrong = codec.loads(sample["output_text"])
        wrong["candidates"][0]["reply"] = "a"
        report = evaluate(FakeModel([codec.dumps(wrong)]), [sample])
        self.assertEqual(report["rates"]["return_exact"], 1.0)
        self.assertEqual(report["rates"]["candidates_exact"], 0.0)
        self.assertEqual(report["rates"]["whole_call_exact"], 0.0)

    def test_envelope_field_order_is_immaterial(self):
        expected = {"return": None, "memory": {"a": 1, "b": 2}, "exception": None}
        actual = {"exception": None, "memory": {"a": 1, "b": 2}, "return": None}
        self.assertTrue(compare_outputs(expected, actual, "observe")["whole_call_exact"])
        actual["memory"] = {"b": 2, "a": 1}
        self.assertFalse(compare_outputs(expected, actual, "observe")["whole_call_exact"])

    def test_whole_call_checks_return_alias_into_candidates(self):
        candidate = {"module": "a", "reply": "hello"}
        expected = {"return": candidate, "memory": {}, "exception": None, "candidates": [candidate]}
        actual = {"return": dict(candidate), "memory": {}, "exception": None,
                  "candidates": [dict(candidate)]}
        scores = compare_outputs(expected, actual, "choose")
        self.assertTrue(scores["return_exact"])
        self.assertTrue(scores["candidates_exact"])
        self.assertFalse(scores["whole_call_exact"])

    def test_missing_unknown_fields_and_invalid_exception_rejected(self):
        for output in (
            {"return": None, "memory": {}},
            {"return": None, "memory": {}, "exception": None, "extra": True},
            {"return": None, "memory": "not a memory dictionary", "exception": None},
            {"return": None, "memory": {}, "exception": {"type": "ValueError", "args": []}},
        ):
            with self.subTest(output=output), self.assertRaises(InvalidOutputError):
                validate_output(output, "respond")

    def test_invalid_and_incomplete_attempts_stay_in_denominator(self):
        class ExampleTooLongError(ValueError):
            pass

        sample = example(answer="yes")
        outputs = [
            sample["output_text"],
            "{",
            codec.dumps({"return": "yes"}),
            {"text": sample["output_text"], "eos_generated": False, "stop_reason": "max_new_tokens"},
            ExampleTooLongError("Too long"),
            RuntimeError("model failed"),
        ]
        report = evaluate(FakeModel(outputs), [sample] * len(outputs))
        self.assertEqual(report["counts"]["examples"], 6)
        self.assertEqual(report["counts"]["whole_call_exact"], 1)
        self.assertEqual(report["rates"]["whole_call_exact"], 1 / 6)
        for status in ("invalid_json", "invalid_schema", "eos_missing", "overflow", "generation_error"):
            self.assertEqual(report["counts"][status], 1)

    def test_invalid_utf8_is_not_accepted_after_replacement_decoding(self):
        sample = example(answer="yes")
        model = FakeModel([{"text": sample["output_text"], "eos_generated": True,
                            "utf8_valid": False, "stop_reason": "eos"}])
        report = evaluate(model, [sample])
        self.assertEqual(report["counts"]["invalid_utf8"], 1)
        self.assertEqual(report["counts"]["whole_call_exact"], 0)

    def test_activity_breakdown_prevents_inactive_success_hiding_failure(self):
        inactive = example()
        active = example(answer="answer")
        report = always_none_baseline([inactive] * 9 + [active])
        self.assertEqual(report["rates"]["whole_call_exact"], 0.9)
        self.assertEqual(report["by_activity"]["active"]["rates"]["whole_call_exact"], 0.0)

    def test_empty_dataset_has_no_fabricated_success_rate(self):
        report = evaluate(FakeModel([]), [])
        self.assertEqual(report["counts"]["examples"], 0)
        self.assertIsNone(report["rates"]["whole_call_exact"])


class IndependentPrediction(unittest.TestCase):
    def _prompt_stub(self):
        module = types.ModuleType("moduleio.data")
        module.input_for_call = mock.Mock(return_value="INPUT WITHOUT TEACHER")
        return mock.patch.dict(sys.modules, {"moduleio.data": module})

    def test_unknown_hooks_are_refused_without_generation(self):
        model = FakeModel([])
        predictor = ModuleIOPredictor(model, {"trained_hooks": ["one.respond"]})
        with self.assertRaises(UnknownHookError):
            predictor.predict_call("two", "respond", {"text": "hi", "memory": {}})
        self.assertEqual(model.calls, [])

    def test_metadata_is_mandatory(self):
        with self.assertRaises(PredictionError):
            ModuleIOPredictor(FakeModel([]), {})

    def test_bad_output_does_not_mutate_inputs_or_call_teacher(self):
        arguments = {"text": "hi", "memory": {"turns": [{"user": "previous"}]}}
        before = copy.deepcopy(arguments)
        predictor = ModuleIOPredictor(FakeModel(["{}"]), {"trained_hooks": ["one.respond"]})
        with self._prompt_stub(), self.assertRaises(PredictionError):
            predictor.predict_call("one", "respond", arguments)
        self.assertEqual(arguments, before)

    def test_complete_valid_output_is_returned_without_applying_it(self):
        arguments = {"text": "hi", "memory": {}}
        output = {"return": "Hello", "memory": {"seen": True}, "exception": None}
        predictor = ModuleIOPredictor(FakeModel([codec.dumps(output)]), {"trained_hooks": ["one.respond"]})
        with self._prompt_stub():
            result = predictor.predict_call("one", "respond", arguments)
        self.assertEqual(result, output)
        self.assertEqual(arguments["memory"], {})

    def test_incomplete_exact_text_is_refused(self):
        sample = example(answer="yes")
        model = FakeModel([{"text": sample["output_text"], "eos_generated": False, "stop_reason": "max_new_tokens"}])
        predictor = ModuleIOPredictor(model, {"trained_hooks": ["one.respond"]})
        with self._prompt_stub(), self.assertRaises(PredictionError):
            predictor.predict_call("one", "respond", {"text": "hi", "memory": {}})

    def test_prediction_with_real_prompt_builder_never_imports_bot(self):
        output = {"return": "hello", "memory": {}, "exception": None}
        model = FakeModel([codec.dumps(output)])
        predictor = ModuleIOPredictor(model, {"trained_hooks": ["one.respond"]})
        original_import = builtins.__import__

        def reject_teacher(name, *args, **kwargs):
            if name == "bot" or name.startswith("bot."):
                raise AssertionError("Learned prediction imported the teacher")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_teacher):
            prediction = predictor.predict_call("one", "respond", {"text": "hi", "memory": {}},
                                                 seed=3, environment_date="2026-11-21")
        self.assertEqual(prediction, output)
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn("hello", model.calls[0][0])

    def test_runtime_rng_input_matches_the_recorded_training_prompt(self):
        from moduleio.data import record_to_example

        arguments = {"text": "hello", "memory": {"held": (1, 2)}}
        output = {"return": "a reply", "memory": arguments["memory"], "exception": None}
        rng_state = random.Random(17).getstate()
        record = {
            "schema_version": 1, "conversation_id": "runtime-one",
            "module": "one", "hook": "respond",
            "input": codec.encode(arguments), "output": codec.encode(output),
            "metadata": {"fixture": "runtime_conversation", "seed": 17,
                         "environment_date": "2026-11-21", "rng_before": codec.encode(rng_state)},
        }
        training_example = record_to_example(record)
        model = FakeModel([training_example["output_text"]])
        predictor = ModuleIOPredictor(model, {"trained_hooks": ["one.respond"],
                                            "training_fixtures": ["runtime_conversation"]})
        before_rng = random.getstate()
        actual = predictor.predict_call("one", "respond", arguments, rng_state=rng_state,
                                        environment_date="2026-11-21")
        self.assertEqual(model.calls[0][0], training_example["input_text"])
        self.assertEqual(actual, output)
        self.assertEqual(random.getstate(), before_rng)

    def test_runtime_checkpoint_refuses_missing_rng_before_generation(self):
        model = FakeModel([])
        predictor = ModuleIOPredictor(model, {"trained_hooks": ["one.respond"],
                                            "training_fixtures": ["runtime_conversation"]})
        with self.assertRaisesRegex(PredictionError, "requires rng_state"):
            predictor.predict_call("one", "respond", {"text": "hi", "memory": {}}, seed=17)
        self.assertEqual(model.calls, [])

    def test_compact_v2_prediction_and_evaluation_expand_sentinel_without_mutating_input(self):
        from moduleio import wire
        from moduleio.data import record_to_example

        arguments = {"text": "hello", "memory": {"user": {"name": "Ada"}}}
        output = {"return": "Hi!", "memory": {"user": {"name": "Ada"}}, "exception": None}
        record = {
            "schema_version": 1, "conversation_id": "compact-test",
            "module": "one", "hook": "respond",
            "input": codec.encode(arguments), "output": codec.encode(output),
            "metadata": {"fixture": "isolated_seeded_call", "seed": 0,
                         "environment_date": "2026-11-21"},
        }
        compact_ex = record_to_example(record, wire_format="compact-v2")
        eval_report = evaluate(FakeModel([compact_ex["output_text"]]), [compact_ex])
        self.assertEqual(eval_report["rates"]["whole_call_exact"], 1.0)
        baseline = always_none_baseline([compact_ex])
        self.assertEqual(baseline["rates"]["whole_call_exact"], 0.0)
        self.assertEqual(baseline["rates"]["memory_exact"], 1.0)

        before = copy.deepcopy(arguments)
        predictor = ModuleIOPredictor(
            FakeModel([wire.encode_output_compact(output, "respond", arguments)]),
            {"trained_hooks": ["one.respond"], "wire_format": "compact-v2"},
        )
        predicted = predictor.predict_call("one", "respond", arguments, seed=0, environment_date="2026-11-21")
        self.assertEqual(predicted, output)
        self.assertEqual(arguments, before)


if __name__ == "__main__":
    unittest.main()
