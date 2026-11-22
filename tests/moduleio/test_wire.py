"""Losslessness and strict fail-closed checks for compact-v2 wire format."""
import json
import math
import random
import struct
import unittest

from moduleio import codec, wire


class WireTests(unittest.TestCase):
    def test_types_values_and_dictionary_order_round_trip(self):
        value = {
            ("key", 2): [None, True, False, 2 ** 200, -(2 ** 200), "日本語\n\ud800"],
            "second": (b"\x00\xff", bytearray(b"abc"), {1, 3}, frozenset(("x", "y"))),
            7: {"inner": ()},
            "$reserved_key": [1, 2],
            "plain_dict": {"z": 1, "a": 2},
        }
        encoded = wire.dumps_compact(value)
        restored = wire.loads_compact(encoded)
        self.assertEqual(restored, value)
        self.assertEqual(list(restored), list(value))
        self.assertEqual(list(restored["plain_dict"]), ["z", "a"])
        self.assertIs(type(restored["second"]), tuple)
        self.assertIs(type(restored["second"][0]), bytes)
        self.assertIs(type(restored["second"][1]), bytearray)
        self.assertIs(type(restored["second"][2]), set)
        self.assertIs(type(restored["second"][3]), frozenset)
        self.assertIs(type(restored[("key", 2)][1]), bool)
        self.assertIs(type(restored[("key", 2)][3]), int)

    def test_ieee_float_bits_survive_including_nan_payload_and_negative_zero(self):
        patterns = ("0000000000000000", "8000000000000000", "3fb999999999999a",
                    "7ff0000000000000", "fff0000000000000", "7ff8000000000042",
                    "fff8000000000001")
        for pattern in patterns:
            with self.subTest(pattern=pattern):
                value = struct.unpack(">d", bytes.fromhex(pattern))[0]
                restored = wire.loads_compact(wire.dumps_compact(value))
                self.assertEqual(struct.pack(">d", restored).hex(), pattern)
        self.assertTrue(math.isnan(wire.loads_compact(wire.dumps_compact(float("nan")))))

    def test_aliases_preserved_across_containers_and_dictionary_keys(self):
        shared = ["kept", {"nested": 1}]
        key = ("same", 5)
        immutable = frozenset((1, 2))
        value = {key: shared, "alias": shared, "tuple": key,
                 "sets": [immutable, immutable]}
        restored = wire.loads_compact(wire.dumps_compact(value))
        restored_key = next(iter(restored))
        self.assertIs(restored[restored_key], restored["alias"])
        self.assertIs(restored_key, restored["tuple"])
        self.assertIs(restored["sets"][0], restored["sets"][1])
        restored["alias"].append("changed")
        self.assertEqual(restored[restored_key][-1], "changed")

    def test_cycles_and_custom_objects_are_explicitly_rejected(self):
        cycle = []
        cycle.append(cycle)
        with self.assertRaisesRegex(codec.UnsupportedValueError, "cyclic"):
            wire.encode_compact(cycle)

        class NoRepr:
            def __repr__(self):
                raise AssertionError("repr must not be called")

        class DictSubclass(dict):
            pass

        for value in (NoRepr(), DictSubclass(x=1), complex(1, 2)):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(codec.UnsupportedValueError):
                    wire.dumps_compact(value)

    def test_random_state_roundtrip_and_canonical_sets(self):
        state = random.Random(129).getstate()
        restored_rng = random.Random()
        restored_rng.setstate(wire.loads_compact(wire.dumps_compact({"state": state}))["state"])
        expected_rng = random.Random(129)
        self.assertEqual([restored_rng.random() for _ in range(8)],
                         [expected_rng.random() for _ in range(8)])
        left = {"x", 17, ("a", "b"), frozenset((9, 4))}
        right = set(reversed(list(left)))
        self.assertEqual(wire.dumps_compact(left), wire.dumps_compact(right))

    def test_malformed_or_ambiguous_encodings_fail_closed(self):
        bad = [
            {"$r": 999},
            {"$t": "l", "$i": 0, "v": [{"$r": 0}]},
            {"$t": "l", "$i": True, "v": []},
            {"$t": "t", "$i": 0, "v": [{"$t": "t", "$i": 0, "v": []}]},
            {"$t": "f", "b": "0000"},
            {"$t": "b", "v": "not base64"},
            {"$t": "unknown", "v": []},
            {"$t": "d", "v": [["x", "a"], ["x", "b"]]},
            {"$t": "s", "v": ["a", "a"]},
            {"$t": "f", "b": "0000000000000000", "extra": True},
            {"$evil": 1},
            1.5,
        ]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    wire.decode_compact(value)
        for text in ("NaN", "1.5", '{"$r":0,"$r":1}'):
            with self.assertRaises(ValueError):
                wire.loads_compact(text)

    def test_unchanged_sentinel_and_cross_field_alias_safety(self):
        mem = {"user": {"name": "Sam"}}
        args = {"text": "hello", "memory": mem}
        out = {"return": None, "memory": {"user": {"name": "Sam"}}, "exception": None}
        encoded = wire.encode_output_compact(out, "observe", args)
        self.assertEqual(json.loads(encoded)["memory"], "=")
        decoded = wire.decode_output_compact(encoded, "observe", args)
        self.assertEqual(decoded, out)
        self.assertIsNot(decoded["memory"], args["memory"])

        # Cross-field alias between return and candidates must suppress "=".
        candidate = {"module": "a", "reply": "hi"}
        choose_args = {"text": "hi", "memory": {}, "candidates": [candidate]}
        choose_out = {
            "return": candidate,
            "memory": {},
            "exception": None,
            "candidates": [candidate],
        }
        choose_encoded = wire.encode_output_compact(choose_out, "choose", choose_args)
        self.assertNotEqual(json.loads(choose_encoded)["candidates"], "=")
        choose_decoded = wire.decode_output_compact(choose_encoded, "choose", choose_args)
        self.assertIs(choose_decoded["return"], choose_decoded["candidates"][0])
        self.assertEqual(json.loads(choose_encoded)["memory"], "=")

        # Missing input_arguments must fail closed when "=" appears.
        with self.assertRaises(ValueError):
            wire.decode_output_compact(encoded, "observe", None)


if __name__ == "__main__":
    unittest.main()
