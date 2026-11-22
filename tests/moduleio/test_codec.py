"""Losslessness checks for actual Python values, not just ordinary JSON data."""
import json
import math
import random
import struct
import unittest

from moduleio import codec


class CodecTests(unittest.TestCase):
    def test_types_values_and_dictionary_order_round_trip(self):
        value = {
            ("key", 2): [None, True, False, 2 ** 200, -(2 ** 200), "日本語\n\ud800"],
            "second": (b"\x00\xff", bytearray(b"abc"), {1, 3}, frozenset(("x", "y"))),
            7: {"inner": ()},
        }
        restored = codec.loads(codec.dumps(value))
        self.assertEqual(restored, value)
        self.assertEqual(list(restored), list(value))
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
                restored = codec.loads(codec.dumps(value))
                self.assertEqual(struct.pack(">d", restored).hex(), pattern)
        self.assertTrue(math.isnan(codec.loads(codec.dumps(float("nan")))))

    def test_aliases_preserved_across_containers_and_dictionary_keys(self):
        shared = ["kept", {"nested": 1}]
        key = ("same", 5)
        immutable = frozenset((1, 2))
        value = {key: shared, "alias": shared, "tuple": key,
                 "sets": [immutable, immutable]}
        restored = codec.loads(codec.dumps(value))
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
            codec.encode(cycle)

        class NoRepr:
            def __repr__(self):
                raise AssertionError("repr must not be called")

        class DictSubclass(dict):
            pass

        for value in (NoRepr(), DictSubclass(x=1), complex(1, 2)):
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises(codec.UnsupportedValueError):
                    codec.dumps(value)

    def test_random_state_and_long_values_are_never_truncated(self):
        state = random.Random(129).getstate()
        value = {"state": state, "long": "x" * 100000, "large": 10 ** 1000}
        encoded = codec.dumps(value)
        self.assertGreater(len(encoded), 100000)
        self.assertEqual(codec.loads(encoded), value)
        restored_rng = random.Random()
        restored_rng.setstate(codec.loads(encoded)["state"])
        expected_rng = random.Random(129)
        self.assertEqual([restored_rng.random() for _ in range(8)],
                         [expected_rng.random() for _ in range(8)])

    def test_set_encoding_is_canonical(self):
        left = {"x", 17, ("a", "b"), frozenset((9, 4))}
        right = set(reversed(list(left)))
        self.assertEqual(codec.dumps(left), codec.dumps(right))

    def test_malformed_or_ambiguous_encodings_fail_closed(self):
        bad = [
            {"$ref": 999},
            {"$type": "list", "$id": 0, "items": [{"$ref": 0}]},
            {"$type": "list", "$id": True, "items": []},
            {"$type": "tuple", "$id": 0, "items": [
                {"$type": "tuple", "$id": 0, "items": []}]},
            {"$type": "int", "value": "01"},
            {"$type": "int", "value": 1},
            {"$type": "float", "bits": "0000"},
            {"$type": "bytes", "base64": "not base64"},
            {"$type": "unknown"},
            {"$type": "dict", "$id": 0, "items": [["x", "a"], ["x", "b"]]},
            {"$type": "set", "$id": 0, "items": ["a", "a"]},
            {"$type": "int", "value": "1", "extra": True},
            2,
        ]
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    codec.decode(value)
        for text in ('NaN', '{"$ref":0,"$ref":1}', '[1,2]'):
            with self.assertRaises(ValueError):
                codec.loads(text)

    def test_encoded_value_is_strict_json(self):
        value = codec.encode([2 ** 100, float("inf"), float("nan"), b"\xff"])
        serialized = json.dumps(value, allow_nan=False)
        self.assertIsInstance(serialized, str)


if __name__ == "__main__":
    unittest.main()
