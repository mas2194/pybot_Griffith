"""Lossless JSON for the built-in values crossing a PyBot hook boundary.

Dict insertion order, tuple/list distinctions, integer precision, IEEE-754 float
bits, bytes, and container aliases are preserved.  Cyclic containers and custom
Python objects are rejected explicitly; this is not a pickle replacement.  The
format never executes code or silently substitutes ``repr(value)``.
"""
from __future__ import annotations

import base64
import json
import struct


class UnsupportedValueError(TypeError):
    """A value cannot be represented without losing its supported semantics."""


_CONTAINERS = (list, tuple, dict, set, frozenset, bytearray)


def encode(value):
    """Return JSON-safe tagged data, preserving shared container references.

    Exact built-in types are accepted.  Subclasses may have extra state, so they
    are deliberately not treated as their built-in base class.  Set order is
    canonicalized because sets have no insertion-order contract.
    """
    seen, active = {}, set()

    def visit(item, path):
        kind = type(item)
        if kind in (type(None), bool, str):
            return item
        if kind is int:
            return {"$type": "int", "value": str(item)}
        if kind is float:
            return {"$type": "float", "bits": struct.pack(">d", item).hex()}
        if kind is bytes:
            return {"$type": "bytes", "base64": base64.b64encode(item).decode("ascii")}
        if kind not in _CONTAINERS:
            raise UnsupportedValueError(
                "unsupported type %s.%s at %s" % (kind.__module__, kind.__qualname__, path))
        identity = id(item)
        if identity in active:
            raise UnsupportedValueError("cyclic container at %s" % path)
        if identity in seen:
            return {"$ref": seen[identity]}
        reference = len(seen)
        seen[identity] = reference
        active.add(identity)
        try:
            result = {"$type": kind.__name__, "$id": reference}
            if kind is bytearray:
                result["base64"] = base64.b64encode(item).decode("ascii")
            elif kind is dict:
                result["items"] = [
                    [visit(key, "%s.key[%d]" % (path, index)),
                     visit(part, "%s.value[%d]" % (path, index))]
                    for index, (key, part) in enumerate(item.items())
                ]
            else:
                ordered = item
                if kind in (set, frozenset):
                    # Set members must be hashable, so they cannot hide a mutable
                    # cycle back into this set.  Independent encodings make this
                    # order independent of hash randomization and iteration order.
                    ordered = sorted(item, key=lambda part: json.dumps(
                        encode(part), ensure_ascii=True, separators=(",", ":"),
                        allow_nan=False))
                result["items"] = [visit(part, "%s[%d]" % (path, index))
                                   for index, part in enumerate(ordered)]
            return result
        finally:
            active.remove(identity)

    return visit(value, "$")


def decode(value):
    """Decode tagged data without imports, executable objects, or type guessing.

    Malformed encodings, duplicate IDs, forward references, and cycles raise
    ``ValueError``.  Container aliases from ``encode`` are reconstructed.
    """
    references, active = {}, set()

    def fields(item, expected):
        if set(item) != set(expected):
            raise ValueError("invalid fields for tagged value")

    def identifier(item):
        reference = item.get("$id")
        if type(reference) is not int or reference < 0 or reference in references:
            raise ValueError("invalid or duplicate container ID")
        return reference

    def visit(item):
        if type(item) in (type(None), bool, str):
            return item
        if type(item) is not dict:
            raise ValueError("expected a primitive or tagged dictionary")
        if "$ref" in item:
            fields(item, ("$ref",))
            reference = item["$ref"]
            if type(reference) is not int or reference not in references or reference in active:
                raise ValueError("unknown, forward, or cyclic reference")
            return references[reference]
        kind = item.get("$type")
        if kind == "int":
            fields(item, ("$type", "value"))
            text = item["value"]
            if type(text) is not str:
                raise ValueError("integer value must be a decimal string")
            try:
                result = int(text)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid integer encoding") from error
            if str(result) != text:
                raise ValueError("noncanonical integer encoding")
            return result
        if kind == "float":
            fields(item, ("$type", "bits"))
            try:
                bits = bytes.fromhex(item["bits"])
                if len(bits) != 8 or bits.hex() != item["bits"]:
                    raise ValueError("invalid IEEE-754 bits")
                return struct.unpack(">d", bits)[0]
            except (TypeError, ValueError, struct.error) as error:
                raise ValueError("invalid float encoding") from error
        if kind == "bytes":
            fields(item, ("$type", "base64"))
            return read_bytes(item["base64"])
        if kind not in ("list", "tuple", "dict", "set", "frozenset", "bytearray"):
            raise ValueError("unknown tagged type")
        fields(item, ("$type", "$id", "base64" if kind == "bytearray" else "items"))
        reference = identifier(item)
        # The placeholder detects active references before an immutable container
        # can be constructed.  Cycles are rejected by the encoder as well.
        references[reference] = None
        active.add(reference)
        try:
            if kind == "bytearray":
                result = bytearray(read_bytes(item["base64"]))
            else:
                parts = item["items"]
                if type(parts) is not list:
                    raise ValueError("container items must be a list")
                if kind == "dict":
                    result = {}
                    for pair in parts:
                        if type(pair) is not list or len(pair) != 2:
                            raise ValueError("dictionary item must be a key/value pair")
                        key, part = visit(pair[0]), visit(pair[1])
                        try:
                            if key in result:
                                raise ValueError("duplicate dictionary key")
                            result[key] = part
                        except TypeError as error:
                            raise ValueError("unhashable dictionary key") from error
                else:
                    parts = [visit(part) for part in parts]
                    try:
                        result = {"list": list, "tuple": tuple, "set": set,
                                  "frozenset": frozenset}[kind](parts)
                    except TypeError as error:
                        raise ValueError("unhashable set member") from error
                    if kind in ("set", "frozenset") and len(result) != len(parts):
                        raise ValueError("duplicate set member")
            references[reference] = result
            return result
        finally:
            active.remove(reference)

    def read_bytes(text):
        if type(text) is not str:
            raise ValueError("bytes must be base64 text")
        try:
            decoded = base64.b64decode(text.encode("ascii"), validate=True)
        except (ValueError, UnicodeError) as error:
            raise ValueError("invalid base64 encoding") from error
        if base64.b64encode(decoded).decode("ascii") != text:
            raise ValueError("noncanonical base64 encoding")
        return decoded

    return visit(value)


def dumps(value):
    """Encode a Python value as compact, ASCII-safe JSON (including lone surrogates)."""
    return json.dumps(encode(value), ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def loads(text):
    """Decode the JSON representation emitted by :func:`dumps`."""
    def reject_constant(value):
        raise ValueError("untagged non-finite JSON constant: " + value)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object field")
            result[key] = value
        return result

    return decode(json.loads(text, parse_constant=reject_constant,
                             object_pairs_hook=unique_object))
