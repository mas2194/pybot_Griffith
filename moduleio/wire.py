"""Compact lossless JSON wire format (compact-v2) for PyBot hook I/O.

Preserves every Python built-in type, dict order, IEEE-754 float bits, and
container alias supported by :mod:`moduleio.codec`, while omitting redundant
``$type``/``$id`` wrappers on plain strings, dicts, lists, and exact integers,
and allowing unchanged ``memory`` / ``candidates`` output fields to be encoded
as the sentinel ``"="`` when they share no container aliases with other output
fields.
"""
from __future__ import annotations

import base64
import copy
import json
import struct

from . import codec

WIRE_FORMATS = ("legacy-v1", "compact-v2")
_CONTAINERS = (list, tuple, dict, set, frozenset, bytearray)


def _container_ids(value) -> set[int]:
    """Collect object IDs of all built-in containers reachable from ``value``."""
    ids: set[int] = set()
    active: set[int] = set()

    def visit(item) -> None:
        kind = type(item)
        if kind not in _CONTAINERS:
            return
        identity = id(item)
        if identity in active or identity in ids:
            return
        ids.add(identity)
        active.add(identity)
        try:
            if kind is bytearray:
                return
            if kind is dict:
                for key, part in item.items():
                    visit(key)
                    visit(part)
            else:
                for part in item:
                    visit(part)
        finally:
            active.remove(identity)

    visit(value)
    return ids


def _count_containers(value) -> dict[int, int]:
    """Traverse ``value`` deterministically, counting container references."""
    counts: dict[int, int] = {}
    active: set[int] = set()

    def visit(item, path: str) -> None:
        kind = type(item)
        if kind in (type(None), bool, str, int, float, bytes):
            return
        if kind not in _CONTAINERS:
            raise codec.UnsupportedValueError(
                "unsupported type %s.%s at %s" % (kind.__module__, kind.__qualname__, path)
            )
        identity = id(item)
        if identity in active:
            raise codec.UnsupportedValueError("cyclic container at %s" % path)
        counts[identity] = counts.get(identity, 0) + 1
        if counts[identity] > 1:
            return
        active.add(identity)
        try:
            if kind is bytearray:
                return
            if kind is dict:
                for index, (key, part) in enumerate(item.items()):
                    visit(key, "%s.key[%d]" % (path, index))
                    visit(part, "%s.value[%d]" % (path, index))
            else:
                ordered = item
                if kind in (set, frozenset):
                    ordered = sorted(item, key=codec.dumps)
                for index, part in enumerate(ordered):
                    visit(part, "%s[%d]" % (path, index))
        finally:
            active.remove(identity)

    visit(value, "$")
    return counts


def encode_compact(value):
    """Return compact JSON-safe data preserving exact types, order, and aliases."""
    counts = _count_containers(value)
    seen: dict[int, int] = {}

    def visit(item):
        kind = type(item)
        if kind in (type(None), bool, str, int):
            return item
        if kind is float:
            return {"$t": "f", "b": struct.pack(">d", item).hex()}
        if kind is bytes:
            return {"$t": "b", "v": base64.b64encode(item).decode("ascii")}
        identity = id(item)
        if identity in seen:
            return {"$r": seen[identity]}
        aliased = counts[identity] > 1
        reference = None
        if aliased:
            reference = len(seen)
            seen[identity] = reference
        if kind is bytearray:
            result = {"$t": "ba", "v": base64.b64encode(item).decode("ascii")}
            if aliased:
                result["$i"] = reference
            return result
        if kind is dict:
            if not aliased and all(type(k) is str and not k.startswith("$") for k in item):
                return {key: visit(part) for key, part in item.items()}
            result = {
                "$t": "d",
                "v": [[visit(key), visit(part)] for key, part in item.items()],
            }
            if aliased:
                result["$i"] = reference
            return result
        ordered = item
        if kind in (set, frozenset):
            ordered = sorted(item, key=codec.dumps)
        items = [visit(part) for part in ordered]
        if kind is list and not aliased:
            return items
        tag = {list: "l", tuple: "t", set: "s", frozenset: "fs"}[kind]
        result = {"$t": tag, "v": items}
        if aliased:
            result["$i"] = reference
        return result

    return visit(value)


def decode_compact(value):
    """Decode compact-v2 data with strict fail-closed type and reference checks."""
    references: dict[int, object] = {}
    active: set[int] = set()

    def fields(item: dict, expected: tuple[str, ...]) -> None:
        if set(item) != set(expected):
            raise ValueError("invalid fields for compact tagged value")

    def identifier(item: dict) -> int:
        reference = item.get("$i")
        if type(reference) is not int or reference < 0 or reference in references:
            raise ValueError("invalid or duplicate container ID")
        return reference

    def read_bytes(text) -> bytes:
        if type(text) is not str:
            raise ValueError("bytes must be base64 text")
        try:
            decoded = base64.b64decode(text.encode("ascii"), validate=True)
        except (ValueError, UnicodeError) as error:
            raise ValueError("invalid base64 encoding") from error
        if base64.b64encode(decoded).decode("ascii") != text:
            raise ValueError("noncanonical base64 encoding")
        return decoded

    def visit(item):
        kind = type(item)
        if kind in (type(None), bool, str, int):
            return item
        if kind is list:
            return [visit(part) for part in item]
        if kind is not dict:
            raise ValueError("bare floats and non-JSON values are not allowed in compact wire format")
        if "$r" in item:
            fields(item, ("$r",))
            reference = item["$r"]
            if type(reference) is not int or reference not in references or reference in active:
                raise ValueError("unknown, forward, or cyclic reference")
            return references[reference]
        if "$t" in item:
            tag = item["$t"]
            if tag == "f":
                fields(item, ("$t", "b"))
                hex_bits = item["b"]
                if type(hex_bits) is not str:
                    raise ValueError("invalid float encoding")
                try:
                    bits = bytes.fromhex(hex_bits)
                    if len(bits) != 8 or bits.hex() != hex_bits:
                        raise ValueError("invalid IEEE-754 bits")
                    return struct.unpack(">d", bits)[0]
                except (TypeError, ValueError, struct.error) as error:
                    raise ValueError("invalid float encoding") from error
            if tag == "b":
                fields(item, ("$t", "v"))
                return read_bytes(item["v"])
            if tag not in ("d", "l", "t", "s", "fs", "ba"):
                raise ValueError("unknown compact tagged type")
            if "$i" in item:
                fields(item, ("$t", "$i", "v"))
                reference = identifier(item)
                references[reference] = None
                active.add(reference)
            else:
                fields(item, ("$t", "v"))
                reference = None
            try:
                if tag == "ba":
                    result = bytearray(read_bytes(item["v"]))
                else:
                    parts = item["v"]
                    if type(parts) is not list:
                        raise ValueError("container items must be a list")
                    if tag == "d":
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
                        decoded_parts = [visit(part) for part in parts]
                        try:
                            result = {"l": list, "t": tuple, "s": set,
                                      "fs": frozenset}[tag](decoded_parts)
                        except TypeError as error:
                            raise ValueError("unhashable set member") from error
                        if tag in ("s", "fs") and len(result) != len(decoded_parts):
                            raise ValueError("duplicate set member")
                if reference is not None:
                    references[reference] = result
                return result
            finally:
                if reference is not None:
                    active.remove(reference)
        for key in item:
            if type(key) is not str or key.startswith("$"):
                raise ValueError("plain compact dictionary keys cannot start with '$'")
        return {key: visit(part) for key, part in item.items()}

    return visit(value)


def dumps_compact(value) -> str:
    """Encode a Python value as compact ASCII-safe JSON."""
    return json.dumps(encode_compact(value), ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def loads_compact(text: str):
    """Decode the compact JSON representation emitted by :func:`dumps_compact`."""
    def reject_constant(value):
        raise ValueError("untagged non-finite JSON constant: " + value)

    def reject_float(value):
        raise ValueError("bare JSON float is not allowed in compact wire format: " + value)

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object field")
            result[key] = value
        return result

    return decode_compact(
        json.loads(
            text,
            parse_constant=reject_constant,
            parse_float=reject_float,
            parse_int=int,
            object_pairs_hook=unique_object,
        )
    )


def validate_hook_output(value, hook: str) -> dict:
    """Validate and canonicalize a decoded hook output envelope."""
    required = {"return", "memory", "exception"}
    if hook == "choose":
        required.add("candidates")
    if type(value) is not dict or set(value) != required:
        found = sorted(str(key) for key in value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"Expected output fields {sorted(required)}, got {found}")
    if type(value["memory"]) is not dict:
        raise ValueError("memory must be the full updated dictionary")
    exception = value["exception"]
    if exception is not None:
        if (type(exception) is not dict or set(exception) != {"type", "args"}
                or not isinstance(exception["type"], str) or not exception["type"]
                or not isinstance(exception["args"], tuple)):
            raise ValueError("exception must be None or {type: nonempty str, args: tuple}")
    if hook == "choose" and type(value["candidates"]) is not list:
        raise ValueError("choose must return the updated candidates list")
    normalized = {"return": value["return"], "memory": value["memory"], "exception": exception}
    if hook == "choose":
        normalized["candidates"] = value["candidates"]
    return normalized


def encode_output_compact(output: dict, hook: str, input_arguments: dict | None = None) -> str:
    """Encode a complete hook output in compact-v2, using ``"="`` for unchanged unaliased state."""
    normalized = validate_hook_output(output, hook)
    compact_payload = dict(normalized)
    if input_arguments is not None:
        state_fields = ("memory", "candidates") if hook == "choose" else ("memory",)
        field_ids = {key: _container_ids(val) for key, val in normalized.items()}
        for field in state_fields:
            if field not in input_arguments:
                continue
            if codec.dumps(normalized[field]) != codec.dumps(input_arguments[field]):
                continue
            other_ids: set[int] = set()
            for other_key, other_val_ids in field_ids.items():
                if other_key != field:
                    other_ids |= other_val_ids
            if field_ids[field].isdisjoint(other_ids):
                compact_payload[field] = "="
    return dumps_compact(compact_payload)


def decode_output_compact(text: str, hook: str, input_arguments: dict | None = None) -> dict:
    """Decode and validate a compact-v2 hook output, expanding ``"="`` sentinels."""
    decoded = loads_compact(text)
    required = {"return", "memory", "exception"}
    if hook == "choose":
        required.add("candidates")
    if type(decoded) is not dict or set(decoded) != required:
        found = sorted(str(key) for key in decoded) if isinstance(decoded, dict) else type(decoded).__name__
        raise ValueError(f"Expected output fields {sorted(required)}, got {found}")
    if decoded["memory"] == "=":
        if not isinstance(input_arguments, dict) or "memory" not in input_arguments:
            raise ValueError("cannot resolve '=' memory sentinel without input arguments")
        decoded["memory"] = copy.deepcopy(input_arguments["memory"])
    if hook == "choose" and decoded["candidates"] == "=":
        if not isinstance(input_arguments, dict) or "candidates" not in input_arguments:
            raise ValueError("cannot resolve '=' candidates sentinel without input arguments")
        decoded["candidates"] = copy.deepcopy(input_arguments["candidates"])
    return validate_hook_output(decoded, hook)
