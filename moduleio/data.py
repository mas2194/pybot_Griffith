"""Lossless, label-isolated module-call tasks and conversation-group splits."""

from __future__ import annotations

import datetime
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from . import codec, wire


CONTRACT = "pybot-module-call-v1"
HOOKS = ("observe", "respond", "choose", "after")


def input_for_call(module, hook, arguments, *, seed=0, environment_date=None,
                   rng_state=None, wire_format="legacy-v1"):
    """Encode only information available before the call, never its answer.

    Independent fixtures explicitly initialize the random generator from ``seed``.
    Natural conversation traces instead supply the full recorded pre-call state.
    The date is an experimental input, not a claim to virtualize Python's clock.
    """
    if wire_format not in wire.WIRE_FORMATS:
        raise ValueError("unsupported wire_format: " + str(wire_format))
    if hook not in HOOKS:
        raise ValueError("unsupported hook: " + str(hook))
    if not isinstance(module, str) or not module:
        raise ValueError("module must be a nonempty name")
    expected = {"text", "memory"}
    if hook == "choose":
        expected.add("candidates")
    if hook == "after":
        expected.add("reply")
    if not isinstance(arguments, dict) or set(arguments) != expected:
        raise ValueError(f"{hook} requires exactly {sorted(expected)}")
    if not isinstance(arguments["memory"], dict):
        raise ValueError("memory must be a dictionary")
    environment = {"date": environment_date or datetime.date.today().isoformat()}
    if rng_state is None:
        environment["seed"] = int(seed)
        environment["fixture"] = "isolated_seeded_call"
    else:
        environment["random_state"] = rng_state
        environment["fixture"] = "conversation_call"
    # Keep one stable argument order regardless of the caller's JSON key order.
    ordered = {key: arguments[key] for key in ("text", "memory", "reply", "candidates")
               if key in arguments}
    if wire_format == "compact-v2":
        payload = {"module": module, "hook": hook,
                   "environment": environment, "arguments": ordered}
        return wire.dumps_compact(payload)
    payload = {"module": module, "hook": hook,
               "arguments": ordered, "environment": environment}
    return codec.dumps(payload)


def record_to_example(record, *, wire_format="legacy-v1"):
    if wire_format not in wire.WIRE_FORMATS:
        raise ValueError("unsupported wire_format: " + str(wire_format))
    if record.get("schema_version") != 1:
        raise ValueError("unsupported trace schema_version")
    arguments = codec.decode(record["input"])
    output = codec.decode(record["output"])
    metadata = record.get("metadata", {})
    isolated = metadata.get("fixture") == "isolated_seeded_call"
    if not isolated and "rng_before" not in metadata:
        raise ValueError("conversation records require the pre-call RNG state")
    prompt = input_for_call(
        record["module"], record["hook"], arguments,
        seed=metadata.get("seed", 0),
        environment_date=metadata.get("environment_date"),
        rng_state=None if isolated else codec.decode(metadata["rng_before"]),
        wire_format=wire_format,
    )
    active = (output["return"] is not None or output["exception"] is not None
              or codec.dumps(arguments["memory"]) != codec.dumps(output["memory"])
              or ("candidates" in arguments and
                  codec.dumps(arguments["candidates"]) != codec.dumps(output["candidates"])))
    output_text = (wire.encode_output_compact(output, record["hook"], arguments)
                   if wire_format == "compact-v2" else codec.dumps(output))
    return {"input_text": prompt, "output_text": output_text,
            "wire_format": wire_format,
            "group": str(record["conversation_id"]), "module": record["module"],
            "hook": record["hook"], "active": active, "record": record}


def read_jsonl(path):
    records = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: a record must be an object")
            records.append(value)
    return records


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                                    allow_nan=False) + "\n")


def prepare_records(records, *, max_seq_len=None, selected_hooks=None, wire_format="legacy-v1"):
    """Keep complete examples or report their exclusion; never truncate either side."""
    if wire_format not in wire.WIRE_FORMATS:
        raise ValueError("unsupported wire_format: " + str(wire_format))
    examples, excluded = [], []
    selected = set(selected_hooks) if selected_hooks else None
    seen_targets = {}
    for index, record in enumerate(records):
        name = record.get("module", "?") + "." + record.get("hook", "?")
        if selected is not None and name not in selected:
            excluded.append({"index": index, "hook": name, "reason": "hook_filter"})
            continue
        example = record_to_example(record, wire_format=wire_format)
        # Full bounded sequence: BOS + prompt + SEP + answer + EOS.
        sequence_tokens = (len(example["input_text"].encode("utf-8"))
                           + len(example["output_text"].encode("utf-8")) + 3)
        if max_seq_len is not None and sequence_tokens > max_seq_len:
            excluded.append({"index": index, "hook": name, "reason": "context_overflow",
                             "required_tokens": sequence_tokens,
                             "max_seq_len": max_seq_len, "active": example["active"]})
            continue
        prompt = example["input_text"]
        target = example["output_text"]
        if prompt in seen_targets and seen_targets[prompt] != target:
            raise ValueError("identical recorded input has contradictory targets: " + name
                             + "; inspect hidden state, date, and RNG before training")
        seen_targets[prompt] = target
        examples.append(example)
    return examples, {"input_records": len(records), "included_records": len(examples),
                      "active_records": sum(e["active"] for e in examples),
                      "excluded_records": len(excluded),
                      "exclusion_reasons": dict(Counter(e["reason"] for e in excluded)),
                      "excluded": excluded,
                      "hooks": dict(Counter(e["module"] + "." + e["hook"] for e in examples))}


def split_examples(examples, *, validation_fraction=0.2, seed=0):
    """Split whole conversation/family groups, joining groups with identical inputs.

    Rephrases of the same synthetic scenario should share a caller-assigned group.
    No output is examined when choosing a partition.
    """
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    parents = {example["group"]: example["group"] for example in examples}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    first_input = {}
    for example in examples:
        group = example["group"]
        prompt = example["input_text"]
        if prompt in first_input:
            a, b = find(group), find(first_input[prompt])
            parents[max(a, b)] = min(a, b)
        else:
            first_input[prompt] = group
    components = sorted({find(group) for group in parents})
    random.Random(seed).shuffle(components)
    if validation_fraction and len(components) < 2:
        raise ValueError("at least two independent conversation groups are needed for validation")
    count = (min(len(components) - 1, max(1, round(len(components) * validation_fraction)))
             if validation_fraction else 0)
    validation_components = set(components[:count])
    # A singleton task family cannot establish held-out generalization. Keep at
    # least one component for every hook in training, using task IDs only (never
    # targets). Report hooks without held-out coverage instead of testing an
    # entirely untrained module as though it were an ordinary unseen wording.
    task_components = {}
    for example in examples:
        task = example["module"] + "." + example["hook"]
        task_components.setdefault(task, set()).add(find(example["group"]))
    moved_to_train = []
    for task in sorted(task_components):
        available = task_components[task]
        if available and available <= validation_components:
            chosen = sorted(available)[0]
            validation_components.remove(chosen)
            moved_to_train.append({"hook": task, "component": chosen})
    if validation_fraction and not validation_components:
        raise ValueError("no independent validation groups remain after preserving training hook coverage")
    train, validation = [], []
    for example in examples:
        destination = validation if find(example["group"]) in validation_components else train
        destination.append(example)
    return train, validation, {
        "method": "conversation/family groups; duplicate inputs join groups",
        "seed": seed, "validation_fraction": validation_fraction,
        "train_groups": sorted({e["group"] for e in train}),
        "validation_groups": sorted({e["group"] for e in validation}),
        "train_records": len(train), "validation_records": len(validation),
        "independent_components": len(components),
        "components_kept_for_training_coverage": moved_to_train,
        "hooks_without_validation": sorted(task for task, groups in task_components.items()
                                            if not groups & validation_components),
    }


def examples_fingerprint(examples):
    digest = hashlib.sha256()
    for example in examples:
        for value in (example["group"], example["input_text"], example["output_text"]):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()
