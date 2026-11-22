"""Observe complete PyBot hook calls without changing the original runtime.

Both ``input`` and ``output`` in every record are a single ``codec.encode`` value;
decode them to recover the named arguments and full result/state.  RNG states are
complete typed values in metadata.  They are provenance, not an implicit promise
that a learner can reproduce stochastic calls without being given that state.

The clock is observed, not frozen.  PyBot also has module caches and a local
knowledge base: traces identify the source and date but do not claim to snapshot
arbitrary Python process state.  Cyclic/custom-object boundaries fail explicitly.
"""
from __future__ import annotations

import copy
import datetime
import functools
import hashlib
import importlib
import inspect
import random
import re
import subprocess
from pathlib import Path

from . import codec


HOOK_ARGUMENTS = {
    "observe": ("text", "memory"),
    "respond": ("text", "memory"),
    "choose": ("text", "memory", "candidates"),
    "after": ("text", "reply", "memory"),
}
_TABLES = {"observe": "_OBSERVERS", "respond": "_HANDLERS", "after": "_AFTER"}


class TeacherCaptureError(RuntimeError):
    """Serialization failed; the original teacher execution was left intact."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__("%d hook snapshot(s) failed; no partial dataset returned" % len(errors))


def _load_runtime():
    return importlib.import_module("bot.runtime")


def _load_module(name):
    if type(name) is not str or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("module must be a name inside bot.modules")
    return importlib.import_module("bot.modules." + name)


def _rows(runtime):
    rows = []
    for hook in ("observe", "respond", "choose", "after"):
        if hook == "choose":
            function = getattr(runtime, "_ARBITER", None)
            if callable(function):
                priority = next((p for n, p, _ in getattr(runtime, "_MODULES", ())
                                 if n == "arbiter"), 0)
                rows.append(("arbiter", "choose", priority, function))
        else:
            rows.extend((name, hook, priority, function)
                        for name, priority, function in getattr(runtime, _TABLES[hook]))
    return rows


def _inventory(runtime):
    return [{"module": name, "hook": hook, "priority": priority,
             "id": name + "." + hook}
            for name, hook, priority, _ in _rows(runtime)]


def discover_hooks():
    """Return actual runtime-dispatched hooks, in phase/priority order.

    Routing callbacks such as ``TAKES_AS_IT_COMES`` are not runtime phase hooks;
    they execute inside the recorded arbiter call and are not a separate target.
    """
    return _inventory(_load_runtime())


def _source(function):
    root = Path(__file__).resolve().parents[1]
    result = {"python_module": getattr(function, "__module__", None),
              "function": getattr(function, "__qualname__", None)}
    try:
        filename = inspect.getsourcefile(function)
        if filename:
            path = Path(filename).resolve()
            try:
                result["path"] = str(path.relative_to(root))
            except ValueError:
                result["path"] = str(path)
            result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, TypeError):
        pass
    try:
        process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                                 capture_output=True, text=True, timeout=5, check=False)
        if process.returncode == 0:
            result["git_commit"] = process.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _validate_common(conversation_id, seed):
    if type(conversation_id) is not str or not conversation_id.strip():
        raise ValueError("conversation_id must be a nonempty string")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")


class _Recorder:
    def __init__(self, conversation_id, seed, fixture):
        self.conversation_id, self.seed, self.fixture = conversation_id, seed, fixture
        self.records, self.errors = [], []
        self.turn_index = 0
        self.call_index = 0
        self.sources = {}

    def snapshot(self, value, name, hook, stage):
        try:
            return codec.encode(value)
        except Exception as error:
            self.errors.append({"module": name, "hook": hook, "turn_index": self.turn_index,
                                "stage": stage, "type": type(error).__name__,
                                "message": str(error)})
            return None

    def invoke(self, name, hook, priority, function, arguments):
        index = self.call_index
        self.call_index += 1
        rng_before = random.getstate()
        inputs = self.snapshot(arguments, name, hook, "input")
        result, exception = None, None
        # Resolve metadata when wrappers are built, never inside teacher execution.
        source = self.sources.get((name, hook), {})
        date = datetime.date.today().isoformat()
        try:
            result = function(*(arguments[key] for key in HOOK_ARGUMENTS[hook]))
            return result
        except BaseException as error:
            exception = {"type": type(error).__module__ + "." + type(error).__qualname__,
                         "args": error.args}
            raise
        finally:
            rng_after = random.getstate()
            outputs = {"return": result, "memory": arguments["memory"], "exception": exception}
            if hook == "choose":
                outputs["candidates"] = arguments["candidates"]
            self.records.append({
                "schema_version": 1, "conversation_id": self.conversation_id,
                "turn_index": self.turn_index, "module": name, "hook": hook,
                "input": inputs, "output": self.snapshot(outputs, name, hook, "output"),
                "metadata": {
                    "call_index": index, "priority": priority,
                    "rng_before": self.snapshot(rng_before, name, hook, "rng_before"),
                    "rng_after": self.snapshot(rng_after, name, hook, "rng_after"),
                    "environment_date": date, "seed": self.seed, "fixture": self.fixture,
                    "source": source, "source_commit": source.get("git_commit"),
                },
            })

    def wrapper(self, name, hook, priority, function):
        self.sources[(name, hook)] = _source(function)

        @functools.wraps(function)
        def wrapped(*args):
            arguments = dict(zip(HOOK_ARGUMENTS[hook], args))
            return self.invoke(name, hook, priority, function, arguments)

        return wrapped


def _selected_keys(selected_hooks, inventory):
    known = {row["id"] for row in inventory}
    if selected_hooks is None:
        return known
    if isinstance(selected_hooks, str):
        selected_hooks = [selected_hooks]
    selected = set()
    for item in selected_hooks:
        if type(item) in (list, tuple) and len(item) == 2:
            item = ".".join(item)
        if type(item) is not str or item not in known:
            raise ValueError("selected_hooks must name existing module.hook IDs")
        selected.add(item)
    return selected


def capture_conversation(texts, *, conversation_id, seed=0, selected_hooks=None):
    """Observe a real conversation and return ``(records, summary)``.

    ``selected_hooks`` is None for all hooks, or IDs such as ``intent.observe``
    (``(module, hook)`` pairs are also accepted).  Filtering changes only what is
    recorded.  Original modules, candidate isolation, return identities, exceptions,
    and RNG progression are unchanged.  Caller RNG state and runtime hook tables
    are restored even on failure.  This temporary wrapper is not thread-safe;
    run collection without concurrent calls to the same runtime.

    Summary ``turn_results`` and ``final_memory`` use the same typed codec.  Failed
    snapshots raise only after the conversation, never inside the original runtime.
    """
    _validate_common(conversation_id, seed)
    if type(texts) not in (list, tuple) or not texts or any(type(t) is not str for t in texts):
        raise ValueError("texts must be a nonempty list or tuple of strings")
    caller_rng = random.getstate()
    originals = {}
    try:
        runtime = _load_runtime()
        inventory = _inventory(runtime)
        selected = _selected_keys(selected_hooks, inventory)
        recorder = _Recorder(conversation_id, seed, "runtime_conversation")
        for hook, table in _TABLES.items():
            originals[table] = getattr(runtime, table)
            setattr(runtime, table, [
                (name, priority, recorder.wrapper(name, hook, priority, function)
                 if name + "." + hook in selected else function)
                for name, priority, function in originals[table]
            ])
        originals["_ARBITER"] = runtime._ARBITER
        if "arbiter.choose" in selected and callable(runtime._ARBITER):
            priority = next(row["priority"] for row in inventory if row["id"] == "arbiter.choose")
            runtime._ARBITER = recorder.wrapper("arbiter", "choose", priority, runtime._ARBITER)
        random.seed(seed)
        memory, results = {}, []
        for index, text in enumerate(texts):
            recorder.turn_index = index
            result = runtime.turn(text, memory)
            results.append(recorder.snapshot(result, "<runtime>", "turn", "turn_result"))
        final_memory = recorder.snapshot(memory, "<runtime>", "turn", "final_memory")
        if recorder.errors:
            raise TeacherCaptureError(recorder.errors)
        records = sorted(recorder.records, key=lambda record: record["metadata"]["call_index"])
        summary = {
            "conversation_id": conversation_id, "seed": seed, "turn_count": len(texts),
            "record_count": len(records), "selected_hooks": sorted(selected),
            "inventory": inventory, "turn_results": results, "final_memory": final_memory,
            "environment_date": datetime.date.today().isoformat(),
            "fixture": "runtime_conversation",
            "limitations": ["RNG states are metadata; stochastic calls require those states for replay.",
                            "Clock and module-global caches are not frozen or snapshotted.",
                            "Routing callbacks execute inside arbiter; they are not separate phase hooks."],
        }
        return records, summary
    finally:
        if originals:
            for table, original in originals.items():
                setattr(runtime, table, original)
        random.setstate(caller_rng)


def capture_call(module, hook, arguments, *, conversation_id, seed=0):
    """Capture an isolated seeded call using a deep copy of caller arguments.

    Arguments must exactly match the named hook interface.  A teacher exception
    is an output record; a collection/import/validation error raises.  No parser,
    lookup result, or other teacher-derived input is added to the caller input.
    """
    _validate_common(conversation_id, seed)
    if hook not in HOOK_ARGUMENTS:
        raise ValueError("unknown hook: expected observe, respond, choose, or after")
    if type(arguments) is not dict or set(arguments) != set(HOOK_ARGUMENTS[hook]):
        raise ValueError("arguments must exactly match the named hook interface")
    caller_rng = random.getstate()
    try:
        loaded = _load_module(module)
        function = getattr(loaded, hook, None)
        if not callable(function):
            raise ValueError("module has no callable " + hook + " hook")
        # Validate before deepcopy so unsupported custom objects cannot run their
        # __deepcopy__ method or be silently converted into a different type.
        codec.encode(arguments)
        arguments = copy.deepcopy(arguments)
        recorder = _Recorder(conversation_id, seed, "isolated_seeded_call")
        recorder.sources[(module, hook)] = _source(function)
        random.seed(seed)
        try:
            recorder.invoke(module, hook, getattr(loaded, "PRIORITY", 0), function, arguments)
        except Exception:
            # The original exception is already a complete output in the record.
            pass
        if recorder.errors:
            raise TeacherCaptureError(recorder.errors)
        return recorder.records[0]
    finally:
        random.setstate(caller_rng)
