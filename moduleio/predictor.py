"""Independent learned call prediction.  Importing this file never imports PyBot.

This predicts the same fixture as the training data: an isolated seeded call or
a recorded runtime call with its explicit pre-call RNG state.  It does not run
the original hook, change the process RNG, or apply predicted state to a live
conversation automatically.
"""

from __future__ import annotations

import copy
from typing import Any

from .evaluation import decode_output


class PredictionError(RuntimeError):
    """Learned inference did not produce a complete, valid output contract."""


class UnknownHookError(PredictionError):
    """The checkpoint contains no training examples for the requested hook."""


class ModuleIOPredictor:
    def __init__(self, model: Any, metadata: dict):
        self.model = model
        self.metadata = dict(metadata)
        trained_hooks = metadata.get("trained_hooks")
        if not isinstance(trained_hooks, (list, tuple)) or not all(isinstance(hook, str) for hook in trained_hooks):
            raise PredictionError("Checkpoint metadata must contain trained_hooks as a list of module.hook names")
        self.trained_hooks = frozenset(trained_hooks)
        fixtures = metadata.get("training_fixtures", ["isolated_seeded_call"])
        supported = {"isolated_seeded_call", "runtime_conversation"}
        if (not isinstance(fixtures, (list, tuple)) or not fixtures
                or not all(isinstance(fixture, str) and fixture in supported for fixture in fixtures)):
            raise PredictionError("training_fixtures must name isolated_seeded_call and/or runtime_conversation")
        self.training_fixtures = frozenset(fixtures)
        wire_format = metadata.get("wire_format")
        if wire_format is None:
            wire_format = getattr(getattr(model, "config", None), "wire_format", "legacy-v1")
        if wire_format not in ("legacy-v1", "compact-v2"):
            raise PredictionError(f"Unsupported wire_format: {wire_format!r}")
        self.wire_format = wire_format

    @classmethod
    def from_checkpoint(cls, path: str) -> "ModuleIOPredictor":
        # Lazy import keeps inspecting the inference contract independent of torch.
        from .training import load_checkpoint

        model, metadata = load_checkpoint(path)
        return cls(model, metadata)

    def predict_call(
        self,
        module: str,
        hook: str,
        arguments: dict,
        *,
        seed: int = 0,
        environment_date: str | None = None,
        rng_state: Any = None,
        max_new_tokens: int | None = None,
        mode: str = "hard",
    ) -> dict:
        """Predict return, complete changed arguments, and exception for one call.

        No teacher fallback or answer lookup is permitted.  The caller's input
        dictionary stays untouched even when a prediction fails.  Runtime-call
        checkpoints require the recorded pre-call ``rng_state``; a seed is not
        silently substituted for that different input contract.
        """
        identity = f"{module}.{hook}"
        if identity not in self.trained_hooks:
            raise UnknownHookError(f"No training examples for {identity}; learned inference is unavailable")
        fixture = "runtime_conversation" if rng_state is not None else "isolated_seeded_call"
        if fixture not in self.training_fixtures:
            if rng_state is None and "runtime_conversation" in self.training_fixtures:
                raise PredictionError("This checkpoint requires rng_state from the runtime call; a seed-only input was not trained")
            raise PredictionError(f"This checkpoint was not trained with {fixture} inputs")
        if type(arguments) is not dict:
            raise PredictionError("arguments must be a dictionary")
        required = {"text", "memory"}
        if hook == "after":
            required.add("reply")
        elif hook == "choose":
            required.add("candidates")
        elif hook not in {"observe", "respond"}:
            raise PredictionError(f"Unsupported hook {hook}")
        if set(arguments) != required:
            raise PredictionError(f"{hook} requires exactly {sorted(required)}")
        from .data import input_for_call

        environment = {"seed": seed, "environment_date": environment_date}
        if rng_state is not None:
            environment["rng_state"] = rng_state
        if self.wire_format != "legacy-v1":
            environment["wire_format"] = self.wire_format
        input_text = input_for_call(module, hook, copy.deepcopy(arguments), **environment)
        try:
            result = self.model.generate(input_text, max_new_tokens=max_new_tokens, mode=mode)
        except Exception as exc:
            raise PredictionError(f"Generation failed: {type(exc).__name__}: {exc}") from exc
        if result.get("eos_generated") is not True:
            raise PredictionError(f"Incomplete generated output: {result.get('stop_reason', 'missing EOS')}")
        if result.get("utf8_valid") is False:
            raise PredictionError("Generated output contains invalid UTF-8 bytes")
        try:
            return decode_output(
                result["text"],
                hook,
                input_arguments=arguments,
                wire_format=self.wire_format,
            )
        except Exception as exc:
            raise PredictionError(f"Invalid generated output: {type(exc).__name__}: {exc}") from exc


def load_predictor(path: str) -> ModuleIOPredictor:
    return ModuleIOPredictor.from_checkpoint(path)
