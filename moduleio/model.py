"""A complete hook-I/O byte model built from the official Griffith library.

The application serializes a hook's inputs and outputs; Griffith owns the
Transformer, LUTs and their differentiable/discrete implementations. Embeddings,
normalization, attention softmax and the vocabulary head remain floating point.
``hard`` means hard Griffith LUT evaluation, not a wholly Boolean machine.

Nothing in this module imports or calls the teacher's ``bot`` package. Generation
uses only the supplied input string and model weights, without answer caches,
response templates, rule fallbacks or a correct-output prefix.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from griffith_logic.convert.byte_vocab import encode_bytes
from griffith_logic.hybrid.policy import LogicSpec
from griffith_logic.layers.dwn import LearnableMapping
from griffith_logic.layers.logic_linear import LogicMLP
from griffith_logic.layers.lut import LutLayer
from griffith_logic.models import LogicLadderGPT, LogicLadderGPTConfig


PAD = 0
BOS = 1
EOS = 2
SEP = 3
BYTE_OFFSET = 4
VOCAB_SIZE = 260
IGNORE_INDEX = -100
TOKENIZER_VERSION = "utf8-bytes-offset4-pad0-bos1-eos2-sep3-v1"
CHECKPOINT_VERSION = "pybot-moduleio-griffith-v1"
GRIFFITH_REVISION = "e0675f4e36596954aa424d522e0ee966bba9a3e1"


def _integer(name: str, value: Any, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def encode_text(text: str) -> list[int]:
    """Official Griffith UTF-8 encoding plus application control-token offsets."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [value + BYTE_OFFSET for value in encode_bytes(text)]


@dataclass(frozen=True)
class ModelConfig:
    max_seq_len: int = 1024
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    mlp_ratio: int = 2
    qk_bits: int | None = None
    v_bits: int | None = None
    logic: bool = True
    attention: str | None = None
    ffn: str | None = None
    attention_score: str = "fp_dot"
    attention_norm: str = "softmax"
    n_inputs: int = 2
    ffn_n_inputs: int = 2
    parameterization: str = "warp"
    proj_depth: int = 1
    wiring: str = "fixed"
    lm_tau: float = 0.05
    gumbel_noise: bool = False
    gumbel_tau: float = 1.0
    warp_gumbel: bool = False
    weight_tying: bool = False
    pos_init: str = "default"
    embed_init_std: float | None = None
    wire_format: str = "legacy-v1"

    def __post_init__(self) -> None:
        # Griffith's LadderAttention returns v_bits channels directly into the
        # d_model-wide residual connection; it has no value output projection.
        # Resolve omitted widths from d_model, including custom-width models.
        if self.qk_bits is None:
            object.__setattr__(self, "qk_bits", self.d_model)
        if self.v_bits is None:
            object.__setattr__(self, "v_bits", self.d_model)
        for name in ("max_seq_len", "d_model", "n_heads", "n_layers", "mlp_ratio",
                     "qk_bits", "v_bits", "n_inputs", "ffn_n_inputs", "proj_depth"):
            _integer(name, getattr(self, name))
        if self.max_seq_len < 4 or self.d_model < 4:
            raise ValueError("max_seq_len and d_model must be at least 4")
        for flag in ("logic", "gumbel_noise", "warp_gumbel", "weight_tying"):
            if not isinstance(getattr(self, flag), bool):
                raise TypeError(f"{flag} must be a Boolean")
        if self.attention is None:
            object.__setattr__(self, "attention", "logic_proj" if self.logic else "fp")
        if self.ffn is None:
            object.__setattr__(self, "ffn", "logic" if self.logic else "fp")
        if self.attention not in ("fp", "logic_proj"):
            raise ValueError("attention must be fp or logic_proj")
        if self.ffn not in ("fp", "logic"):
            raise ValueError("ffn must be fp or logic")
        object.__setattr__(self, "logic", (self.attention == "logic_proj") or (self.ffn == "logic"))
        if self.n_inputs not in (2, 4, 6):
            raise ValueError("n_inputs must be 2, 4, or 6")
        if self.ffn_n_inputs not in (2, 4, 6):
            raise ValueError("ffn_n_inputs must be 2, 4, or 6")
        if self.attention_score not in ("fp_dot", "xnor_popcount"):
            raise ValueError("attention_score must be fp_dot or xnor_popcount")
        if self.attention_norm not in ("softmax", "mux_gumbel", "softmax_int", "topk_ste", "threshold"):
            raise ValueError("unsupported attention_norm")
        if self.parameterization not in ("warp", "softmax", "independent"):
            raise ValueError("parameterization must be warp, softmax, or independent")
        if self.wiring not in ("fixed", "learnable"):
            raise ValueError("wiring must be fixed or learnable")
        if (
            isinstance(self.lm_tau, bool)
            or not isinstance(self.lm_tau, (int, float))
            or not math.isfinite(self.lm_tau)
            or self.lm_tau <= 0
        ):
            raise ValueError("lm_tau must be finite and positive")
        if isinstance(self.gumbel_tau, bool) or not isinstance(self.gumbel_tau, (int, float)) or self.gumbel_tau <= 0:
            raise ValueError("gumbel_tau must be positive")
        if self.attention == "logic_proj" and self.gumbel_noise and self.parameterization != "softmax":
            raise ValueError("gumbel_noise requires parameterization='softmax'")
        if self.pos_init not in ("default", "sinusoidal"):
            raise ValueError("pos_init must be default or sinusoidal")
        if self.embed_init_std is not None and (
            isinstance(self.embed_init_std, bool)
            or not isinstance(self.embed_init_std, (int, float))
            or not math.isfinite(self.embed_init_std)
            or self.embed_init_std <= 0
        ):
            raise ValueError("embed_init_std must be finite and positive")
        if self.wire_format not in ("legacy-v1", "compact-v2"):
            raise ValueError("wire_format must be legacy-v1 or compact-v2")
        if any(value % self.n_heads for value in (self.d_model, self.qk_bits, self.v_bits)):
            raise ValueError("d_model, qk_bits and v_bits must be divisible by n_heads")
        if self.attention == "logic_proj" and self.v_bits != self.d_model:
            raise ValueError("logic attention requires v_bits == d_model for its residual connection")


class ExampleTooLongError(ValueError):
    """The full input and output cannot fit; no text has been truncated."""


def prepare_example(input_text: str, output_text: str, max_seq_len: int) -> dict:
    """Encode every input/output byte, supervising output bytes and true EOS only.

    The first answer byte is predicted from the input's SEP. Later teacher-forced
    predictions see earlier output bytes only. Full-sequence length, including
    EOS, must fit the declared context; overlength examples fail explicitly.
    """
    _integer("max_seq_len", max_seq_len, 4)
    prompt = [BOS] + encode_text(input_text) + [SEP]
    answer = encode_text(output_text) + [EOS]
    full = prompt + answer
    if len(full) > max_seq_len:
        raise ExampleTooLongError(
            f"complete example needs {len(full)} tokens "
            f"(input {len(prompt)}, output including EOS {len(answer)}), "
            f"but max_seq_len={max_seq_len}; no input or output was truncated"
        )
    return {
        "input_ids": full[:-1],
        "labels": [IGNORE_INDEX] * (len(prompt) - 1) + answer,
        "prompt_ids": prompt,
        "answer_ids": answer,
        "input_tokens": len(prompt),
        "output_tokens": len(answer),
        "total_tokens": len(full),
    }


@contextmanager
def griffith_mode(module: nn.Module, mode: str):
    """Use native deterministic soft, train_soft, hard-ST, or discrete LUT execution.

    ``soft`` and ``hard`` are strictly deterministic (all Gumbel noise disabled).
    ``train_soft`` enables stochastic Gumbel smoothing/noise when configured on the
    model during training only. All caller modes are restored on exit.
    """
    if mode not in ("soft", "train_soft", "hard", "ste"):
        raise ValueError("mode must be soft, train_soft, hard or ste")
    states = [(child, child.training) for child in module.modules()]
    lut_states = [(child, child.selection, child.gumbel, child.gumbel_noise)
                  for child, _ in states if isinstance(child, LutLayer)]
    lm_states = [(child, child.forward_mode)
                 for child, _ in states if isinstance(child, LearnableMapping)]
    cfg = getattr(module, "config", None)
    use_warp_gumbel = bool(getattr(cfg, "warp_gumbel", False)) if mode == "train_soft" else False
    use_softmax_gumbel = bool(getattr(cfg, "gumbel_noise", False)) if mode == "train_soft" else False
    try:
        module.train(mode != "hard")
        for layer, _, _, _ in lut_states:
            if mode == "ste":
                if layer.parameterization not in ("softmax", "warp"):
                    raise ValueError(
                        f"ste mode requires softmax or warp parameterization, got {layer.parameterization!r}"
                    )
                layer.selection = "hard_st"
            else:
                layer.selection = "soft"
            layer.gumbel = use_warp_gumbel and (layer.parameterization == "warp")
            layer.gumbel_noise = use_softmax_gumbel and (layer.parameterization == "softmax")
        if mode != "train_soft":
            for lm, _ in lm_states:
                lm.forward_mode = "hard"
        yield
    finally:
        for child, training in states:
            child.training = training
        for layer, selection, gumbel, gumbel_noise in lut_states:
            layer.selection = selection
            layer.gumbel = gumbel
            layer.gumbel_noise = gumbel_noise
        for lm, forward_mode in lm_states:
            lm.forward_mode = forward_mode


def _upgrade_to_learnable_wiring(module: nn.Module, lm_tau: float) -> None:
    """Recursively replace LutLayer modules with official LearnableMapping-enabled LutLayers."""
    for name, child in list(module.named_children()):
        if isinstance(child, LutLayer):
            setattr(module, name, LutLayer(
                in_dim=child.in_dim,
                out_dim=child.out_dim,
                n_inputs=child.n_inputs,
                grad_factor=child.grad_factor,
                parameterization=child.parameterization,
                selection=child.selection,
                gumbel_noise=child.gumbel_noise,
                gumbel_tau=child.gumbel_tau,
                wiring="learnable",
                lm_tau=lm_tau,
            ))
        else:
            _upgrade_to_learnable_wiring(child, lm_tau)


class ModuleIOModel(nn.Module):
    """Griffith's causal Transformer with a lossless byte I/O interface."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.network = LogicLadderGPT(LogicLadderGPTConfig(
            vocab_size=VOCAB_SIZE,
            seq_len=c.max_seq_len,
            d_model=c.d_model,
            n_heads=c.n_heads,
            n_layers=c.n_layers,
            mlp_ratio=c.mlp_ratio,
            dropout=0.0,
            attention=c.attention,
            ffn=c.ffn,
            weight_tying=c.weight_tying,
            attention_kwargs=({
                "qk_bits": c.qk_bits,
                "v_bits": c.v_bits,
                "n_inputs": c.n_inputs,
                "parameterization": c.parameterization,
                "proj_depth": c.proj_depth,
                "score": c.attention_score,
                "norm": c.attention_norm,
                "tau": 1.0,
                "gumbel_noise": c.gumbel_noise,
                "gumbel_tau": c.gumbel_tau,
            } if c.attention == "logic_proj" else {}),
        ))
        if c.ffn == "logic" and c.ffn_n_inputs != 2:
            ffn_spec = LogicSpec(
                parameterization="warp",
                encoder="sigmoid",
                decoder="groupsum_centered",
                depth=2,
                width=c.d_model * c.mlp_ratio,
                n_inputs=c.ffn_n_inputs,
            )
            for block in self.network.blocks:
                block.ffn = LogicMLP(c.d_model, c.d_model, spec=ffn_spec)
        if c.wiring == "learnable":
            _upgrade_to_learnable_wiring(self.network, c.lm_tau)
        for layer in self.network.modules():
            if isinstance(layer, LutLayer):
                layer.gumbel = False
                layer.gumbel_noise = False
                layer.selection = "soft"
        with torch.no_grad():
            if c.pos_init == "sinusoidal":
                scale = float(c.embed_init_std if c.embed_init_std is not None else 0.25)
                pe = torch.zeros(c.max_seq_len, c.d_model, dtype=torch.float32)
                positions = torch.arange(c.max_seq_len, dtype=torch.float32).unsqueeze(1)
                div = torch.exp(
                    torch.arange(0, c.d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / c.d_model)
                )
                pe[:, 0::2] = torch.sin(positions * div)
                pe[:, 1::2] = torch.cos(positions * div[: c.d_model // 2])
                self.network.pos_embed.weight.copy_(pe * scale)
                self.network.token_embed.weight.normal_(0.0, scale)
            elif c.embed_init_std is not None:
                scale = float(c.embed_init_std)
                self.network.token_embed.weight.normal_(0.0, scale)
                self.network.pos_embed.weight.normal_(0.0, scale)

    def forward(self, input_ids: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty shape [batch, time]")
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError("input_ids must contain integer token IDs")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ExampleTooLongError("input exceeds max_seq_len; context is never cropped")
        if mode is None:
            return self.network(input_ids)
        with griffith_mode(self, mode):
            return self.network(input_ids)

    @torch.no_grad()
    def generate(self, input_text: str, max_new_tokens: int | None = None,
                 mode: str = "hard") -> dict:
        """Greedily generate a complete output, reporting EOS or budget exhaustion.

        ``max_new_tokens`` includes the possible EOS token. Omitting it uses the
        unused context capacity. A requested budget that would exceed capacity
        is an error, not permission to crop the original input. ``token_ids``
        contains content bytes only; ``generated_tokens`` includes EOS if seen.
        """
        if mode not in ("hard", "soft"):
            raise ValueError("generation mode must be hard or soft")
        prefix = [BOS] + encode_text(input_text) + [SEP]
        capacity = self.config.max_seq_len
        remaining = capacity - len(prefix)
        from_context = max_new_tokens is None
        if max_new_tokens is None:
            max_new_tokens = remaining
        else:
            _integer("max_new_tokens", max_new_tokens, 0)
        if remaining < 1:
            raise ExampleTooLongError(
                f"input uses {len(prefix)} tokens with max_seq_len={capacity}; "
                "no output token fits and the input will not be cropped"
            )
        if max_new_tokens > remaining:
            raise ExampleTooLongError(
                f"input {len(prefix)} + requested output budget {max_new_tokens} "
                f"exceeds max_seq_len={capacity}; maximum output budget is {remaining}"
            )
        device = next(self.parameters()).device
        sequence = torch.tensor([prefix], dtype=torch.long, device=device)
        generated: list[int] = []
        eos_generated = False
        with griffith_mode(self, mode):
            for _ in range(max_new_tokens):
                scores = self.network(sequence)[:, -1, :].clone()
                scores[:, [PAD, BOS, SEP]] = -torch.inf
                token = scores.argmax(dim=-1)
                value = int(token.item())
                if value == EOS:
                    eos_generated = True
                    break
                generated.append(value)
                sequence = torch.cat((sequence, token[:, None]), dim=1)
        raw = bytes(value - BYTE_OFFSET for value in generated)
        try:
            text = raw.decode("utf-8", errors="strict")
            utf8_valid = True
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            utf8_valid = False
        return {
            "text": text,
            "token_ids": generated,
            "stop_reason": ("eos" if eos_generated else
                            "context_limit" if from_context else "max_new_tokens"),
            "eos_generated": eos_generated,
            "utf8_valid": utf8_valid,
            "input_tokens": len(prefix),
            "generated_tokens": len(generated) + int(eos_generated),
            "output_bytes": len(raw),
            "max_new_tokens": max_new_tokens,
            "max_seq_len": capacity,
            "mode": mode,
        }


def save_checkpoint(model: ModuleIOModel, path, metadata: dict | None = None) -> Path:
    """Save tensors and primitive metadata; no teacher source or response cache."""
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")
    metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
    try:
        version = importlib_metadata.version("griffith-logic")
    except importlib_metadata.PackageNotFoundError:
        version = "source-checkout"
    payload = {
        "format": CHECKPOINT_VERSION,
        "model_config": asdict(model.config),
        "tokenizer": TOKENIZER_VERSION,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": metadata,
        "griffith": {"package_version": version,
                     "source_revision_constraint": GRIFFITH_REVISION},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return destination


def load_checkpoint(path, device: str = "cpu") -> tuple[ModuleIOModel, dict]:
    """Load safely without importing PyBot, then default to hard LUT evaluation."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_VERSION:
        raise ValueError("not a supported module-I/O Griffith checkpoint")
    if payload.get("tokenizer") != TOKENIZER_VERSION:
        raise ValueError("checkpoint tokenizer does not match the byte I/O contract")
    model = ModuleIOModel(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, payload.get("metadata", {})
