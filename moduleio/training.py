"""Teacher-output-only training for complete, bounded module I/O strings.

Teacher-forced token accuracy is a diagnostic. It is deliberately labeled as
such and must not be described as free-generation or complete-hook accuracy.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from griffith_logic.convert.distill import kd_kl_loss_seq
from griffith_logic.layers.dwn import LearnableMapping
from griffith_logic.layers.lut import LutLayer

from .model import (IGNORE_INDEX, PAD, ModelConfig, ModuleIOModel,
                    load_checkpoint, prepare_example, save_checkpoint)


def resolve_device(device: str = "auto") -> str:
    """Resolve ``auto``/``cpu``/``mps``/``cuda`` into an available PyTorch device."""
    if device == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available")
        return "mps"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"unsupported device: {device!r}")


def _prepare(examples: list[dict], config: ModelConfig) -> list[dict]:
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a nonempty list")
    prepared = []
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise TypeError(f"example {index} must be a dictionary")
        if "input_text" not in example or "output_text" not in example:
            raise ValueError(f"example {index} requires input_text and output_text")
        try:
            prepared.append(prepare_example(example["input_text"], example["output_text"],
                                             config.max_seq_len))
        except ValueError as exc:
            raise ValueError(f"example {index} ({example.get('module', '?')}."
                             f"{example.get('hook', '?')}): {exc}") from exc
    return prepared


def collate(examples: list[dict], device: str | torch.device = "cpu") -> dict:
    """Right-pad inputs; causal masking makes future padding unavailable to loss."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    length = max(len(example["input_ids"]) for example in examples)
    inputs = torch.full((len(examples), length), PAD, dtype=torch.long, device=device)
    labels = torch.full_like(inputs, IGNORE_INDEX)
    for row, example in enumerate(examples):
        size = len(example["input_ids"])
        inputs[row, :size] = torch.tensor(example["input_ids"], device=device)
        labels[row, :size] = torch.tensor(example["labels"], device=device)
    return {"input_ids": inputs, "labels": labels}


@torch.no_grad()
def _metrics(model: ModuleIOModel, prepared: list[dict], mode: str, batch_size: int) -> dict:
    device = next(model.parameters()).device
    loss_sum = 0.0
    token_count = correct = sequence_correct = 0
    for start in range(0, len(prepared), batch_size):
        batch = collate(prepared[start:start + batch_size], device)
        logits = model(batch["input_ids"], mode=mode)
        labels = batch["labels"]
        mask = labels != IGNORE_INDEX
        loss_sum += float(
            F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            ).detach()
        )
        hits = (logits.argmax(-1) == labels) | ~mask
        token_count += int(mask.sum())
        correct += int((hits & mask).sum())
        sequence_correct += int(hits.all(dim=1).sum())
    nll = loss_sum / token_count
    return {
        "conditioning": "correct output prefix (teacher forcing); not free generation",
        "mode": mode,
        "examples": len(prepared),
        "output_tokens_including_eos": token_count,
        "output_nll_nats": nll,
        "output_bits_per_token": nll / math.log(2),
        "teacher_forced_token_accuracy": correct / token_count,
        "teacher_forced_sequence_accuracy": sequence_correct / len(prepared),
    }


def teacher_forced_metrics(model: ModuleIOModel, examples: list[dict], mode: str = "hard",
                           batch_size: int = 4) -> dict:
    if mode not in ("soft", "hard"):
        raise ValueError("evaluation mode must be soft or hard")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return _metrics(model, _prepare(examples, model.config), mode, batch_size)


def _batch_stream(prepared: list[dict], batch_size: int, rng: random.Random, sampling: str):
    if sampling == "random" or len(prepared) < batch_size:
        while True:
            yield [rng.choice(prepared) for _ in range(batch_size)]
    mega_size = min(len(prepared), max(batch_size * 8, batch_size))
    pool: list[dict] = []
    while True:
        while len(pool) < mega_size:
            epoch_items = list(prepared)
            rng.shuffle(epoch_items)
            pool.extend(epoch_items)
        chunk = pool[:mega_size]
        del pool[:mega_size]
        chunk.sort(key=lambda item: item["total_tokens"] + rng.uniform(-8.0, 8.0))
        batches = [
            chunk[index:index + batch_size]
            for index in range(0, len(chunk) - batch_size + 1, batch_size)
        ]
        leftover = len(chunk) % batch_size
        if leftover:
            pool = chunk[-leftover:] + pool
        rng.shuffle(batches)
        for batch_items in batches:
            yield batch_items


def _lr_multiplier(step: int, total_steps: int, lr_schedule: str,
                   warmup_steps: int, min_lr_ratio: float) -> float:
    if lr_schedule == "constant":
        if warmup_steps > 0 and step <= warmup_steps:
            return step / warmup_steps
        return 1.0
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    if lr_schedule == "wsd":
        decay_start = max(warmup_steps, int(0.7 * total_steps))
        if step <= decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))


def fit(examples: list[dict], config: ModelConfig | None = None, *,
        steps: int = 100, batch_size: int = 4, grad_accum: int = 1, seed: int = 0,
        learning_rate: float = 0.003, threads: int = 1,
        train_mode: str = "soft", device: str = "cpu",
        loss_weighting: str = "example", lr_schedule: str = "cosine",
        warmup_steps: int = 0, min_lr_ratio: float = 0.05,
        weight_decay: float = 0.0, lut_lr_scale: float = 1.0,
        wire_lr_scale: float = 3.0,
        temp_start: float = 2.0, temp_end: float = 0.5,
        aux_lambda: float = 0.01, sampling: str = "epoch_bucketed",
        init_from: str | Path | None = None,
        teacher_checkpoint: str | Path | None = None,
        kd_alpha: float = 0.5,
        kd_temp: float = 2.0,
        attn_kd_weight: float = 0.0,
        progress=None) -> tuple[ModuleIOModel, dict]:
    """Fit all declared output bytes; reject every overlength record explicitly.

    Callers supply only training examples. Split assignment, held-out evaluation
    and free-generation semantic comparison belong to the dataset/evaluation
    layers. This function never imports or executes the teacher.
    """
    config = config or ModelConfig()
    for name, value in (("steps", steps), ("batch_size", batch_size),
                        ("grad_accum", grad_accum), ("threads", threads)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("warmup_steps must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    for name, value in (("learning_rate", learning_rate), ("lut_lr_scale", lut_lr_scale),
                        ("wire_lr_scale", wire_lr_scale),
                        ("temp_start", temp_start), ("temp_end", temp_end),
                        ("kd_temp", kd_temp)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name, value in (("weight_decay", weight_decay), ("aux_lambda", aux_lambda),
                        ("min_lr_ratio", min_lr_ratio), ("attn_kd_weight", attn_kd_weight)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if isinstance(kd_alpha, bool) or not isinstance(kd_alpha, (int, float)) \
            or not math.isfinite(kd_alpha) or not (0.0 <= kd_alpha <= 1.0):
        raise ValueError("kd_alpha must be finite and in [0.0, 1.0]")
    if attn_kd_weight > 0.0 and teacher_checkpoint is None:
        raise ValueError("attn_kd_weight requires teacher_checkpoint")
    if train_mode not in ("soft", "train_soft", "ste", "soft_to_ste"):
        raise ValueError("train_mode must be soft, train_soft, ste, or soft_to_ste")
    if loss_weighting not in ("example", "token"):
        raise ValueError("loss_weighting must be example or token")
    if lr_schedule not in ("wsd", "cosine", "constant"):
        raise ValueError("lr_schedule must be wsd, cosine, or constant")
    if sampling not in ("epoch_bucketed", "random"):
        raise ValueError("sampling must be epoch_bucketed or random")
    prepared = _prepare(examples, config)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = ModuleIOModel(config)
    if init_from is not None:
        src_model, _ = load_checkpoint(init_from, device="cpu")
        src_sd = src_model.state_dict()
        own_sd = model.state_dict()
        for key, value in src_sd.items():
            if key in own_sd and own_sd[key].shape == value.shape:
                own_sd[key] = value.detach().cpu().clone()
        model.load_state_dict(own_sd, strict=True)
        del src_model
    model = model.to(device)
    teacher = None
    if teacher_checkpoint is not None:
        teacher, _ = load_checkpoint(teacher_checkpoint, device=device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    lm_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, LearnableMapping)
        for param in module.parameters()
    }
    lut_param_ids = {
        id(param)
        for module in model.modules()
        if isinstance(module, LutLayer)
        for param in module.parameters()
    } - lm_param_ids
    base_params, lut_params, lm_params = [], [], []
    seen_ids = set()
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen_ids:
            continue
        seen_ids.add(id(param))
        if id(param) in lm_param_ids:
            lm_params.append(param)
        elif id(param) in lut_param_ids:
            lut_params.append(param)
        else:
            base_params.append(param)
    param_groups = []
    if base_params:
        param_groups.append({"params": base_params, "lr": learning_rate, "weight_decay": weight_decay})
    if lut_params:
        param_groups.append({"params": lut_params, "lr": learning_rate * lut_lr_scale, "weight_decay": 0.0})
    if lm_params:
        param_groups.append({"params": lm_params, "lr": learning_rate * wire_lr_scale, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    lm_modules = [module for module in model.modules() if isinstance(module, LearnableMapping)]

    before = {mode: _metrics(model, prepared, mode, batch_size) for mode in ("soft", "hard")}
    curve = []
    batches = _batch_stream(prepared, batch_size, rng, sampling)
    ste_start = int(0.8 * steps)
    started = time.perf_counter()
    for step in range(1, steps + 1):
        mult = _lr_multiplier(step, steps, lr_schedule, warmup_steps, min_lr_ratio)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * mult
        if config.attention == "logic_proj" and config.attention_norm in ("mux_gumbel", "topk_ste", "threshold"):
            tau = temp_start + (temp_end - temp_start) * ((step - 1) / max(1, steps - 1))
            model.network.set_temperature(tau)
        if lm_modules:
            if train_mode == "soft_to_ste" and step <= ste_start:
                tau_soft = 0.2 * ((0.01 / 0.2) ** ((step - 1) / max(1, ste_start - 1)))
                for lm in lm_modules:
                    lm.forward_mode = "soft"
                    lm.tau_soft = tau_soft
            else:
                for lm in lm_modules:
                    lm.forward_mode = "hard"
        active_mode = (
            "ste" if (train_mode == "soft_to_ste" and step > ste_start)
            else ("train_soft" if train_mode == "soft_to_ste" else train_mode)
        )
        optimizer.zero_grad(set_to_none=True)
        ce_sum = 0.0
        total_sum = 0.0
        for _ in range(grad_accum):
            chosen = next(batches)
            batch = collate(chosen, device)
            logits = model(batch["input_ids"], mode=active_mode)
            labels = batch["labels"]
            mask = labels != IGNORE_INDEX
            if loss_weighting == "token":
                ce_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
            else:
                token_losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                ).reshape(labels.shape)
                per_example_tokens = mask.sum(dim=1).clamp_min(1)
                ce_loss = (token_losses.sum(dim=1) / per_example_tokens).mean()
            loss = ce_loss
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(batch["input_ids"], mode="hard")
                    t_attns = (
                        [a.detach() for a in teacher.network.last_attn_weights if a is not None]
                        if attn_kd_weight > 0.0 else []
                    )
                kd_loss = (
                    kd_kl_loss_seq(logits, t_logits, temperature=kd_temp, mask=mask)
                    if mask.any() else logits.new_zeros(())
                )
                loss = (1.0 - kd_alpha) * ce_loss + kd_alpha * kd_loss
                if attn_kd_weight > 0.0:
                    akd_terms = [
                        (-(ta * sa.clamp_min(1e-7).log()).sum(dim=-1)).mean()
                        for ta, sa in zip(t_attns, model.network.last_attn_weights)
                        if ta is not None and sa is not None and ta.shape == sa.shape
                    ]
                    if akd_terms:
                        loss = loss + attn_kd_weight * (sum(akd_terms) / len(akd_terms))
            aux = model.network.pop_aux_loss()
            total_loss = loss + aux_lambda * aux if (aux_lambda and isinstance(aux, torch.Tensor)) else loss
            if not torch.isfinite(total_loss):
                raise RuntimeError("nonfinite training loss; checkpoint has not been saved")
            (total_loss if grad_accum == 1 else total_loss / grad_accum).backward()
            ce_sum += float(ce_loss.detach())
            total_sum += float(total_loss.detach())
            if str(device) == "mps" and batch["input_ids"].shape[1] > 512:
                torch.mps.empty_cache()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("nonfinite training gradients; checkpoint has not been saved")
        optimizer.step()
        if str(device) == "mps" and (step % 50 == 0 or step == steps):
            torch.mps.empty_cache()
        if step == 1 or step == steps or step % max(1, steps // 10) == 0:
            point = {"step": step, "output_nll_nats": ce_sum / grad_accum,
                     "total_loss": total_sum / grad_accum,
                     "lr": float(optimizer.param_groups[0]["lr"]),
                     "mode": active_mode,
                     "gradient_norm_before_clipping": float(gradient_norm),
                     "elapsed_seconds": time.perf_counter() - started}
            curve.append(point)
            if progress is not None:
                progress(point)
    elapsed = time.perf_counter() - started
    for lm in lm_modules:
        lm.forward_mode = "hard"
    if teacher is not None:
        del teacher
    if str(device) == "mps":
        torch.mps.empty_cache()
    after = {mode: _metrics(model, prepared, mode, batch_size) for mode in ("soft", "hard")}
    if str(device) == "mps":
        torch.mps.empty_cache()
    source = [{"input_text": example["input_text"], "output_text": example["output_text"]}
              for example in examples]
    dataset_hash = hashlib.sha256(json.dumps(source, sort_keys=True,
        ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    report = {
        "format": "moduleio-training-report-v1",
        "model": asdict(config),
        "training": {"steps": steps, "batch_size": batch_size, "grad_accum": grad_accum, "seed": seed,
                     "learning_rate": learning_rate, "lut_lr_scale": lut_lr_scale,
                     "wire_lr_scale": wire_lr_scale,
                     "lr_schedule": lr_schedule, "warmup_steps": warmup_steps,
                     "weight_decay": weight_decay, "loss_weighting": loss_weighting,
                     "temp_start": temp_start, "temp_end": temp_end,
                     "aux_lambda": aux_lambda, "sampling": sampling,
                     "init_from": str(init_from) if init_from else None,
                     "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint else None,
                     "kd_alpha": kd_alpha if teacher_checkpoint else 0.0,
                     "kd_temp": kd_temp if teacher_checkpoint else None,
                     "attn_kd_weight": attn_kd_weight if teacher_checkpoint else 0.0,
                     "threads": threads, "train_mode": train_mode, "device": str(device)},
        "training_examples": len(examples),
        "dataset_sha256": dataset_hash,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "max_input_tokens": max(example["input_tokens"] for example in prepared),
        "max_output_tokens_including_eos": max(example["output_tokens"] for example in prepared),
        "max_total_tokens": max(example["total_tokens"] for example in prepared),
        "truncated_examples": 0,
        "training_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "teacher_forced_before": before,
        "teacher_forced_after": after,
        "curve": curve,
        "torch_version": str(torch.__version__),
        "implementation": (
            f"official Griffith LogicLadderGPT (attention={config.attention}, ffn={config.ffn}, "
            f"n_inputs={config.n_inputs}, ffn_n_inputs={config.ffn_n_inputs}, wiring={config.wiring}); "
            "FP embedding/head/normalization remain (and softmax unless mux_gumbel/selector norm)"
        ),
        "limitations": "Teacher-forced diagnostics do not establish complete free-generation I/O agreement.",
    }
    return model.eval(), report
