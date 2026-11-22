"""Explicit teacher collection, full-call training, and teacher-free inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import codec
from .data import (CONTRACT, examples_fingerprint, prepare_records, read_jsonl,
                   split_examples, write_jsonl)
from .predictor import PredictionError
from .teacher import TeacherCaptureError


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _write_report(path, report):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_json(report) + "\n", encoding="utf-8")


def _progress(value):
    brief = ({key: part for key, part in value.items()
              if key not in {"predicted_text", "token_ids"}} if isinstance(value, dict) else value)
    print(json.dumps(brief, ensure_ascii=False), file=sys.stderr, flush=True)


def _collect_calls(args):
    from .teacher import capture_call
    records = []
    for spec in read_jsonl(args.spec):
        record = capture_call(spec["module"], spec["hook"], spec["arguments"],
                              conversation_id=spec["conversation_id"],
                              seed=spec.get("seed", args.seed))
        records.append(record)
    write_jsonl(args.output, records)
    print(_json({"output": str(args.output), "records": len(records),
                 "contract": "isolated_seeded_call"}))


def _collect(args):
    from .teacher import capture_conversation
    records, summaries = [], []
    for spec in read_jsonl(args.scenarios):
        captured, summary = capture_conversation(
            spec["texts"], conversation_id=spec["conversation_id"],
            seed=spec.get("seed", args.seed), selected_hooks=args.hook)
        records.extend(captured)
        summaries.append(summary)
    write_jsonl(args.output, records)
    _write_report(str(args.output) + ".summary.json", {"conversations": summaries})
    print(_json({"output": str(args.output), "records": len(records)}))


def _train(args):
    from .model import ModelConfig
    from .training import fit, resolve_device, save_checkpoint

    wire_format = args.wire_format
    examples, coverage = prepare_records(
        read_jsonl(args.data),
        max_seq_len=args.max_seq_len,
        selected_hooks=args.hook,
        wire_format=wire_format,
    )
    if not examples:
        raise ValueError("no complete examples fit the model context; increase --max-seq-len")
    if args.validation_data:
        train = examples
        validation, val_coverage = prepare_records(
            read_jsonl(args.validation_data),
            selected_hooks=args.hook,
            wire_format=wire_format,
        )
        split = {
            "method": "explicit external validation file",
            "train_records": len(train),
            "validation_records": len(validation),
            "validation_coverage": val_coverage,
        }
    else:
        train, validation, split = split_examples(
            examples, validation_fraction=args.validation_fraction, seed=args.split_seed)
    device = resolve_device(args.device)
    if args.fp_baseline and (args.attention == "logic_proj" or args.ffn == "logic"):
        raise ValueError("--fp-baseline conflicts with logic --attention/--ffn")
    config = ModelConfig(
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        qk_bits=args.d_model,
        v_bits=args.d_model,
        logic=not args.fp_baseline,
        attention=args.attention,
        ffn=args.ffn,
        attention_score=args.attention_score,
        attention_norm=args.attention_norm,
        n_inputs=args.n_inputs,
        ffn_n_inputs=args.ffn_n_inputs,
        parameterization=args.parameterization,
        proj_depth=args.proj_depth,
        wiring=args.wiring,
        lm_tau=args.lm_tau,
        gumbel_noise=args.gumbel_noise,
        gumbel_tau=args.gumbel_tau,
        warp_gumbel=args.warp_gumbel,
        weight_tying=args.weight_tying,
        pos_init=args.pos_init,
        embed_init_std=args.embed_init_std,
        wire_format=wire_format,
    )
    model, training_log = fit(
        train,
        config,
        steps=args.steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        learning_rate=args.learning_rate,
        threads=args.threads,
        train_mode=args.train_mode,
        device=device,
        loss_weighting=args.loss_weighting,
        lr_schedule=args.lr_schedule,
        warmup_steps=args.warmup_steps,
        lut_lr_scale=args.lut_lr_scale,
        wire_lr_scale=args.wire_lr_scale,
        weight_decay=args.weight_decay,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        aux_lambda=args.aux_lambda,
        sampling=args.sampling,
        init_from=args.init_from,
        teacher_checkpoint=args.teacher_checkpoint,
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp,
        attn_kd_weight=args.attn_kd_weight,
        progress=_progress,
    )
    metadata = {"contract": CONTRACT,
                "wire_format": wire_format,
                "trained_hooks": sorted({e["module"] + "." + e["hook"] for e in train}),
                "training_fixtures": sorted({e["record"].get("metadata", {}).get("fixture", "unknown")
                                             for e in train}),
                "training_data_sha256": examples_fingerprint(train),
                "validation_data_sha256": examples_fingerprint(validation),
                "coverage": coverage, "split": split,
                "training_log": training_log,
                "griffith_revision": "e0675f4e36596954aa424d522e0ee966bba9a3e1",
                "scope": "return, full memory, mutable candidates, and exception; "
                         "isolated fixture RNG is external, not a learned global side effect",
                "source_commits": sorted({
                    e["record"].get("metadata", {}).get("source_commit")
                    or e["record"].get("metadata", {}).get("source", {}).get("git_commit", "unknown")
                    for e in train})}
    save_checkpoint(model, args.checkpoint, metadata=metadata)
    # Explicit split artifacts support independent evaluation without resplitting.
    write_jsonl(str(args.checkpoint) + ".train.jsonl.gz", [e["record"] for e in train])
    if not args.validation_data:
        write_jsonl(str(args.checkpoint) + ".validation.jsonl.gz", [e["record"] for e in validation])
    report_path = args.report or str(args.checkpoint) + ".training.json"
    _write_report(report_path, metadata)
    print(_json({"checkpoint": str(args.checkpoint), "report": report_path,
                 "train_records": len(train), "validation_records": len(validation),
                 "excluded_records": coverage["excluded_records"],
                 "device": device,
                 "wire_format": wire_format,
                 "note": "Training loss is not whole-call reproduction accuracy; run evaluate."}))


def _evaluate(args):
    import torch
    from .evaluation import evaluate, always_none_baseline
    from .training import load_checkpoint, resolve_device
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    wire_format = metadata.get("wire_format") or getattr(model.config, "wire_format", "legacy-v1")
    # Evaluation keeps overflowing examples in the denominator. Generation will
    # reject them explicitly; filtering them here would inflate reported accuracy.
    examples, coverage = prepare_records(
        read_jsonl(args.data),
        selected_hooks=args.hook,
        wire_format=wire_format,
    )
    total = len(examples)
    if args.limit is not None:
        examples = examples[:args.limit]
    report = evaluate(model, examples, mode=args.mode, max_new_tokens=args.max_new_tokens,
                      progress=_progress)
    report["always_none_baseline"] = always_none_baseline(examples)
    report["dataset"] = {"available_records": total, "evaluated_records": len(examples),
                         "limit": args.limit, "coverage": coverage,
                         "wire_format": wire_format,
                         "sha256": examples_fingerprint(examples),
                         "training_sha256": metadata.get("training_data_sha256"),
                         "evaluation_contract": "one call with recorded input memory; "
                                                "not an end-to-end learned conversation"}
    _write_report(args.output, report)
    print(_json({"output": str(args.output), "counts": report["counts"],
                 "rates": report["rates"], "latency_seconds": report["latency_seconds"],
                 "always_none_baseline_rates": report["always_none_baseline"]["rates"]}))


def _predict(args):
    from .predictor import load_predictor
    raw = Path(args.arguments).read_text(encoding="utf-8")
    arguments = codec.loads(raw) if args.typed_arguments else json.loads(raw)
    predictor = load_predictor(args.checkpoint)
    extra = {}
    if args.rng_state:
        extra["rng_state"] = codec.loads(Path(args.rng_state).read_text(encoding="utf-8"))
    output = predictor.predict_call(args.module, args.hook, arguments, seed=args.seed,
                                    environment_date=args.date,
                                    max_new_tokens=args.max_new_tokens, **extra)
    print(codec.dumps(output))


def _export_c(args):
    from .export_c import export_c_model
    report = export_c_model(
        args.checkpoint,
        args.output_dir,
        compile_shared=not args.no_compile,
    )
    print(_json(report))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="list all real PyBot module hooks")
    inventory.add_argument("--output")
    collect = commands.add_parser("collect", help="trace original conversations without changing routing")
    collect.add_argument("--scenarios", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--hook", action="append", help="module.hook; omit to trace every hook")
    collect.add_argument("--seed", type=int, default=0)
    calls = commands.add_parser("collect-calls", help="record independent complete module calls")
    calls.add_argument("--spec", required=True)
    calls.add_argument("--output", required=True)
    calls.add_argument("--seed", type=int, default=0)
    train = commands.add_parser("train", help="learn complete calls with official Griffith")
    train.add_argument("--data", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--report")
    train.add_argument("--hook", action="append")
    train.add_argument("--wire-format", choices=("legacy-v1", "compact-v2"), default="compact-v2")
    train.add_argument("--validation-data")
    train.add_argument("--max-seq-len", type=int, default=2048)
    train.add_argument("--d-model", type=int, default=32)
    train.add_argument("--n-heads", type=int, default=2)
    train.add_argument("--n-layers", type=int, default=1)
    train.add_argument("--mlp-ratio", type=int, default=2)
    train.add_argument("--attention", choices=("fp", "logic_proj"), default=None)
    train.add_argument("--ffn", choices=("fp", "logic"), default=None)
    train.add_argument("--attention-score", choices=("fp_dot", "xnor_popcount"), default="xnor_popcount")
    train.add_argument(
        "--attention-norm",
        choices=("softmax", "mux_gumbel", "softmax_int", "topk_ste", "threshold"),
        default="softmax",
    )
    train.add_argument("--n-inputs", type=int, choices=(2, 4, 6), default=2)
    train.add_argument("--ffn-n-inputs", type=int, choices=(2, 4, 6), default=2)
    train.add_argument("--parameterization", choices=("warp", "softmax", "independent"), default="warp")
    train.add_argument("--proj-depth", type=int, default=1)
    train.add_argument("--wiring", choices=("fixed", "learnable"), default="fixed")
    train.add_argument("--lm-tau", type=float, default=0.05)
    train.add_argument("--gumbel-noise", action="store_true")
    train.add_argument("--gumbel-tau", type=float, default=1.0)
    train.add_argument("--warp-gumbel", action="store_true")
    train.add_argument("--weight-tying", action="store_true")
    train.add_argument("--pos-init", choices=("default", "sinusoidal"), default="sinusoidal")
    train.add_argument("--embed-init-std", type=float, default=0.25)
    train.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--grad-accum", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--lut-lr-scale", type=float, default=1.0)
    train.add_argument("--wire-lr-scale", type=float, default=3.0)
    train.add_argument("--lr-schedule", choices=("wsd", "cosine", "constant"), default="wsd")
    train.add_argument("--warmup-steps", type=int, default=0)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--loss-weighting", choices=("example", "token"), default="example")
    train.add_argument("--temp-start", type=float, default=2.0)
    train.add_argument("--temp-end", type=float, default=0.5)
    train.add_argument("--aux-lambda", type=float, default=0.01)
    train.add_argument("--sampling", choices=("epoch_bucketed", "random"), default="epoch_bucketed")
    train.add_argument("--init-from")
    train.add_argument("--teacher-checkpoint")
    train.add_argument("--kd-alpha", type=float, default=0.5)
    train.add_argument("--kd-temp", type=float, default=2.0)
    train.add_argument("--attn-kd-weight", type=float, default=0.0)
    train.add_argument("--train-mode", choices=("soft", "train_soft", "ste", "soft_to_ste"), default="train_soft")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--split-seed", type=int, default=0)
    train.add_argument("--validation-fraction", type=float, default=0.2)
    train.add_argument("--threads", type=int, default=1)
    train.add_argument("--fp-baseline", action="store_true",
                       help="official Griffith FP ladder control, not the logic model")
    evaluate_parser = commands.add_parser("evaluate", help="free-generate full outputs and compare all fields")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--mode", choices=("soft", "hard"), default="hard")
    evaluate_parser.add_argument("--max-new-tokens", type=int)
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument("--hook", action="append")
    evaluate_parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    evaluate_parser.add_argument("--threads", type=int, default=1)
    predict = commands.add_parser("predict", help="run a learned call without importing the PyBot teacher")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--module", required=True)
    predict.add_argument("--hook", required=True, choices=("observe", "respond", "choose", "after"))
    predict.add_argument("--arguments", required=True, help="JSON file with text/memory and other hook arguments")
    predict.add_argument("--typed-arguments", action="store_true", help="decode arguments with moduleio.codec")
    predict.add_argument("--rng-state", help="typed JSON pre-call random state for a natural conversation model")
    predict.add_argument("--seed", type=int, default=0)
    predict.add_argument("--date")
    predict.add_argument("--max-new-tokens", type=int)
    export_c = commands.add_parser("export-c", help="compile a trained checkpoint to native C99")
    export_c.add_argument("--checkpoint", required=True)
    export_c.add_argument("--output-dir", required=True)
    export_c.add_argument("--no-compile", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "inventory":
            from .teacher import discover_hooks
            report = discover_hooks()
            if args.output:
                _write_report(args.output, report)
            print(_json(report))
        else:
            {"collect": _collect, "collect-calls": _collect_calls, "train": _train,
             "evaluate": _evaluate, "predict": _predict,
             "export-c": _export_c}[args.command](args)
    except (ImportError, ValueError, OSError, PredictionError, TeacherCaptureError) as exc:
        print(f"moduleio: {exc}", file=sys.stderr)
        return 2
    return 0
