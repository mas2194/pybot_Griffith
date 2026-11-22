"""Learning contracts: full outputs, causal inputs, native LUTs and hard inference."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HAS_MODEL = (importlib.util.find_spec("torch") is not None and
             importlib.util.find_spec("griffith_logic") is not None)


@unittest.skipUnless(HAS_MODEL, "install the module-I/O extra (Griffith and torch)")
class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def model(self, length=96):
        import torch
        from moduleio.model import ModelConfig, ModuleIOModel
        torch.manual_seed(91)
        return ModuleIOModel(ModelConfig(max_seq_len=length, d_model=8,
                                        qk_bits=8, v_bits=8)).eval()

    def test_output_only_mask_covers_all_bytes_and_real_eos(self):
        from moduleio.model import EOS, IGNORE_INDEX, SEP, encode_text, prepare_example
        example = prepare_example('日本語', '{"return":null}', 96)
        self.assertEqual(example["input_ids"][len(example["prompt_ids"]) - 1], SEP)
        targets = [token for token in example["labels"] if token != IGNORE_INDEX]
        self.assertEqual(targets, encode_text('{"return":null}') + [EOS])
        self.assertEqual(len(example["input_ids"]), len(example["labels"]))
        self.assertEqual(example["answer_ids"], targets)
        empty = prepare_example("", "", 4)
        self.assertEqual(empty["labels"], [IGNORE_INDEX, EOS])

    def test_custom_model_width_resolves_attention_width_and_backpropagates(self):
        from torch.nn import functional as F
        from griffith_logic.layers.lut import LutLayer
        from moduleio.model import (IGNORE_INDEX, VOCAB_SIZE, ModelConfig,
                                    ModuleIOModel, prepare_example)
        from moduleio.training import collate
        config = ModelConfig(max_seq_len=32, d_model=64)
        self.assertEqual((config.qk_bits, config.v_bits), (64, 64))
        self.assertEqual(config, ModelConfig(max_seq_len=32, d_model=64,
                                            qk_bits=64, v_bits=64))
        with self.assertRaisesRegex(ValueError, "v_bits == d_model"):
            ModelConfig(d_model=64, v_bits=32)
        model = ModuleIOModel(config)
        batch = collate([prepare_example("in", "out", 32)])
        logits = model(batch["input_ids"], mode="soft")
        self.assertEqual(tuple(logits.shape), (*batch["input_ids"].shape, VOCAB_SIZE))
        mask = batch["labels"] != IGNORE_INDEX
        F.cross_entropy(logits[mask], batch["labels"][mask]).backward()
        layers = [part for part in model.modules() if isinstance(part, LutLayer)]
        self.assertTrue(any(part.weights.grad is not None and
                            part.weights.grad.abs().sum() > 0 for part in layers))

    def test_overlength_input_or_output_is_rejected_without_truncation(self):
        from moduleio.model import ExampleTooLongError, prepare_example
        for prompt, answer in (("x" * 100, "{}"), ("{}", "y" * 100)):
            with self.subTest(prompt=prompt), self.assertRaises(ExampleTooLongError):
                prepare_example(prompt, answer, 32)
        model = self.model(length=32)
        with self.assertRaises(ExampleTooLongError):
            model.generate("x" * 32, max_new_tokens=1)
        with self.assertRaises(ExampleTooLongError):
            model.generate("x", max_new_tokens=32)

    def test_labels_cannot_enter_prompt_and_future_output_is_causally_masked(self):
        import torch
        from moduleio.model import prepare_example
        model = self.model()
        a = prepare_example('{"x":1}', 'alpha', 96)
        b = prepare_example('{"x":1}', 'zebra', 96)
        self.assertEqual(a["prompt_ids"], b["prompt_ids"])
        boundary = len(a["prompt_ids"])
        for mode in ("soft", "hard"):
            one = model(torch.tensor([a["input_ids"]]), mode=mode)
            two = model(torch.tensor([b["input_ids"]]), mode=mode)
            self.assertTrue(torch.allclose(one[:, :boundary], two[:, :boundary], atol=1e-6))

    def test_native_griffith_parameters_receive_gradients_and_modes_restore(self):
        import torch
        from torch.nn import functional as F
        from griffith_logic.layers.lut import LutLayer
        from griffith_logic.models import LogicLadderGPT
        from moduleio.model import IGNORE_INDEX, prepare_example
        from moduleio.training import collate
        model = self.model()
        self.assertIsInstance(model.network, LogicLadderGPT)
        layers = [layer for layer in model.modules() if isinstance(layer, LutLayer)]
        self.assertTrue(layers)
        batch = collate([prepare_example("a", "b", 96), prepare_example("c", "d", 96)])
        settings = [(part.training, getattr(part, "gumbel", None),
                     getattr(part, "selection", None)) for part in model.modules()]
        mask = batch["labels"] != IGNORE_INDEX
        for mode in ("soft", "ste"):
            model.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], mode=mode)
            F.cross_entropy(logits[mask], batch["labels"][mask]).backward()
            self.assertTrue(any(layer.weights.grad is not None and
                                torch.count_nonzero(layer.weights.grad) for layer in layers))
            self.assertEqual(settings, [(part.training, getattr(part, "gumbel", None),
                getattr(part, "selection", None)) for part in model.modules()])

    def test_soft_and_hard_generation_are_deterministic_and_report_stopping(self):
        model = self.model()
        for mode in ("soft", "hard"):
            a = model.generate("{}", max_new_tokens=5, mode=mode)
            b = model.generate("{}", max_new_tokens=5, mode=mode)
            self.assertEqual(a, b)
            self.assertIn(a["stop_reason"], ("eos", "max_new_tokens"))
            self.assertLessEqual(a["generated_tokens"], 5)
        empty = model.generate("{}", max_new_tokens=0)
        self.assertEqual(empty["text"], "")
        self.assertFalse(empty["eos_generated"])
        self.assertEqual(empty["stop_reason"], "max_new_tokens")

    def test_hard_checkpoint_roundtrip_works_with_teacher_imports_blocked(self):
        from moduleio.model import load_checkpoint, save_checkpoint
        model = self.model()
        expected = model.generate("{}", max_new_tokens=4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            save_checkpoint(model, path, {"purpose": "contract test"})
            restored, metadata = load_checkpoint(path)
            self.assertEqual(restored.generate("{}", max_new_tokens=4), expected)
            self.assertEqual(metadata, {"purpose": "contract test"})
            script = '''
import importlib.abc
import json
import sys
class BlockTeacher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bot" or fullname.startswith("bot."):
            raise AssertionError("teacher import attempted: " + fullname)
sys.meta_path.insert(0, BlockTeacher())
import torch
torch.set_num_threads(1)
from moduleio.model import load_checkpoint
model, _ = load_checkpoint(sys.argv[1])
print(json.dumps(model.generate("{}", max_new_tokens=4)))
assert not any(name == "bot" or name.startswith("bot.") for name in sys.modules)
'''
            result = subprocess.run([sys.executable, "-c", script, str(path)],
                                    cwd=Path(__file__).resolve().parents[2],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), expected)

    def test_tiny_fit_uses_all_output_tokens_and_reports_teacher_forcing(self):
        from moduleio.model import ModelConfig
        from moduleio.training import fit, teacher_forced_metrics
        examples = [{"input_text": "x", "output_text": "{}"}]
        config = ModelConfig(max_seq_len=24, d_model=8, qk_bits=8, v_bits=8)
        model, report = fit(examples, config, steps=3, batch_size=1)
        self.assertEqual(report["truncated_examples"], 0)
        self.assertEqual(report["max_output_tokens_including_eos"], 3)
        metrics = teacher_forced_metrics(model, examples)
        self.assertEqual(metrics["output_tokens_including_eos"], 3)
        self.assertIn("not free generation", metrics["conditioning"])

    def test_l2_and_l3c_griffith_configs_train_and_generate_deterministically(self):
        import torch
        from moduleio.model import ModelConfig, ModuleIOModel, load_checkpoint, save_checkpoint
        from moduleio.training import fit

        examples = [
            {"input_text": '{"a":1}', "output_text": '{"return":null,"memory":"=","exception":null}'},
            {"input_text": '{"b":2}', "output_text": '{"return":"ok","memory":"=","exception":null}'},
        ]
        configs = [
            ModelConfig(
                max_seq_len=96, d_model=16, n_heads=2, n_layers=2, mlp_ratio=2,
                attention_score="xnor_popcount", attention_norm="softmax",
                n_inputs=2, parameterization="softmax", proj_depth=1,
                gumbel_noise=True, gumbel_tau=1.0, pos_init="sinusoidal",
                embed_init_std=0.25, wire_format="compact-v2",
            ),
            ModelConfig(
                max_seq_len=96, d_model=16, n_heads=2, n_layers=2, mlp_ratio=2,
                attention_score="xnor_popcount", attention_norm="mux_gumbel",
                n_inputs=2, parameterization="warp", proj_depth=1,
                weight_tying=True, pos_init="sinusoidal", embed_init_std=0.25,
                wire_format="compact-v2",
            ),
        ]
        for cfg in configs:
            mode = "train_soft"
            model, report = fit(examples, cfg, steps=4, batch_size=2, train_mode=mode,
                                loss_weighting="example", lr_schedule="wsd", warmup_steps=1)
            self.assertTrue(report["curve"])
            self.assertEqual(model.config.wire_format, "compact-v2")
            out_hard1 = model.generate('{"a":1}', max_new_tokens=8, mode="hard")
            out_hard2 = model.generate('{"a":1}', max_new_tokens=8, mode="hard")
            self.assertEqual(out_hard1, out_hard2)
            out_soft1 = model.generate('{"a":1}', max_new_tokens=8, mode="soft")
            out_soft2 = model.generate('{"a":1}', max_new_tokens=8, mode="soft")
            if cfg.attention_norm == "softmax":
                self.assertEqual(out_soft1, out_soft2)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "ckpt.pt"
                save_checkpoint(model, path, {"wire_format": cfg.wire_format, "trained_hooks": ["a.respond"]})
                restored, _ = load_checkpoint(path)
                self.assertEqual(
                    restored.generate('{"a":1}', max_new_tokens=8, mode="hard"),
                    model.generate('{"a":1}', max_new_tokens=8, mode="hard"),
                )

    def test_hybrid_attention_and_ffn_options_train_and_roundtrip(self):
        from griffith_logic.layers.lut import LutLayer
        from moduleio.model import ModelConfig, ModuleIOModel, load_checkpoint, save_checkpoint
        from moduleio.training import fit

        self.assertEqual((ModelConfig(logic=True).attention, ModelConfig(logic=True).ffn), ("logic_proj", "logic"))
        self.assertEqual((ModelConfig(logic=False).attention, ModelConfig(logic=False).ffn), ("fp", "fp"))
        with self.assertRaisesRegex(ValueError, "attention"):
            ModelConfig(attention="invalid")
        with self.assertRaisesRegex(ValueError, "ffn"):
            ModelConfig(ffn="invalid")

        examples = [
            {"input_text": '{"a":1}', "output_text": '{"return":null,"memory":"=","exception":null}'},
            {"input_text": '{"b":2}', "output_text": '{"return":"ok","memory":"=","exception":null}'},
        ]
        cases = [
            (
                ModelConfig(
                    max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                    attention="fp", ffn="logic", wire_format="compact-v2",
                ),
                2,  # 2-layer LogicMLP WARP FFN
            ),
            (
                ModelConfig(
                    max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                    attention="logic_proj", ffn="fp",
                    attention_score="xnor_popcount", attention_norm="softmax",
                    parameterization="warp", n_inputs=2, proj_depth=1,
                    wire_format="compact-v2",
                ),
                3,  # q_proj, k_proj, v_proj in attention
            ),
        ]
        for cfg, expected_luts in cases:
            self.assertTrue(cfg.logic)
            raw_model = ModuleIOModel(cfg)
            lut_layers = [m for m in raw_model.modules() if isinstance(m, LutLayer)]
            self.assertEqual(len(lut_layers), expected_luts)
            model, report = fit(
                examples, cfg, steps=5, batch_size=2, train_mode="soft_to_ste",
                loss_weighting="example", lr_schedule="wsd", warmup_steps=1,
            )
            self.assertEqual(report["curve"][-1]["mode"], "ste")
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "hybrid.pt"
                save_checkpoint(model, path, {"wire_format": cfg.wire_format})
                restored, _ = load_checkpoint(path)
                self.assertEqual(
                    restored.generate('{"a":1}', max_new_tokens=8, mode="hard"),
                    model.generate('{"a":1}', max_new_tokens=8, mode="hard"),
                )

    def test_learnable_wiring_trains_discretizes_and_roundtrips(self):
        import torch
        from griffith_logic.layers.dwn import LearnableMapping
        from moduleio.model import ModelConfig, ModuleIOModel, load_checkpoint, save_checkpoint
        from moduleio.training import fit

        with self.assertRaisesRegex(ValueError, "wiring"):
            ModelConfig(wiring="invalid")
        with self.assertRaisesRegex(ValueError, "lm_tau"):
            ModelConfig(lm_tau=0.0)

        examples = [
            {"input_text": '{"a":1}', "output_text": '{"return":null,"memory":"=","exception":null}'},
            {"input_text": '{"b":2}', "output_text": '{"return":"ok","memory":"=","exception":null}'},
        ]
        cfg = ModelConfig(
            max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
            attention="logic_proj", ffn="fp",
            attention_score="xnor_popcount", attention_norm="softmax",
            parameterization="warp", n_inputs=2, proj_depth=1,
            wiring="learnable", lm_tau=0.05, wire_format="compact-v2",
        )
        torch.manual_seed(0)
        raw_model = ModuleIOModel(cfg)
        lms = [m for m in raw_model.modules() if isinstance(m, LearnableMapping)]
        self.assertEqual(len(lms), 3)  # q_proj, k_proj, v_proj
        init_lms = [m.weights.detach().clone() for m in raw_model.modules() if isinstance(m, LearnableMapping)]
        model, report = fit(
            examples, cfg, steps=6, batch_size=2, seed=0, train_mode="soft_to_ste",
            loss_weighting="example", lr_schedule="wsd", warmup_steps=1,
            learning_rate=0.01, lut_lr_scale=1.5, wire_lr_scale=3.0,
        )
        self.assertEqual(report["training"]["wire_lr_scale"], 3.0)
        self.assertIn("wiring=learnable", report["implementation"])
        for m in model.modules():
            if isinstance(m, LearnableMapping):
                self.assertEqual(m.forward_mode, "hard")
        # Confirm learnable wiring weights actually moved from uniform U[0,1)
        trained_lms = [m.weights.detach().clone() for m in model.modules() if isinstance(m, LearnableMapping)]
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(init_lms, trained_lms)))
        h1 = model.generate('{"a":1}', max_new_tokens=8, mode="hard")
        h2 = model.generate('{"a":1}', max_new_tokens=8, mode="hard")
        self.assertEqual(h1, h2)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "learnable.pt"
            save_checkpoint(model, path, {"wire_format": cfg.wire_format})
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.generate('{"a":1}', max_new_tokens=8, mode="hard"), h1)

    def test_griffith_kd_and_init_from_distills_without_teacher_imports(self):
        from moduleio.model import ModelConfig, load_checkpoint, save_checkpoint
        from moduleio.training import fit

        examples = [
            {"input_text": '{"a":1}', "output_text": '{"return":null,"memory":"=","exception":null}'},
            {"input_text": '{"b":2}', "output_text": '{"return":"ok","memory":"=","exception":null}'},
        ]
        teacher_cfg = ModelConfig(
            max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
            logic=False, wire_format="compact-v2",
        )
        teacher, _ = fit(examples, teacher_cfg, steps=4, batch_size=2, seed=0)

        student_cfg = ModelConfig(
            max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
            attention="fp", ffn="logic", wiring="learnable", lm_tau=0.05,
            wire_format="compact-v2",
        )
        with self.assertRaisesRegex(ValueError, "attn_kd_weight requires teacher_checkpoint"):
            fit(examples, student_cfg, steps=2, batch_size=2, attn_kd_weight=0.1)

        with tempfile.TemporaryDirectory() as temporary:
            teacher_path = Path(temporary) / "teacher.pt"
            student_path = Path(temporary) / "student.pt"
            save_checkpoint(teacher, teacher_path, {"wire_format": "compact-v2"})

            student, report = fit(
                examples,
                student_cfg,
                steps=6,
                batch_size=2,
                seed=0,
                train_mode="soft_to_ste",
                learning_rate=0.005,
                lut_lr_scale=2.0,
                wire_lr_scale=3.0,
                init_from=teacher_path,
                teacher_checkpoint=teacher_path,
                kd_alpha=0.5,
                kd_temp=2.0,
                attn_kd_weight=0.1,
            )
            self.assertEqual(report["training"]["kd_alpha"], 0.5)
            self.assertEqual(report["training"]["kd_temp"], 2.0)
            self.assertEqual(report["training"]["attn_kd_weight"], 0.1)
            self.assertEqual(report["training"]["init_from"], str(teacher_path))
            self.assertEqual(report["training"]["teacher_checkpoint"], str(teacher_path))
            save_checkpoint(student, student_path, {"wire_format": "compact-v2"})
            teacher_path.unlink()

            expected = student.generate('{"a":1}', max_new_tokens=8, mode="hard")
            restored, _ = load_checkpoint(student_path)
            self.assertEqual(restored.generate('{"a":1}', max_new_tokens=8, mode="hard"), expected)

            script = '''
import importlib.abc
import json
import sys
class BlockTeacher(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bot" or fullname.startswith("bot."):
            raise AssertionError("teacher import attempted: " + fullname)
sys.meta_path.insert(0, BlockTeacher())
import torch
torch.set_num_threads(1)
from moduleio.model import load_checkpoint
model, _ = load_checkpoint(sys.argv[1])
print(json.dumps(model.generate('{"a":1}', max_new_tokens=8, mode="hard")))
'''
            result = subprocess.run([sys.executable, "-c", script, str(student_path)],
                                    cwd=Path(__file__).resolve().parents[2],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout), expected)

    def test_ffn_and_attention_n_inputs_4_and_6_train_and_roundtrip(self):
        import torch
        from griffith_logic.layers.lut import LutLayer
        from moduleio.model import ModelConfig, ModuleIOModel, load_checkpoint, save_checkpoint
        from moduleio.training import fit

        with self.assertRaisesRegex(ValueError, "ffn_n_inputs"):
            ModelConfig(ffn_n_inputs=3)
        examples = [
            {"input_text": '{"a":1}', "output_text": '{"return":null,"memory":"=","exception":null}'},
            {"input_text": '{"b":2}', "output_text": '{"return":"ok","memory":"=","exception":null}'},
        ]
        with self.assertRaisesRegex(ValueError, "grad_accum"):
            fit(examples, ModelConfig(max_seq_len=96, d_model=16), steps=1, grad_accum=0)

        cases = [
            (
                ModelConfig(
                    max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                    attention="fp", ffn="logic", ffn_n_inputs=4, wiring="fixed",
                    wire_format="compact-v2",
                ),
                4,
            ),
            (
                ModelConfig(
                    max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                    attention="fp", ffn="logic", ffn_n_inputs=6, wiring="learnable",
                    lm_tau=0.05, wire_format="compact-v2",
                ),
                6,
            ),
            (
                ModelConfig(
                    max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                    attention="logic_proj", ffn="fp", attention_score="xnor_popcount",
                    attention_norm="softmax", n_inputs=4, parameterization="warp",
                    proj_depth=1, wiring="learnable", lm_tau=0.05,
                    wire_format="compact-v2",
                ),
                4,
            ),
            (
                ModelConfig(
                    max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
                    attention="logic_proj", ffn="fp", attention_score="xnor_popcount",
                    attention_norm="softmax", n_inputs=6, parameterization="warp",
                    proj_depth=1, wiring="learnable", lm_tau=0.05,
                    wire_format="compact-v2",
                ),
                6,
            ),
        ]
        for cfg, expected_n in cases:
            raw = ModuleIOModel(cfg)
            lut_layers = [m for m in raw.modules() if isinstance(m, LutLayer)]
            self.assertTrue(lut_layers)
            for layer in lut_layers:
                self.assertEqual(layer.n_inputs, expected_n)
                self.assertEqual(layer.weights.shape[1], 2 ** expected_n)
            model, report = fit(
                examples, cfg, steps=4, batch_size=1, grad_accum=2,
                train_mode="soft_to_ste", loss_weighting="example",
                lr_schedule="wsd", warmup_steps=1,
            )
            self.assertEqual(report["training"]["grad_accum"], 2)
            out1 = model.generate('{"a":1}', max_new_tokens=8, mode="hard")
            out2 = model.generate('{"a":1}', max_new_tokens=8, mode="hard")
            self.assertEqual(out1, out2)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "n_inputs.pt"
                save_checkpoint(model, path, {"wire_format": cfg.wire_format})
                restored, _ = load_checkpoint(path)
                self.assertEqual(restored.generate('{"a":1}', max_new_tokens=8, mode="hard"), out1)

        # Verify legacy checkpoints without ffn_n_inputs in model_config load cleanly.
        legacy_cfg = ModelConfig(
            max_seq_len=96, d_model=16, n_heads=2, n_layers=1, mlp_ratio=2,
            attention="fp", ffn="logic", wiring="fixed", wire_format="compact-v2",
        )
        legacy_model = ModuleIOModel(legacy_cfg).eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            save_checkpoint(legacy_model, path, {"wire_format": "compact-v2"})
            payload = torch.load(path, map_location="cpu", weights_only=True)
            del payload["model_config"]["ffn_n_inputs"]
            torch.save(payload, path)
            restored, _ = load_checkpoint(path)
            self.assertEqual(restored.config.ffn_n_inputs, 2)
            self.assertEqual(
                restored.generate('{"a":1}', max_new_tokens=8, mode="hard"),
                legacy_model.generate('{"a":1}', max_new_tokens=8, mode="hard"),
            )


if __name__ == "__main__":
    unittest.main()
