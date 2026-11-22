"""Parity tests for native C99 export against PyTorch hard evaluation."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

HAS_RUNTIME = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("griffith_logic") is not None
    and shutil.which("cc") is not None
)


@unittest.skipUnless(HAS_RUNTIME, "requires torch, griffith_logic, and a C compiler")
class ExportCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def _fit_and_check_parity(self, cfg, *, binarize_tologic: bool = False):
        import torch
        from moduleio.export_c import CompiledCModel, export_c_model
        from moduleio.model import BOS, SEP, encode_text
        from moduleio.training import fit

        examples = [
            {"input_text": '{"a":1}', "output_text": '{"return":null,"memory":"=","exception":null}'},
            {"input_text": '{"b":2}', "output_text": '{"return":"ok","memory":"=","exception":null}'},
        ]
        model, _ = fit(
            examples, cfg, steps=5, batch_size=2, seed=7,
            train_mode="soft_to_ste", loss_weighting="example",
            lr_schedule="wsd", warmup_steps=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.pt"
            c_dir = Path(tmp) / "c_out"
            from moduleio.model import save_checkpoint
            save_checkpoint(model, ckpt, {"wire_format": cfg.wire_format})

            info = export_c_model(ckpt, c_dir, compile_shared=True)
            self.assertTrue(info["compiled"])
            self.assertTrue(Path(info["library_path"]).exists())

            tokens = [BOS] + encode_text('{"a":1}') + [SEP]
            with torch.no_grad():
                pt_logits = model(torch.tensor([tokens], dtype=torch.long), mode="hard")[0, -1].numpy()
            pt_gen = model.generate('{"a":1}', max_new_tokens=16, mode="hard")

            with CompiledCModel(c_dir) as cm:
                c_logits = cm.forward_last_logits(tokens, binarize_tologic=False)
                max_diff = float(np.max(np.abs(pt_logits - c_logits)))
                self.assertLess(max_diff, 1e-4)

                for use_kv in (True, False):
                    c_gen = cm.generate('{"a":1}', max_new_tokens=16, use_kv_cache=use_kv, profile=True)
                    self.assertEqual(c_gen["token_ids"], pt_gen["token_ids"])
                    self.assertEqual(c_gen["eos_generated"], pt_gen["eos_generated"])
                    self.assertEqual(c_gen["text"], pt_gen["text"])
                    self.assertGreater(c_gen["profile_ns"]["total_ns"], 0)

                if binarize_tologic and cfg.ffn == "logic":
                    for blk in model.network.blocks:
                        blk.ffn.encode.mode = "binarize_ste"
                    with torch.no_grad():
                        pt_bin_logits = model(torch.tensor([tokens], dtype=torch.long), mode="hard")[0, -1].numpy()
                    c_bin_logits = cm.forward_last_logits(tokens, binarize_tologic=True)
                    self.assertLess(float(np.max(np.abs(pt_bin_logits - c_bin_logits))), 1e-4)

    def test_c_export_matches_pytorch_for_n2_n4_n6_logic_and_fp(self):
        from moduleio.model import ModelConfig

        configs = [
            ModelConfig(
                max_seq_len=96, d_model=16, n_heads=2, n_layers=2, mlp_ratio=2,
                attention="fp", ffn="logic", ffn_n_inputs=2, wiring="learnable",
                lm_tau=0.05, wire_format="compact-v2",
            ),
            ModelConfig(
                max_seq_len=96, d_model=16, n_heads=2, n_layers=2, mlp_ratio=2,
                attention="fp", ffn="logic", ffn_n_inputs=4, wiring="fixed",
                wire_format="compact-v2",
            ),
            ModelConfig(
                max_seq_len=96, d_model=16, n_heads=2, n_layers=2, mlp_ratio=2,
                attention="fp", ffn="logic", ffn_n_inputs=6, wiring="learnable",
                lm_tau=0.05, wire_format="compact-v2",
            ),
            ModelConfig(
                max_seq_len=96, d_model=16, n_heads=2, n_layers=2, mlp_ratio=2,
                logic=False, wire_format="compact-v2",
            ),
        ]
        for idx, cfg in enumerate(configs):
            with self.subTest(idx=idx, ffn=cfg.ffn, ffn_n_inputs=cfg.ffn_n_inputs, wiring=cfg.wiring):
                self._fit_and_check_parity(cfg, binarize_tologic=(idx == 1))

    def test_export_c_cli_subcommand(self):
        from moduleio.cli import main
        from moduleio.model import ModelConfig, ModuleIOModel, save_checkpoint

        cfg = ModelConfig(
            max_seq_len=64, d_model=8, n_heads=2, n_layers=1, mlp_ratio=2,
            attention="fp", ffn="logic", ffn_n_inputs=4, wiring="fixed",
            wire_format="compact-v2",
        )
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "m.pt"
            out_dir = Path(tmp) / "exported"
            save_checkpoint(ModuleIOModel(cfg), ckpt, {"wire_format": "compact-v2"})
            rc = main(["export-c", "--checkpoint", str(ckpt), "--output-dir", str(out_dir)])
            self.assertEqual(rc, 0)
            meta = json.loads((out_dir / "export_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(meta["compiled"])
            self.assertTrue((out_dir / "weights.bin").exists())
            self.assertTrue((out_dir / "moduleio_runtime.c").exists())


if __name__ == "__main__":
    unittest.main()
