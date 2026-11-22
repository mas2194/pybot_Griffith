# Complete module I/O: measured experiment

Using the official [Griffith](https://github.com/tsukushi-gate/Griffith) library
(commit `e0675f4e36596954aa424d522e0ee966bba9a3e1`), complete input-only held-out module I/O
reproduction improved from **0/18**
(both old LUT and old FP baseline) to **15/18**
in discretized **hard** LUT evaluation (**16/18** soft),
surpassing the upgraded FP control (**14/18**) on the
untouched 18-call benchmark.
On the strictly leak-free **318-call diverse unseen-entity validation split**,
switching Griffith `LogicMLP` FFNs from **2-input** to **4-input WARP LUTs** (`ffn_n_inputs=4`,
`model.pt` / `model-logic-ffn-kd-fixed-n4.pt`, `497,160` params) raised
promoted hard exact match from **201/318 (63.2%)** (`N=2`, `Learnable`, `1,259,016` params)
to **214/318 (67.3%)**
(`ok 311/318`, `N=4 Learnable` reaches **211/318**),
while the 4,096-LUT `N=2` fixed-wiring variant (`4L/mlp4`) reaches
**223/318 (70.1%)**
(`13/18` Orig 18), compared to **38/318 (11.9%)**
for Always-None and **247/318 (77.7%)**
for the FP control.

## What was broken in the 0/18 baseline and how it was fixed

1. **Attention collapse & under-capacity**: The initial `d_model=64, n_layers=1` model
   with `N(0, 1)` position embeddings and FP-dot on binary vectors failed to form
   entity-copying induction heads (`0/18` for both LUT and FP). Upgrading to `d_model=128,
   n_layers=4, n_heads=4` with sinusoidal + scaled (`0.25`) embedding init and WSD LR
   scheduling raised FP control from **0/18 -> 14/18**
   (**247/318** on 318 diverse calls).
2. **Verbose JSON wire bloat**: Replaced `legacy-v1` typed JSON dumps with lossless
   `compact-v2` ([wire.py](../../moduleio/wire.py)), using `"="` unchanged-memory sentinels
   and placing `environment` before `arguments` (cutting max original-18 sequence length
   from 890 to 336 tokens).
3. **Training curriculum & soft-hard gap**: Switched from step-1 `ste` to WARP soft
   relaxation with late `soft_to_ste` (`0.8 * steps`) + `example` loss weighting,
   nearly eliminating the soft-to-hard discretization drop (`8/18` soft == `8/18` hard scratch;
   `15/18` soft == `15/18` hard at `N=2`; `16/18` soft -> `15/18` hard at `N=4`).
4. **Saddle-point escape via official Griffith KD (`kd_kl_loss_seq`)**: Warm-starting
   non-LUT tensors from `fp-control.pt` (trained strictly on `diverse-train.jsonl.gz`) and
   distilling with Griffith's `kd_kl_loss_seq` + `split_param_groups` LR separation (`10×`
   on `LutLayer`) enabled exact induction-head byte copying through discretized Boolean LUTs
   (**8/18 -> 15/18**, **117/318 -> 214/318** at `N=4 Fixed` / **223/318** at `N=2 4096-LUT`).
5. **2-input vs 4-input vs 6-input LUT fan-in (`n_inputs ∈ {2, 4, 6}`)**: Exposed Griffith's
   `LogicMLP` `ffn_n_inputs` alongside attention `n_inputs`. Upgrading `LogicMLP` FFN from
   `N=2` (4 WH coeffs/LUT) to **`N=4`** (16 WH coeffs/LUT, fan-in `16` across 2 FFN layers)
   improved both `4L/mlp2 Learnable` (**201/318 -> 211/318**) and **`4L/mlp2 Fixed`**
   (**214/318, 497,160 params — 60% smaller than `N=2 Learnable`**). By contrast,
   **`N=6`** (64 WH coeffs/LUT) hit the `O(2^N)` compute/memory wall (~3× slower CPU eval,
   `2.95M` params with `LearnableMapping`, reduced batch `4×2` and `1800` steps) and dropped
   under STE switching (**136/318**, `13/18` Orig 18).
6. **Diverse leak-free corpus**: Built 3,681 real PyBot teacher calls (`3,363` train /
   `318` diverse val) covering unseen colors, favourites categories, age groups, and names
   with programmatic zero-overlap checks against the untouched 18-call validation file.

## 2-input vs 4-input vs 6-input Griffith LUT comparison

| Family | LUT Arity (`2^N`) | Steps (`b×acc=eff`) | Params | MPS train (s / step) | Orig 18 hard / soft | Diverse 318 hard | CPU eval mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| FFN KD (Learnable, 4L/mlp2) | N=2 (2^2=4) | 2400 (16×1=16) | 1,259,016 | 590.3s (246 ms/st) | 15/18 / 15/18 | 201/318 (63.2%) | 1.135s |
| FFN KD (Fixed, 4L/mlp2) | N=4 (2^4=16) | 2400 (8×2=16) | 497,160 | 1228.5s (512 ms/st) | 15/18 / 16/18 | 214/318 (67.3%) | 1.329s |
| FFN KD (Learnable, 4L/mlp2) | N=4 (2^4=16) | 2400 (8×2=16) | 2,070,024 | 1244.4s (518 ms/st) | 15/18 / 15/18 | 211/318 (66.4%) | 1.436s |
| FFN KD (Learnable, 4L/mlp2) | N=6 (2^6=64) | 1800 (4×2=8) | 2,954,760 | 1348.5s (749 ms/st) | 13/18 / 13/18 | 136/318 (42.8%) | 3.170s |
| Attn L2 (Learnable, 4L/mlp4) | N=2 (2^2=4) | 4800 (16×1=16) | 1,126,408 | 1218.7s (254 ms/st) | 4/18 / 4/18 | 36/318 (11.3%) | 1.345s |
| Attn L2 (Learnable, 4L/mlp4) | N=4 (2^4=16) | 3000 (8×2=16) | 1,538,056 | 1360.3s (453 ms/st) | 4/18 / 4/18 | 34/318 (10.7%) | 1.687s |
| Attn L2 (Learnable, 4L/mlp4) | N=6 (2^6=64) | 2000 (4×2=8) | 2,005,000 | 1169.1s (585 ms/st) | 4/18 / 4/18 | 23/318 (7.2%) | 2.905s |

- **Sweet spot (`N=4`) for FFN `LogicMLP`**: `N=4` raises single-LUT truth tables from 4 to 16 entries without needing `LearnableMapping`'s massive softmax routing tables (`497,160` params total vs `1,259,016` for `N=2 Learnable` and `2,070,024` for `N=4 Learnable`), reaching **15/18** Orig 18 and **214/318 (67.3%)** Diverse 318.
- **Why `N=6` degraded here**: With `2^6 = 64` Walsh-Hadamard parameters per neuron (`16×` more than `N=2`) and 6× `LearnableMapping` tables (`2,954,760` params), MPS memory required halving effective batch (`4×2=8`) and fewer steps (`1,800`), while STE discretization after step `1,440` suffered a larger drop (teacher-forced sequence accuracy fell to `50.8%` hard vs `85.7%` for `N=4 Fixed`, CPU generation slowed `2.4×` to `3.170s/call`).
- **Why `N=4` / `N=6` alone did not lift `L2-LogicAttn`**: Across `N=2 -> 4 -> 6`, from-scratch `L2-LogicAttn` stayed flat at `4/18` Orig 18 (`36/318 -> 34/318 -> 23/318` Diverse 318). In `LadderAttention`, `Binarize(threshold=0.0)` quantizes hidden states to 1 bit/channel before the LUTs and `xnor_popcount` compares 32 binary bits per head, so widening Q/K/V truth tables alone cannot bypass that front-end binarization and saddle point without working Attention KD.

## Input-only free generation across all rungs

All models are evaluated with input-only prompts (no teacher imports, no answer lookup,
no prefix hints) and full semantic equivalence (return value, complete updated memory,
and exception descriptor).

| Model | Params | MPS train | Orig 18 hard | Orig 18 soft | Orig 18 valid JSON | Diverse 318 hard | Diverse 318 valid | Diverse 318 active |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous 0/18 LUT baseline | — | — | 0/18 | 0/18 | 0/18 | — | — | — |
| Previous 0/18 FP baseline | — | — | 0/18 | — | 5/18 | — | — | — |
| Always None + unchanged memory | 0 | — | 4/18 | — | 18/18 | 38/318 (11.9%) | 318/318 | 0/280 |
| FP Control (L0, fp/fp, 4L/mlp4) | 991,236 | 276.1s | 14/18 | — | 18/18 | 247/318 (77.7%) | 312/318 | 209/280 |
| Promoted: L0+LogicFFN-KD-Fixed (N=4, 4L/mlp2, wiring=fixed) | 497,160 | 1228.5s | 15/18 | 16/18 | 18/18 | 214/318 (67.3%) | 311/318 | 176/280 |
| L0+LogicFFN-KD-Learnable (N=4, 4L/mlp2, wiring=learnable) | 2,070,024 | 1244.4s | 15/18 | 15/18 | 18/18 | 211/318 (66.4%) | 309/318 | 173/280 |
| L0+LogicFFN-KD-4096 (N=2, 4L/mlp4, wiring=fixed) | 480,776 | 934.8s | 13/18 | 14/18 | 17/18 | 223/318 (70.1%) | 305/318 | 185/280 |
| L0+LogicFFN-KD-Learnable (N=2, 4L/mlp2, wiring=learnable) | 1,259,016 | 590.3s | 15/18 | 15/18 | 18/18 | 201/318 (63.2%) | 307/318 | 163/280 |
| L0+LogicFFN-KD-Learnable (N=6, 4L/mlp2, wiring=learnable) | 2,954,760 | 1348.5s | 13/18 | 13/18 | 18/18 | 136/318 (42.8%) | 283/318 | 99/280 |
| L0+LogicFFN (from-scratch, N=2, 4L/mlp4, wiring=fixed) | 480,776 | 1484.7s | 8/18 | 8/18 | 18/18 | 117/318 (36.8%) | 305/318 | 79/280 |
| L2-LogicAttn (from-scratch, N=2, xnor_popcount+softmax, wiring=learnable) | 1,126,408 | 1218.7s | 4/18 | 4/18 | 18/18 | 36/318 (11.3%) | 281/318 | 1/280 |
| L2-LogicAttn (from-scratch, N=4, xnor_popcount+softmax, wiring=learnable) | 1,538,056 | 1360.3s | 4/18 | 4/18 | 12/18 | 34/318 (10.7%) | 237/318 | 0/280 |
| L2-LogicAttn (from-scratch, N=6, xnor_popcount+softmax, wiring=learnable) | 2,005,000 | 1169.1s | 4/18 | 4/18 | 10/18 | 23/318 (7.2%) | 143/318 | 0/280 |
| L2-LogicAttn-KD (AttnKL, N=2, xnor_popcount+softmax, wiring=learnable) | 1,126,408 | 1035.3s | 0/18 | 0/18 | 0/18 | 2/318 (0.6%) | 30/318 | 0/280 |
| L3-C-LogicAttn (from-scratch, N=2, xnor_popcount+mux_gumbel, wiring=learnable) | 1,126,408 | 1840.3s | 0/18 | 0/18 | 5/18 | 12/318 (3.8%) | 80/318 | 0/280 |
| L2-Full-KD (Attn+FFN logic, N=2, xnor_popcount+softmax, wiring=fixed) | 214,540 | 2176.8s | 0/18 | 0/18 | 0/18 | 0/318 (0.0%) | 0/318 | 0/280 |

## Promoted checkpoint breakdown (`experiments/moduleio/model.pt`)

Promoted checkpoint: `experiments/moduleio/model-logic-ffn-kd-fixed-n4.pt`
(`d_model=128, n_heads=4, n_layers=4, mlp_ratio=2, attention='fp', ffn='logic', ffn_n_inputs=4, wiring='fixed'`).

- **Discretized Griffith logic components**: 8 WARP **4-input** `LutLayer` layers
  (**2,048 LUT neurons** × 16 truth-table entries = 32,768 entries across 4 Transformer blocks,
  `wiring='fixed'`, evaluated deterministically with hard truth tables at inference).
- **Remaining FP components**: `token_embed`, `pos_embed`, `nn.MultiheadAttention` (`L0`),
  `LayerNorm`, and output `head` (`nn.Linear`), plus continuous `ToLogic('sigmoid')` /
  `FromLogic('groupsum_centered')` bridges around each LUT block.
- **CPU eval latency (`threads=8`)**: mean `1.233s`
  (median `1.474s`) on Original 18;
  mean `1.329s`
  (median `1.415s`) on Diverse 318
  (vs `0.311s` mean for FP Control).

### By activity

| Split | Activity | Promoted Hard exact | FP Control exact | Always-None exact |
|---|---|---:|---:|---:|
| Original 18 | Active (state change or reply) | 11/14 | 10/14 | 0/14 |
| Original 18 | Inactive (`None`, unchanged) | 4/4 | 4/4 | 4/4 |
| Diverse 318 | Active (state change or reply) | 176/280 (62.9%) | 209/280 (74.6%) | 0/280 |
| Diverse 318 | Inactive (`None`, unchanged) | 38/38 (100.0%) | 38/38 (100.0%) | 38/38 |

### By module and hook (Promoted hard model)

| Module | Orig 18 exact | Diverse 318 exact | Diverse 318 Return | Diverse 318 Memory | Diverse 318 FP Control |
|---|---:|---:|---:|---:|---:|
| `a_cake_between_its_layers` | 0/0 (train-only) | 2/2 (100.0%) | 2/2 | 2/2 | 2/2 (100.0%) |
| `favourites` | 5/8 | 72/122 (59.0%) | 98/122 | 95/122 | 83/122 (68.0%) |
| `group_ages` | 4/4 | 88/116 (75.9%) | 91/116 | 111/116 | 101/116 (87.1%) |
| `name_taken_back` | 6/6 | 52/78 (66.7%) | 73/78 | 52/78 | 61/78 (78.2%) |

| Hook | Orig 18 exact | Diverse 318 exact | Diverse 318 Return | Diverse 318 Memory | Diverse 318 Valid JSON |
|---|---:|---:|---:|---:|---:|
| `observe` | 10/12 | 139/196 (70.9%) | 189/196 | 139/196 | 189/196 |
| `respond` | 5/6 | 75/122 (61.5%) | 75/122 | 121/122 | 122/122 |

On the untouched 18-call benchmark, `model.pt` scores **18/18 valid schema JSON**,
**17/18 return_exact**, **16/18 memory_exact**, **4/4 `group_ages`**, and **6/6 `name_taken_back`**;
all 3 remaining errors (`indices 0, 1, 2`) are minor character slips on the unseen colour
`"purple"` (`"purrcele"` / `"purrcrupc"` / `"purqe."`), while preserving the `"colour"` vs
`"color"` key distinction and nested memory schema.

## Why full-attention LUT rungs trailed and AP法 compatibility

- **Logic-Attention & Full-Logic blocks (`L2-LogicAttn`, `L3-C`, `L2-Full`)**:
  XNOR-popcount on 1-layer LUT Q/K/V projections (`proj_depth=1`) showed almost no gain from
  increasing LUT arity alone (`N=2`: `4/18` & `36/318`, `N=4`: `4/18` & `34/318`,
  `N=6`: `4/18` & `23/318`), while replacing softmax with 1-token `mux_gumbel` (`L3-C`) or
  placing fixed sparse LUTs in both sublayers (`L2-Full`) prevented accurate UTF-8 byte induction
  copying under soft-to-STE switching (`0/18` hard exact, `36.3 min` for `L2-Full`).
- **Why Griffith AP法 (`APTrainer`) was not used directly here**:
  In the pinned Griffith library (`griffith_logic/training/rules/ap.py`), `APTrainer`
  requires flat classification pipelines (`nn.Sequential` ending in `GroupSum`) composed
  solely of `LutLayer(parameterization="independent", wiring="fixed")`. It explicitly
  rejects `LogicLadderGPT`, autoregressive self-attention, residual transformers, and
  `wiring="learnable"`. This experiment therefore uses official Griffith `LogicLadderGPT`
  with WARP `LutLayer`s, `LearnableMapping`, and `kd_kl_loss_seq`.

## Training diagnostics summary (Promoted vs FP control)

| Quantity | Promoted LUT (`model.pt`) | FP control (`fp-control.pt`) |
|---|---:|---:|
| Parameters | 497,160 | 991,236 |
| MPS training seconds | 1228.5 | 276.1 |
| Teacher-forced token accuracy (hard, train) | 98.77% | 99.71% |
| Teacher-forced NLL nats (hard, train) | 0.0379 | 0.0096 |
| Teacher-forced seq accuracy (hard, train) | 85.70% | 98.51% |

## Scope, contract, and artifacts

- Teacher source revision: `6b73dd979ab3ef22171aaf0016723607631d71fe`. Original `python -m pybot`
  runtime under `bot/` is untouched.
- Promoted artifacts: [model.pt](model.pt), [training.json](training.json),
  [hard-validation.json](hard-validation.json), [soft-validation.json](soft-validation.json),
  [hard-diverse-validation.json](hard-diverse-validation.json).
- Control & ablation artifacts: [fp-control.pt](fp-control.pt),
  [fp-control-validation.json](fp-control-validation.json),
  [fp-control-diverse-validation.json](fp-control-diverse-validation.json),
  [training-logic-ffn-kd.json](training-logic-ffn-kd.json),
  [logic-ffn-kd-hard-diverse-validation.json](logic-ffn-kd-hard-diverse-validation.json),
  [training-logic-ffn-kd-fixed-n4.json](training-logic-ffn-kd-fixed-n4.json),
  [training-logic-ffn-kd-learnable-n4.json](training-logic-ffn-kd-learnable-n4.json),
  [training-logic-ffn-kd-learnable-n6.json](training-logic-ffn-kd-learnable-n6.json),
  [training-logic-attn-l2-n4.json](training-logic-attn-l2-n4.json),
  [training-logic-attn-l2-n6.json](training-logic-attn-l2-n6.json),
  [baseline-0of18-summary.json](baseline-0of18-summary.json), [summary.json](summary.json).
- [Japanese guide](../../docs/moduleio/README.ja.md) · [Repository README](../../README.md).
