# PyBot × Griffith: complete module I/O distillation

An experimental fork of [PyBot](https://github.com/huggingface/pybot) that distills **complete module call results** — return value, updated conversation memory, mutable candidate arguments, and exceptions — directly from raw serialized call arguments using the official [Griffith](https://github.com/tsukushi-gate/Griffith) library (`e0675f4e`).

[日本語ガイド](docs/moduleio/README.ja.md) · [Measured Results](experiments/moduleio/RESULTS.md) · [Summary JSON](experiments/moduleio/summary.json)

> **Scope**: Bounded conformance experiment over **4 trained modules / 6 hooks** (`favourites.observe/respond`, `group_ages.observe/respond`, `name_taken_back.observe`, `a_cake_between_its_layers.respond`), not a replacement for all 116 PyBot hooks. The original rule-based chatbot (`python -m pybot` under `bot/`) is untouched, and learned inference via `moduleio.predictor` or `python -m moduleio predict` runs strictly without importing `bot/`.

## Measured Results

All models are evaluated with input-only prompts (no teacher fallback, no answer lookup, no prefix hints) under **hard** discretized LUT inference (`whole_call_exact`: exact match across return value, full updated memory, candidates, and exception).

| Model | LUT Arity (`2^N`) | Params | Orig 18 Hard / Soft | Diverse 318 Hard | Diverse 318 Valid JSON | CPU Mean (318) |
|---|---|---:|---:|---:|---:|---:|
| Previous baseline (`0/18`) | N=2 (4) | — | 0/18 / 0/18 | — | — | — |
| Always-None + unchanged memory | — | 0 | 4/18 / — | 38/318 (11.9%) | 318/318 | — |
| FP Control (`L0`, `fp/fp`, `4L/mlp4`) | — | 991,236 | 14/18 / — | 247/318 (77.7%) | 312/318 | 0.311s |
| **Promoted: `L0+LogicFFN-KD-Fixed N=4` (`4L/mlp2`, `model.pt`)** | **N=4 (16)** | **497,160** | **15/18 / 16/18** | **214/318 (67.3%)** | **311/318 (97.8%)** | **1.329s** |
| `L0+LogicFFN-KD-Learnable N=4` (`4L/mlp2`) | N=4 (16) | 2,070,024 | 15/18 / 15/18 | 211/318 (66.4%) | 309/318 | 1.436s |
| `L0+LogicFFN-KD-4096` (`4L/mlp4`, fixed) | N=2 (4) | 480,776 | 13/18 / 14/18 | 223/318 (70.1%) | 305/318 | 1.184s |
| `L0+LogicFFN-KD-Learnable N=2` (`4L/mlp2`) | N=2 (4) | 1,259,016 | 15/18 / 15/18 | 201/318 (63.2%) | 307/318 | 1.135s |
| `L0+LogicFFN-KD-Learnable N=6` (`4L/mlp2`) | N=6 (64) | 2,954,760 | 13/18 / 13/18 | 136/318 (42.8%) | 283/318 | 3.170s |
| `L2-LogicAttn` (scratch, `xnor_popcount+softmax`, learnable) | N=2 / 4 / 6 | 1.13M / 1.54M / 2.01M | 4/18 each | 36 / 34 / 23 of 318 | 281 / 237 / 143 | 1.35s / 1.69s / 2.91s |

- **From `0/18` to `15/18` & `214/318`**: Fixing initial attention collapse, switching to compact lossless `compact-v2` serialization ([`moduleio/wire.py`](moduleio/wire.py)), annealing WARP LUTs with late `soft_to_ste`, and distilling from [`experiments/moduleio/fp-control.pt`](experiments/moduleio/fp-control.pt) via Griffith `kd_kl_loss_seq` eliminated the induction-head saddle point without a large soft-to-hard drop.
- **Why `N=4` fixed wiring won promotion**: Raising `LogicMLP` arity from `N=2` (4 entries/LUT) to **`N=4`** (16 entries/LUT) boosts 2-layer FFN fan-in (`4 → 16` inputs) enough that fixed sparse wiring outperforms both `N=2 Learnable` (`214/318` vs `201/318`) and `N=4 Learnable` (`211/318`) while cutting parameters by **60%** (`497,160` vs `1,259,016`).
- **Why `N=6` and `L2-LogicAttn` trailed**: `N=6` multiplies Walsh-Hadamard basis work `16×` (`O(2^N)`), forcing smaller micro-batches (`4×2`) on 16 GB MPS, slowing CPU inference `2.4×` (`3.170s`), and hurting late STE convergence (`136/318`). In `L2-LogicAttn`, pre-LUT `Binarize(threshold=0.0)` and 32-bit `xnor_popcount` cap attention capacity before truth-table evaluation, leaving `N=2/4/6` flat at `4/18` from scratch.

### Compiled C99 Hard Inference (`export-c`)

[`moduleio/export_c.py`](moduleio/export_c.py) compiles a trained checkpoint into a self-contained C99 shared library with causal KV-cache, folded static LUT wiring, and 64-bit Boolean truth-table lookups. Benchmarked on Apple Silicon ARM64 (`Apple clang 21.0.0`, `-O3 -mcpu=native`, single-threaded C99 vs 8-thread PyTorch CPU on Diverse 318, `batch=1`, `hard` mode):

| Model | `.pt` → `.bin` | Orig 18 (Hard) | Diverse 318 (Hard) | PT CPU (8t) | C99 + KV (1t) | C99 Tok/s | Speedup vs PT |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Promoted `model.pt` (`FFN N=4 Fixed`)** | **2.13 → 1.87 MB** | **15/18** | **214/318 (67.3%)** | **1,328.7 ms** (`75.2 tok/s`) | **24.16 ms** (p95 `27.96 ms`) | **4,134 tok/s** | **55.0×** |
| `FFN N=2 Learn` (`4L/mlp2`) | 4.98 → 1.81 MB | 15/18 | 201/318 (63.2%) | 1,134.7 ms (`89.5 tok/s`) | 23.92 ms (p95 `27.53 ms`) | 4,247 tok/s | 47.4× |
| `FFN N=4 Learn` (`4L/mlp2`) | 8.20 → 1.87 MB | 15/18 | 211/318 (66.4%) | 1,435.5 ms (`70.3 tok/s`) | 24.22 ms (p95 `27.84 ms`) | 4,166 tok/s | 59.3× |
| `FFN N=6 Learn` (`4L/mlp2`) | 11.70 → 2.06 MB | 13/18 | 137/318* (`PT 136/318`) | 3,169.6 ms (`31.8 tok/s`) | 28.69 ms (p95 `32.83 ms`) | 3,512 tok/s | 110.5× |
| `FP Control` (`L0`, `4L/mlp4`) | 3.80 → 3.78 MB | 14/18 | 247/318 (77.7%) | 310.9 ms (`327.3 tok/s`) | 66.12 ms (p95 `74.91 ms`) | 1,539 tok/s | 4.7× |

<sub>\* `FFN N=6 Learn` had 1 borderline float32 ULP sign flip near zero (`max_diff = 5.72e-6`) giving C99 `137/318` vs PyTorch `136/318`; the other 4 models matched PyTorch 100% identically on both Orig 18 and Diverse 318.</sub>

- **Why C99 makes the LUT model `2.74×` faster than C99 FP Control**: PyTorch `hard` mode still runs float32 Walsh-Hadamard transforms without KV-cache (`O(T^2)`). In C99 + KV-cache (Orig 18 profile), `FP Control` (`66.75 ms`) spends `43.80 ms` (65.6%) in FP FFN vs `22.25 ms` (33.3%) in Attention, whereas `model.pt` (`23.68 ms`) cuts FFN **37×** to `1.18 ms` (`0.80 ms` `LutLayer_0` WH + `0.38 ms` discrete 64-bit `LutLayer_1` truth-table lookup + `FromLogic`), leaving Attention at `21.79 ms` (92.0%). Disabling KV-cache in C99 drops `model.pt` to `1,892.8 ms/call` (`52.8 tok/s`).
- **Compile-time wiring elimination & official Griffith C kernel**: `export-c` lowers truth tables via `griffith_logic.export.discretize.truth_tables_from_lut_layer` and folds `LearnableMapping` weights via `argmax` into static `uint16_t` indices (`11.70 → 2.06 MB` for `N=6`; compiled shared library is `49.6 KB`). Compiling block 0's pure-Boolean 2-layer `N=4` LUT stack with official Griffith `deploy.cbackend.generate_c` runs in **`444.7 ns/block`** (`1.78 µs` for 4 blocks; 100/100 exact match vs `ir.simulate`).
- **`ToLogic` binarization trade-off**: Passing `binarize_tologic=True` converts `LutLayer_0` into a discrete truth-table lookup too, but saves only `0.35 ms/call` (`24.16 → 23.81 ms`, `4,209 tok/s`) because Attention dominates 92% of runtime, while dropping exact match to `10/18` (Orig 18) and `109/318` (Diverse 318).

## Architecture & Boundaries

- **Discretized Griffith LUT components in [`experiments/moduleio/model.pt`](experiments/moduleio/model.pt)**: 8 WARP **4-input** `LutLayer` layers (**2,048 LUT neurons** × 16 truth-table entries = 32,768 entries across all 4 `LogicMLP` FFN blocks, `--ffn-n-inputs 4 --wiring fixed`), evaluated deterministically with hard truth tables at inference.
- **Remaining FP components**: `token_embed`, `pos_embed`, `LayerNorm`, `nn.MultiheadAttention` (`L0`), vocabulary output `head`, and continuous `ToLogic('sigmoid')` / `FromLogic('groupsum_centered')` bridges remain floating-point. Hard evaluation discretizes every LUT (and learned wiring when `--wiring learnable` is enabled); it does **not** make the entire Transformer a pure Boolean circuit.
- **Griffith library & AP法 note**: Pinned to official Griffith `e0675f4e36596954aa424d522e0ee966bba9a3e1`. Griffith's `APTrainer` (Akiho Propagation) only supports flat `nn.Sequential` fixed-wiring `independent` classifiers and explicitly rejects `LogicLadderGPT` and `wiring="learnable"`, so training uses `LogicLadderGPT` with `soft_to_ste` and `kd_kl_loss_seq`.

## Module I/O Contract

| Hook | Input | Learned Output |
|---|---|---|
| `observe` | user text, full memory before call | return value, full memory after call, exception |
| `respond` | user text, full memory before call | reply or `None`, full memory after call, exception |
| `choose` | user text, full memory, candidate list | chosen item, full updated memory & candidates, exception |
| `after` | user text, chosen reply, full memory | return value, full memory after call, exception |

Across PyBot revision `7d63c7fb`, `python -m moduleio inventory` discovers **116 hooks in 102 modules** (16 `observe`, 97 `respond`, 1 `choose`, 2 `after`). The `compact-v2` codec encodes `environment` before `arguments` and represents unchanged memory/candidate objects with `"="` sentinels while preserving tuple/list distinctions, dict order, float bits, bytes, and shared references (cycles and unsupported objects fail explicitly).

## Quickstart

Requires **Python 3.11+** and PyTorch for `moduleio` (the original rule-based `pybot` continues to need only Python 3.8+).

```sh
pip install '.[moduleio]'
```

### Run Teacher-Free Inference

Python API using the checked-in promoted checkpoint [`experiments/moduleio/model.pt`](experiments/moduleio/model.pt):

```python
from moduleio.predictor import load_predictor

predictor = load_predictor("experiments/moduleio/model.pt")
result = predictor.predict_call(
    "group_ages", "respond",
    {
        "text": "how old is my youngest",
        "memory": {
            "group_ages": {
                "word": "kids",
                "ages": [4, 7],
                "line": "my kids are 4 and 7",
            }
        },
    },
    seed=0,
    environment_date="2026-11-21",
)
print(result["return"])  # "Your youngest is 4."
```

Or via the CLI:

```sh
cat <<'EOF' > /tmp/pybot_call.json
{
  "text": "how old is my youngest",
  "memory": {
    "group_ages": {
      "word": "kids",
      "ages": [4, 7],
      "line": "my kids are 4 and 7"
    }
  }
}
EOF

python -m moduleio predict \
  --checkpoint experiments/moduleio/model.pt \
  --module group_ages \
  --hook respond \
  --arguments /tmp/pybot_call.json \
  --seed 0 \
  --date 2026-11-21
```

### Compile & Run in Native C99

Export the promoted checkpoint to self-contained C99 (`moduleio_runtime.c`, `moduleio_runtime.h`, `weights.bin`, and compiled shared library) and run KV-cached hard inference via [`CompiledCModel`](moduleio/export_c.py):

```sh
python -m moduleio export-c \
  --checkpoint experiments/moduleio/model.pt \
  --output-dir /tmp/pybot_c
```

```python
from moduleio.data import input_for_call
from moduleio.evaluation import decode_output
from moduleio.export_c import CompiledCModel

args = {
    "text": "how old is my youngest",
    "memory": {
        "group_ages": {
            "word": "kids",
            "ages": [4, 7],
            "line": "my kids are 4 and 7",
        }
    },
}
prompt = input_for_call(
    "group_ages", "respond", args,
    seed=0, environment_date="2026-11-21", wire_format="compact-v2",
)
with CompiledCModel("/tmp/pybot_c") as model:
    gen = model.generate(prompt, profile=True)
result = decode_output(gen["text"], "respond", input_arguments=args, wire_format="compact-v2")
print(result["return"])  # "Your youngest is 4."
```

### Evaluate Checked-In Models

```sh
# Untouched 18-call canonical held-out benchmark (hard mode)
python -m moduleio evaluate \
  --checkpoint experiments/moduleio/model.pt \
  --data experiments/moduleio/model.pt.validation.jsonl.gz \
  --output /tmp/hard-validation.json \
  --mode hard --device cpu --threads 8

# Leak-free 318-call diverse unseen-entity benchmark (hard mode)
python -m moduleio evaluate \
  --checkpoint experiments/moduleio/model.pt \
  --data experiments/moduleio/diverse-validation.jsonl.gz \
  --output /tmp/hard-diverse-validation.json \
  --mode hard --device cpu --threads 8
```

### Reproduce Training & Run Tests

Training expects a single GPU/MPS process at a time on 16 GB unified-memory Macs (`--grad-accum 2` keeps peak MPS driver memory around ~4.2 GB for `N=4`). You can also explore `--ffn-n-inputs {2,4,6}`, `--n-inputs {2,4,6}`, and `--wiring {fixed,learnable}`.

```sh
# Optional: regenerate the 3,681-call leak-free teacher corpus
python examples/moduleio/make_diverse_specs.py
python -m moduleio collect-calls \
  --spec examples/moduleio/diverse_calls.jsonl \
  --output examples/moduleio/diverse_teacher_calls.jsonl.gz

# 1. Train the L0 FP control (teacher for Griffith KD; trained on diverse-train only)
python -m moduleio train \
  --data experiments/moduleio/diverse-train.jsonl.gz \
  --validation-data experiments/moduleio/diverse-validation.jsonl.gz \
  --checkpoint experiments/moduleio/fp-control.pt \
  --report experiments/moduleio/training-fp-control.json \
  --wire-format compact-v2 --fp-baseline \
  --max-seq-len 1024 --d-model 128 --n-heads 4 --n-layers 4 --mlp-ratio 4 \
  --pos-init sinusoidal --embed-init-std 0.25 \
  --steps 4000 --batch-size 16 --learning-rate 0.003 \
  --weight-decay 0.01 --lr-schedule wsd --warmup-steps 150 \
  --loss-weighting example --train-mode soft \
  --device mps --seed 0 --threads 8

# 2. Train the promoted 4-input Griffith WARP LUT FFN model with Griffith KD
python -m moduleio train \
  --data experiments/moduleio/diverse-train.jsonl.gz \
  --validation-data experiments/moduleio/diverse-validation.jsonl.gz \
  --checkpoint experiments/moduleio/model.pt \
  --report experiments/moduleio/training.json \
  --wire-format compact-v2 \
  --attention fp --ffn logic --ffn-n-inputs 4 \
  --wiring fixed \
  --init-from experiments/moduleio/fp-control.pt \
  --teacher-checkpoint experiments/moduleio/fp-control.pt \
  --kd-alpha 0.5 --kd-temp 2.0 \
  --max-seq-len 1024 --d-model 128 --n-heads 4 --n-layers 4 --mlp-ratio 2 \
  --pos-init sinusoidal --embed-init-std 0.25 \
  --steps 2400 --batch-size 8 --grad-accum 2 --learning-rate 0.001 --lut-lr-scale 10.0 \
  --weight-decay 0.01 --lr-schedule wsd --warmup-steps 100 \
  --loss-weighting example --train-mode soft_to_ste \
  --device mps --seed 0 --threads 8

# Trace full all-hook conversations & run all test suites
python -m moduleio inventory
python -m moduleio collect \
  --scenarios examples/moduleio/conversations.jsonl \
  --output /tmp/runtime_trace.jsonl.gz
python -m unittest discover -s tests/moduleio -v   # 67 tests
python -m unittest discover -s tests -v            # 12 original CLI tests
```

> **Evaluation Contract**: Evaluation never crops overlength prompts or outputs, keeps malformed/missing-EOS outputs in the denominator, and scores single-call predictions given recorded pre-call memory rather than multi-turn end-to-end rollouts.

## Repository Layout

| Path | Contents |
|---|---|
| `moduleio/` | Teacher trace collector, typed `codec` & `compact-v2` `wire` codec, `LogicLadderGPT` wrapper, Griffith KD & `soft_to_ste` trainer, evaluator, teacher-free predictor, native C99 exporter/runtime (`export_c.py`) |
| `docs/moduleio/` | [Japanese design & reproduction guide](docs/moduleio/README.ja.md) |
| `examples/moduleio/` | Leak-free diverse call spec generator (`3,681` calls), small `120`-call fixture, and `232`-call natural trace across all 116 hooks |
| `experiments/moduleio/` | Promoted [`model.pt`](experiments/moduleio/model.pt), [`fp-control.pt`](experiments/moduleio/fp-control.pt), train/validation splits, measured JSON reports, [`RESULTS.md`](experiments/moduleio/RESULTS.md), and [`summary.json`](experiments/moduleio/summary.json) |
| `tests/moduleio/` | Unit tests for codec, wire format, data splits, evaluation contract, training, `N ∈ {2,4,6}` LUTs, teacher-blocked inference, and C99 parity (`67` tests) |
| `bot/`, `pybot.py`, `tests/` | Original rule-based PyBot runtime and CLI tests (unmodified) |

---

## Original PyBot

The description below documents the original rule-based chatbot.

<p align="center"><img src="assets/banner.png" alt="PyBot: no weights, no training, just rules" width="100%"></p>

<div align="center">
  <a href="https://huggingface.co/spaces/HuggingFaceSwarms/PyBot" target="_blank">
    <img alt="Hugging Face Space: PyBot demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-ffc107?color=ffc107&logoColor=white" />
  </a>
</div>

### PyBot: can Python be a chatbot?

LLMs drive today’s chatbots and agents, but chatbots long predate them. Sixty years ago, [ELIZA](https://en.wikipedia.org/wiki/ELIZA) held conversations by matching patterns in what the user typed and answering with predefined templates. It could seem insightful, but its repertoire was narrow, and expanding it meant writing more rules by hand. How far could a rule-based chatbot go if a swarm of agents did that work instead?

We challenged a swarm of over 1,000 agents to build a chatbot entirely in Python: no weights, no training, just if-else branches, regex patterns and lookup tables. The agents wrote and evaluated over 5,000 versions, and the chatbot’s win rate on [SmolTalkBench](https://huggingface.co/datasets/HuggingFaceSwarms/SmolTalkBench), where an LLM judge compares each reply with a reference answer, climbed from 0% to 26%. That puts it on par with SmolLM-360M-Instruct, which just a few years ago was state of the art for models under 1B parameters.

![PyBot's win rate over the evaluated commits on a log scale: every commit as a dot and the best so far as a line, with three SmolLM baselines as dashed lines, and its lines of Python code in grey on the right axis](assets/progress.png)

The result is PyBot: around 640k lines of code that can answer everyday questions, backed by 550k lines of tests the agents wrote themselves.

### Try it out

Try PyBot in your terminal below, or in the browser on [Hugging Face](https://huggingface.co/spaces/HuggingFaceSwarms/PyBot), where each reply shows which of PyBot’s modules produced it.

PyBot needs Python 3.8 or newer and nothing else: no packages to install and no model to download.

```sh
git clone https://github.com/huggingface/pybot
cd pybot
python -m pybot
```

That starts a chat in your terminal:

```text
    ____        ____        __
   / __ \__  __/ __ )____  / /_
  / /_/ / / / / __  / __ \/ __/
 / ____/ /_/ / /_/ / /_/ / /_
/_/    \__, /_____/\____/\__/
      /____/

A chatbot written in plain Python. No weights, no training, just rules.
Type a message and press Enter. /reset starts a new conversation, /exit quits.

user: hi, i'm Sam
pybot: Nice to meet you, Sam. What can I help with?

user: my kettle is full of limescale, how do i get rid of it?
pybot: Half fill the kettle with equal parts water and white vinegar, or a sachet of citric acid, boil it and leave it half an hour, then tip it out and boil fresh water twice before you use it again. In a hard-water area do it every couple of months: the scale forms because hard water leaves calcium behind as it boils, and a thick layer makes the kettle slow and puts a scum on the tea.

user: how long does that take
pybot: Half an hour.

user: what's my name?
pybot: You're Sam.
```

Type `/reset` to clear the screen and start a new conversation, and `/exit` to quit. PyBot takes a few seconds to start, because Python has about 640,000 lines of code to load, and about a second to reply.

To ask one question and exit, or to pipe messages in and get one reply per line:

```sh
python -m pybot -p "Why do onions make you cry?"
echo "how long should i boil an egg" | python -m pybot
```

| Option | What it does |
|---|---|
| `-p TEXT`, `--prompt TEXT` | reply to this one message and exit |
| `-i`, `--interactive` | with `-p`, keep chatting after the first reply |
| `-s N`, `--seed N` | fix PyBot's choice of wording, to get the same replies again |
| `-v`, `--verbose` | after each reply, show the module that produced it and the time it took |

For a `pybot` command you can run from anywhere, install the clone with `pip install .`.

### What is in this repository

| Path | What it holds |
|---|---|
| `bot/` | PyBot itself: 528 Python files and about 640,000 lines, exactly as the swarm wrote them |
| `pybot.py` | the command line shown above. It contains no chatbot logic |
| `tests/` | tests for the command line: `python -m unittest discover -s tests` |
| `assets/` | the banner and the chart |

Every turn, each module in `bot/modules/` may propose a reply, and an arbiter (`bot/modules/arbiter.py`) picks the one you get. The modules build on four shared parts:

| Part | Where | Lines |
|---|---|---:|
| Understanding engine: what you typed and meant, with typos, chat-speak, the topic and what "it" refers to | `bot/understanding/` | 25,000 |
| Memory store: what you and PyBot have said so far | `bot/memory/` | 109,000 |
| Knowledge base: facts about the world, kept as data | `bot/knowledge/` | 317,000 |
| Response composer: how a reply is phrased, and what to say when nothing fits | `bot/compose/` | 104,000 |

The modules themselves are another 81,000 lines.

## License

Apache License 2.0. See [LICENSE](LICENSE).
