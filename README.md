# PyBot × Griffith: complete module I/O distillation

This fork adds an experimental way to learn **each module's complete call result**
with the official [Griffith](https://github.com/tsukushi-gate/Griffith) library.
The target includes the return value, the entire updated conversation memory,
mutable candidate arguments, and exceptions. The model receives the original
call arguments, rather than six precomputed eligibility flags.

**This is a bounded training and conformance experiment over 4 trained modules (6 hooks),
not a full 116-hook PyBot replacement.** The original `python -m pybot` chatbot
continues to run its existing rules under `bot/`. Learned calls are exposed separately through
`python -m moduleio predict` and never call the original teacher as a fallback.

**Measured improvement in hard discretized LUT evaluation: 0/18 → 15/18 whole-call exact matches**
on the untouched 18-call held-out validation set (`15/18` hard / `16/18` soft; `18/18` valid JSON,
`17/18` return exact, `16/18` memory exact, beating the upgraded floating-point control's `14/18`).
On a strictly leak-free **318-call diverse unseen-entity validation split**, upgrading Griffith's
`LogicMLP` FFNs from **2-input** to **4-input** WARP LUTs (`model.pt`, `497,160` params,
2,048 4-input LUT neurons) matches **214/318 (67.3%)** (`211/318` with learnable wiring), beating
the 2-input learnable-wiring model (**201/318 = 63.2%**, `1,259,016` params) with 60% fewer parameters,
while 6-input LUTs (`ffn_n_inputs=6`) score **136/318 (42.8%)** and the 4,096-LUT 2-input variant
matches **223/318 (70.1%)** (`13/18` Orig 18), versus **38/318 (11.9%)** Always-None and
**247/318 (77.7%)** for the FP control.

[日本語の設計・再現手順](docs/moduleio/README.ja.md) ·
[Measured experiment](experiments/moduleio/RESULTS.md)

## Module I/O contract

| Hook | Input | Learned output |
|---|---|---|
| `observe` | text, full memory before the call | return, full memory after the call, exception |
| `respond` | text, full memory before the call | reply or `None`, full memory after the call, exception |
| `choose` | text, full memory, candidates | return, full updated memory and candidates, exception |
| `after` | text, chosen reply, full memory | return, full memory after the call, exception |

The collector discovers **116 phase hooks in 102 modules** in the recorded PyBot
revision. It traces the original runtime without changing the order of calls,
candidate memory copies, selected response, or random-number progression.
`None` responses with state changes are retained. The `compact-v2` wire codec
([wire.py](moduleio/wire.py)) places `environment` before `arguments` and uses `"="`
unchanged-state sentinels while preserving tuple/list distinctions, dictionary order,
floating-point bits, and shared container references. Unsupported objects or cycles
fail explicitly.

The promoted student (`experiments/moduleio/model.pt`) uses Griffith's `LogicLadderGPT`
with **discretized 4-input WARP `LutLayer` FFNs (2,048 LUT neurons × 16 entries,
`ffn_n_inputs=4`, `wiring="fixed"`)** across all 4 Transformer blocks, distilled with
Griffith's `kd_kl_loss_seq` from the FP control (`fp-control.pt`). We also expose
2-, 4-, and 6-input LUT support across both FFN (`--ffn-n-inputs`) and Attention
(`--n-inputs`) with optional `LearnableMapping` (`--wiring learnable`).
Embeddings, `LayerNorm`, `nn.MultiheadAttention`, and the vocabulary head remain
floating-point: **hard evaluation means all LUTs and learned LUT wiring are discretized;
it does not mean the entire Transformer is a pure Boolean circuit.** Notice also that
Griffith's `APTrainer` (Akiho Propagation) only supports flat `nn.Sequential`
fixed-wiring `independent` LUT classifiers and rejects `LogicLadderGPT` and
`wiring="learnable"`, so training uses `LogicLadderGPT` with `soft_to_ste` and
Griffith KD instead. The library is pinned to `e0675f4e36596954aa424d522e0ee966bba9a3e1`.

## Install and reproduce

The learning experiment requires Python 3.11+ and PyTorch; the original PyBot
continues to need only Python 3.8+.

```sh
pip install '.[moduleio]'
python -m moduleio inventory

# Generate the 3,681-call leak-free dataset (diverse-train / diverse-validation are checked in).
python examples/moduleio/make_diverse_specs.py
python -m moduleio collect-calls \
  --spec examples/moduleio/diverse_calls.jsonl \
  --output examples/moduleio/diverse_teacher_calls.jsonl.gz

# 1. Train the L0 FP control (teacher for Griffith KD; trained on diverse-train only).
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

# 2. Train the promoted 4-input Griffith WARP LUT FFN model with Griffith KD.
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

# 3. Evaluate on both held-out benchmarks in discretized hard mode.
python -m moduleio evaluate \
  --checkpoint experiments/moduleio/model.pt \
  --data experiments/moduleio/model.pt.validation.jsonl.gz \
  --output experiments/moduleio/hard-validation.json --mode hard --device cpu --threads 8

python -m moduleio evaluate \
  --checkpoint experiments/moduleio/model.pt \
  --data experiments/moduleio/diverse-validation.jsonl.gz \
  --output experiments/moduleio/hard-diverse-validation.json --mode hard --device cpu --threads 8
```

Evaluation freely generates the **entire typed output** from the input alone.
It reports whole-call, return, and memory exact match, valid completed outputs,
active/inactive cases, per-hook results, and a `None`/unchanged-memory baseline.
Malformed outputs, missing EOS, and overlength inputs remain failures in the
denominator. Teacher-forced token accuracy is a separate diagnostic.

Natural conversations can also be traced without reducing their states:

```sh
python -m moduleio collect \
  --scenarios examples/moduleio/conversations.jsonl \
  --output examples/moduleio/runtime_trace.jsonl.gz
python -m unittest discover -s tests/moduleio -v
```

Natural traces include the complete pre-call RNG state in model inputs. They
can be much longer than the small demonstration model's context. Training reports
every excluded record and never silently truncates inputs or outputs. Independent
call fixtures instead supply a seed and preserve the caller's process RNG.
Dates are recorded, not virtualized; module-global caches are not snapshotted.
No guarantee for arbitrary input lengths, hidden process state, or an entire
learned conversation is implied by a single-call benchmark.

## Added files

| Path | Purpose |
|---|---|
| `moduleio/teacher.py`, `codec.py`, `wire.py` | Complete teacher calls and lossless `legacy-v1` / `compact-v2` serialization |
| `moduleio/data.py` | Input-only prompts, full targets, group splits, explicit coverage |
| `moduleio/model.py`, `training.py` | Official Griffith `LogicLadderGPT`, `wiring=learnable`, Griffith KD, checkpoints |
| `moduleio/evaluation.py`, `predictor.py` | Free-generation conformance and teacher-free prediction |
| `examples/moduleio/` | Input fixtures, recorded calls, all-hook conversation trace |
| `experiments/moduleio/` | Reproducible weights, split records, measured results |

---

## Original PyBot

The description below documents the original rule-based chatbot.

<p align="center"><img src="assets/banner.png" alt="PyBot: no weights, no training, just rules" width="100%"></p>

<div align="center">
  <a href="https://huggingface.co/spaces/HuggingFaceSwarms/PyBot" target="_blank">
    <img alt="Hugging Face Space: PyBot demo" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-ffc107?color=ffc107&logoColor=white" />
  </a>
</div>

# PyBot: can Python be a chatbot?

LLMs drive today’s chatbots and agents, but chatbots long predate them. Sixty years ago, [ELIZA](https://en.wikipedia.org/wiki/ELIZA) held conversations by matching patterns in what the user typed and answering with predefined templates. It could seem insightful, but its repertoire was narrow, and expanding it meant writing more rules by hand. How far could a rule-based chatbot go if a swarm of agents did that work instead?

We challenged a swarm of over 1,000 agents to build a chatbot entirely in Python: no weights, no training, just if-else branches, regex patterns and lookup tables. The agents wrote and evaluated over 5,000 versions, and the chatbot’s win rate on [SmolTalkBench](https://huggingface.co/datasets/HuggingFaceSwarms/SmolTalkBench), where an LLM judge compares each reply with a reference answer, climbed from 0% to 26%. That puts it on par with SmolLM-360M-Instruct, which just a few years ago was state of the art for models under 1B parameters.

![PyBot's win rate over the evaluated commits on a log scale: every commit as a dot and the best so far as a line, with three SmolLM baselines as dashed lines, and its lines of Python code in grey on the right axis](assets/progress.png)

The result is PyBot: around 640k lines of code that can answer everyday questions, backed by 550k lines of tests the agents wrote themselves.

## Try it out

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

## What is in this repository

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
