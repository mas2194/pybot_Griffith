# 各モジュールの入出力全体を学習する

`moduleio` は、PyBot の各モジュールを教師として、その呼び出しの入出力全体を
公式 [Griffith](https://github.com/tsukushi-gate/Griffith) で学習する実験です。
入力文を既存の解析処理で条件フラグに変換してから判断させる方式ではなく、
元の呼び出し引数全体をバイト列のまま入力にします。

**未学習・リーク無しの検証セットにおける最新実測結果（hard自由生成、離散化LUT推論）:**
- 従来の未開封18件ベンチマーク（`experiments/moduleio/model.pt.validation.jsonl.gz`）:
  初回の **0/18**（LUT・FP対照ともに0件）から **15/18（hard / 16/18 soft）** へ改善し、
  同規模の改良版FP対照（**14/18**）を上回りました（`ok 18/18`・`return_exact 17/18`・`memory_exact 16/18`）。
- 未知のエンティティを含む318件の多様化検証セット（`experiments/moduleio/diverse-validation.jsonl.gz`）:
  昇格チェックポイント `model.pt`（`L0+LogicFFN-KD-Fixed N=4`、`ffn_n_inputs=4` の4入力WARP LUT 2,048ニューロン、497,160パラメータ）は
  **214/318（67.3%）**（`ok 311/318 = 97.8%`）を達成し、従来の2入力版 `model-logic-ffn-kd-learnable.pt`（**201/318 = 63.2%**、1,259,016パラメータ）を約2.5分の1のパラメータ数で上回りました。
  同2入力×4,096 LUT版 `model-logic-ffn-kd.pt` は **223/318（70.1%）**、4入力・接続学習版は **211/318（66.4%）**、6入力版は **136/318（42.8%）** です（常に `None` 基準は **38/318（11.9%）**、FP対照は **247/318（77.7%）**）。

| モデル | 入力数 `2^N` | パラメータ数 | Orig 18 (hard / soft) | Diverse 318 (hard) | CPU 平均推論時間 |
|---|---:|---:|---:|---:|---:|
| FP対照 (`L0`, `fp/fp`) | — | 991,236 | 14/18 / — | 247/318 (77.7%) | 0.311s |
| **昇格 `N=4` 固定配線 (`model.pt`)** | **N=4 (16)** | **497,160** | **15/18 / 16/18** | **214/318 (67.3%)** | **1.329s** |
| `N=4` 接続学習 (`Learnable`) | N=4 (16) | 2,070,024 | 15/18 / 15/18 | 211/318 (66.4%) | 1.436s |
| `N=2` 接続学習 (`Learnable`) | N=2 (4) | 1,259,016 | 15/18 / 15/18 | 201/318 (63.2%) | 1.135s |
| `N=6` 接続学習 (`Learnable`) | N=6 (64) | 2,954,760 | 13/18 / 13/18 | 136/318 (42.8%) | 3.170s |
| `L2-LogicAttn` (AttentionまでLUT, 無KD) | N=2 / 4 / 6 | 1.13M–2.01M | 4/18 each | 36 / 34 / 23 of 318 | 1.35–2.91s |

> **適用範囲の注記**: 昇格モデル `experiments/moduleio/model.pt` は全4ブロックのFFNを
> 公式 Griffith `LogicMLP`（WARP **4入力** `LutLayer` 2,048ニューロン × 16エントリ、
> `--ffn-n-inputs 4 --wiring fixed`、評価時は hard 離散LUT）に置き換えています。
> 一方、埋め込み層・`LayerNorm`・`nn.MultiheadAttention`・出力 `head` は浮動小数点演算のまま残っており、
> モデル全体が純粋ブール回路になったわけではありません。また学習対象は4モジュール・6呼び出し口です。

例えば `respond(text, memory)` については、次の関数を学習対象にします。

```python
(module_id, text, memory_before, seed, date)
    -> (return_value, memory_after, exception)
```

返答だけ合っていても、更新後のメモリが違えば「全体一致」には数えません。
`None` を返した場合も、メモリ更新や例外を含めて比較します。

## 実装した境界

| 呼び出し口 | 入力 | 教師出力 |
|---|---|---|
| `observe` | 入力文・呼び出し前メモリ | 戻り値・呼び出し後メモリ・例外 |
| `respond` | 入力文・呼び出し前メモリ | 回答候補または `None`・呼び出し後メモリ・例外 |
| `choose` | 入力文・メモリ・候補一覧 | 選択結果・呼び出し後メモリ・呼び出し後候補一覧・例外 |
| `after` | 入力文・確定した回答・メモリ | 戻り値・呼び出し後メモリ・例外 |

元 PyBot の対象リビジョン `7d63c7fbebb44c7e466e5c8b5c1f341ab767929e` には、
これらの呼び出し口を持つモジュールが102個、呼び出し口が合計116個あります。
内訳は observe 16、respond 97、choose 1、after 2 です。

会話を記録するときは元の runtime の呼び出しを一時的に包みます。`observe` の
順次更新、`respond` ごとのメモリコピー、採用された候補のメモリだけを引き継ぐ
処理をそのまま実行します。採用されなかった候補の入出力も収集します。
元の `bot/runtime.py` および `bot/` 配下は変更していません。

通常のJSONでは tuple が list に変わるため、型付きのシリアライズを用います。
`compact-v2`（[wire.py](../../moduleio/wire.py)）では `environment` を先頭に置き、
未変更のメモリ／候補フィールドを `"="` センチネルで表すことで、辞書の順序、
キーの型、大きな整数、浮動小数のビット、bytes、共有参照を欠落なく保ったまま系列長を圧縮しています。
循環参照や独自クラスは、黙って文字列に変換せず明示的にエラーにします。

## モデルが行う処理と AP法（Akiho Propagation）について

元の引数全体を `compact-v2` でUTF-8バイト列化し、モジュールIDとともに入力します。
学習するのは、完全な出力を型付き文字列表現で書き出したバイト列と、真の終端記号 EOS です。
入力部分に損失はかけず、出力トークンのみに損失（Cross-Entropy + `griffith_logic.convert.distill.kd_kl_loss_seq`）をかけます。

使用する学習器は公式 Griffith（コミット `e0675f4e36596954aa424d522e0ee966bba9a3e1`）の
`LogicLadderGPT` です。LUT・接続学習 `LearnableMapping`・緩和・代理勾配・hard化を独自に再実装していません。

- **接続学習と LUT 学習の仕組み**: `--wiring learnable` により公式 `LutLayer(..., wiring='learnable')`
  内の `LearnableMapping` が漸進的温度減衰 (`tau_soft: 0.2 -> 0.01`) ののちに `argmax` 離散配線へ移行し、
  同時に LUT 真理値表パラメータ（WARP）が `soft_to_ste` スケジュール（WSD 学習率減衰 + 終盤20% `hard_st`）で離散論理ゲートへ収束します。
- **FP 蒸留ウォームスタートの役割**: 未学習状態からの LUT 学習（`8/18`・`117/318`）はエンティティ文字列のバイト単位コピー回路（induction head）形成の鞍点で停滞しました。訓練データのみで学習した FP 対照モデル（`fp-control.pt`）からの `--init-from` と公式 Griffith `kd_kl_loss_seq` により、2入力接続学習版で **15/18・201/318**、さらに4入力固定配線版（`--ffn-n-inputs 4 --wiring fixed`）では **15/18・214/318**（497,160パラメータ）まで引き上げられました。
- **AP法（`griffith_logic.training.rules.ap.APTrainer`）の制約**: 公式 Griffith の APTrainer は
  `nn.Sequential(..., LutLayer(parameterization="independent", wiring="fixed"), ..., GroupSum)` 形式の
  平坦な分類器専用であり、自己回帰 Attention・残差接続・`LogicLadderGPT`・`wiring="learnable"` を明示的に拒否します。
  そのため本モジュールでは `LogicLadderGPT` 上で公式 `LutLayer` + `LearnableMapping` + `kd_kl_loss_seq` を採用しています。

推論時は学習済み重みと入力だけを使います。元モジュールを呼び出したり、正解の
回答候補を検索したり、不正出力を元のルールで補ったりする経路はありません。
未学習の呼び出し口、長さ超過、EOS未生成、不正な出力は明示的に失敗します。

## 教師データの収集方法

### 独立した呼び出し（多様化リーク無しデータ）

```sh
python examples/moduleio/make_diverse_specs.py
python -m moduleio collect-calls \
  --spec examples/moduleio/diverse_calls.jsonl \
  --output examples/moduleio/diverse_teacher_calls.jsonl.gz
```

`examples/moduleio/make_diverse_specs.py` は未開封18件ベンチマーク（`experiments/moduleio/model.pt.validation.jsonl.gz`）と
エンティティ・グループ・完全一致入力が一切重複しないよう検証した上で 3,681 件の呼び出し仕様を生成します
（学習用 `experiments/moduleio/diverse-train.jsonl.gz` 3,363件、多様検証用 `experiments/moduleio/diverse-validation.jsonl.gz` 318件）。
従来の小規模 120件フィクスチャ（`make_specs.py` / `teacher_calls.jsonl.gz`）も比較用にそのまま残しています。

### 実際の会話内の全呼び出し

```sh
python -m moduleio collect \
  --scenarios examples/moduleio/conversations.jsonl \
  --output examples/moduleio/runtime_trace.jsonl.gz
```

このトレースは実際の会話で流れるメモリと乱数の進行を保持します。2ターンから
232件、116種類すべての呼び出し口を記録したサンプルを同梱します。
自然会話トレースは小型モデルのコンテキスト上限を超えることがあり、訓練時は超過レコードを明示的に除外・報告します。

## 学習と検証（単一GPUプロセス推奨・16GB環境）

Python 3.11 以上で `pip install '.[moduleio]'` を実行してください。Apple Silicon 環境では `--device mps` を使って単一プロセスで順に学習し、評価は `--device cpu --threads 8` で行えます。

```sh
# 1. FP対照（L0）の学習
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

# 2. 4入力 Griffith LUT FFN（L0+LogicFFN-KD-Fixed N=4）の学習
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

# 3. 入力のみからの hard 自由生成評価
python -m moduleio evaluate \
  --checkpoint experiments/moduleio/model.pt \
  --data experiments/moduleio/model.pt.validation.jsonl.gz \
  --output experiments/moduleio/hard-validation.json --mode hard --device cpu --threads 8

python -m moduleio evaluate \
  --checkpoint experiments/moduleio/model.pt \
  --data experiments/moduleio/diverse-validation.jsonl.gz \
  --output experiments/moduleio/hard-diverse-validation.json --mode hard --device cpu --threads 8
```

## 推論API

```python
from moduleio.predictor import load_predictor

predictor = load_predictor("experiments/moduleio/model.pt")
result = predictor.predict_call(
    "group_ages", "respond",
    {"text": "how old are they",
     "memory": {"group_ages": {
         "word": "kids", "ages": [4, 7], "line": "my kids are 4 and 7"}}},
    seed=0,
    environment_date="2026-11-21",
)
# result: {"return": ..., "memory": ..., "exception": ...}
```

このAPIは予測結果（`return`, `memory`, `exception`）のみを返し、元モジュールへのフォールバックは一切行いません。
未学習モジュール・不正JSON・EOS未生成・長さ超過は `PredictionError` になります。

## ネイティブ C99 コンパイルと推論速度（`export-c`）

[export_c.py](../../moduleio/export_c.py) は、学習済みチェックポイントの離散 LUT 真理値表（公式 `truth_tables_from_lut_layer`）と学習済み接続（`LearnableMapping` の `argmax` を静的 `uint16_t` 配線へ畳み込み）を抽出し、因果 KV-cache 付き自己完結型 C99 ランタイム（共有ライブラリ + `weights.bin`）へコンパイルします。

Apple Silicon ARM64（`Apple clang 21.0.0`, `-O3 -mcpu=native`、シングルスレッド C99 + KV-cache vs 8スレッド PyTorch CPU、Diverse 318 検証セット `batch=1`・`hard` モード）での実測結果:

| モデル | `.pt` → `.bin` | Orig 18 (hard) | Diverse 318 (hard) | PT CPU (8t) | C99 + KV (1t) | C99 スループット | 高速化倍率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **昇格 `model.pt` (`FFN N=4` 固定)** | **2.13 → 1.87 MB** | **15/18** | **214/318 (67.3%)** | **1,328.7 ms** (`75.2 tok/s`) | **24.16 ms** (p95 `27.96 ms`) | **4,134 tok/s** | **55.0×** |
| `FFN N=2` 接続学習 | 4.98 → 1.81 MB | 15/18 | 201/318 (63.2%) | 1,134.7 ms (`89.5 tok/s`) | 23.92 ms (p95 `27.53 ms`) | 4,247 tok/s | 47.4× |
| `FFN N=4` 接続学習 | 8.20 → 1.87 MB | 15/18 | 211/318 (66.4%) | 1,435.5 ms (`70.3 tok/s`) | 24.22 ms (p95 `27.84 ms`) | 4,166 tok/s | 59.3× |
| `FFN N=6` 接続学習 | 11.70 → 2.06 MB | 13/18 | 137/318* (`PT 136/318`) | 3,169.6 ms (`31.8 tok/s`) | 28.69 ms (p95 `32.83 ms`) | 3,512 tok/s | 110.5× |
| `FP対照` (`L0`, `4L/mlp4`) | 3.80 → 3.78 MB | 14/18 | 247/318 (77.7%) | 310.9 ms (`327.3 tok/s`) | 66.12 ms (p95 `74.91 ms`) | 1,539 tok/s | 4.7× |

<sub>\* `FFN N=6` 接続学習では 0 付近の境界値で float32 ULP 1ビット差（`max_diff = 5.72e-6`）により C99 が 1 件改善して `137/318`（PT `136/318`）となりました。他の4モデルは Orig 18・Diverse 318 ともに PyTorch と 100% 完全一致しました。</sub>

- **PyTorch では遅い LUT モデルが C99 で FP 対照の `2.74×` 高速になる理由**: PyTorch `hard` 推論では KV-cache 無し（`O(T^2)`）のまま浮動小数点 Walsh-Hadamard 演算を実行します。一方 C99 + KV-cache 下（Orig 18 内訳）では、`FP対照`（`66.75 ms`）が FFN に `43.80 ms`（65.6%）・Attention に `22.25 ms`（33.3%）を費やすのに対し、`model.pt`（`23.68 ms`）では FFN が **`1.18 ms`**（`LutLayer_0` WH `0.80 ms` + 離散64bit `LutLayer_1` 引表参照 & `FromLogic` `0.38 ms`）へと **37分の1** に縮小し、全体時間の **92.0%（`21.79 ms`）は FP Attention** が占めます（KV-cache 無効時は `1,892.8 ms/call`, `52.8 tok/s`）。
- **接続行列のコンパイル時消去と公式 Griffith C カーネル**: 接続学習ありモデルの重み行列をコンパイル時に静的 `uint16_t` 配線へ畳み込むことで、`N=6` は `11.70 MB → 2.06 MB` まで縮小しました（生成共有ライブラリは `49.6 KB`）。なお、ブロック0の純粋ブール 2層 `N=4` LUT スタックを公式 Griffith `deploy.cbackend.generate_c` で単独 C 化すると **`444.7 ns / block`**（4ブロック合計 `1.78 µs`、`ir.simulate` と 100/100 一致）で動作します。
- **`ToLogic` 二値化のトレードオフ**: `binarize_tologic=True` を指定すると第1層 LUT も純粋ブール参照になりますが、既に Attention が 92% を占めるためレイテンシ短縮は `0.35 ms/call`（`24.16 → 23.81 ms`, `4,209 tok/s`）にとどまり、一致率は Orig `10/18`・Diverse `109/318` に低下します。

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
    "memory": {"group_ages": {"word": "kids", "ages": [4, 7], "line": "my kids are 4 and 7"}},
}
prompt = input_for_call("group_ages", "respond", args, seed=0, environment_date="2026-11-21", wire_format="compact-v2")
with CompiledCModel("/tmp/pybot_c") as model:
    gen = model.generate(prompt, profile=True)
result = decode_output(gen["text"], "respond", input_arguments=args, wire_format="compact-v2")
print(result["return"])  # "Your youngest is 4."
```

## 達成範囲と限界

- 学習済み4モジュール・6呼び出し口の単一呼び出し評価（記録された呼び出し前メモリを入力）では高い一致率に達しましたが、全116呼び出し口の置換や複数ターン会話全体のロールアウト評価は本実験の範囲外です。
- 2 / 4 / 6 入力LUTの比較では、FFN側は `--ffn-n-inputs 4`（16エントリ・固定配線 `214/318`、接続学習あり `211/318`）が2入力（`201/318`）を上回る最良点となった一方、6入力は `O(2^N)` の計算・メモリ・パラメータ増大と `ste` 移行時の劣化により `136/318` に低下しました。また Attention まで LUT 化した構成（`L2-LogicAttn` / `L3-C` / `L2-Full`）は `n_inputs` を 2→4→6 に増やしても `Binarize` + XNOR 制約から `4/18`・`23〜36/318` に留まっています。詳しくは [RESULTS.md](../../experiments/moduleio/RESULTS.md) と [summary.json](../../experiments/moduleio/summary.json) を参照してください。
