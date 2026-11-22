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
- **FP 蒸留ウォームスタートの役割**: 未学習状態からの LUT 学習（`8/18`・`117/318`）はエンティティ文字列のバイト単位コピー回路（induction head）形成の鞍点で停滞しました。訓練データのみで学習した FP 対照モデル（`fp-control.pt`）からの `--init-from` と公式 Griffith `kd_kl_loss_seq` により、hard 離散化ギャップ無しで **15/18・201/318** まで引き上げられました。
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

## 達成範囲と限界

- 学習済み4モジュール・6呼び出し口の単一呼び出し評価（記録された呼び出し前メモリを入力）では高い一致率に達しましたが、全116呼び出し口の置換や複数ターン会話全体のロールアウト評価は本実験の範囲外です。
- 2 / 4 / 6 入力LUTの比較では、FFN側は `--ffn-n-inputs 4`（16エントリ・固定配線 `214/318`、接続学習あり `211/318`）が2入力（`201/318`）を上回る最良点となった一方、6入力は `O(2^N)` の計算・メモリ・パラメータ増大と `ste` 移行時の劣化により `136/318` に低下しました。また Attention まで LUT 化した構成（`L2-LogicAttn` / `L3-C` / `L2-Full`）は `n_inputs` を 2→4→6 に増やしても `Binarize` + XNOR 制約から `4/18`・`23〜36/318` に留まっています。詳しくは [RESULTS.md](../../experiments/moduleio/RESULTS.md) と [summary.json](../../experiments/moduleio/summary.json) を参照してください。
