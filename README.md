# Datascience2026_cgdetr

実践的データサイエンス演習用リポジトリ。CG-DETR をベースにした音声モーメント検索モデル。

> 事前学習・ファインチューニングのデータセットや特徴量は各自で配置してください。  
> ハイパーパラメータはいくつか調整中のため、必要に応じて `config.yml` を変更してください。

---

## ディレクトリ構成

```
Datascience2026_cgdetr/
├── config_pretraining.yml   # Stage1 設定（Clotho-Moment 事前学習）
├── config.yml               # Stage2 設定（CASTELLA ファインチューニング）
├── data/                    # アノテーション jsonl（配置済み）
│   ├── clotho_moment_{train,val,test}_release.jsonl
│   └── castella_{train,val,test}_release.jsonl
├── features/                # CLAP 特徴量（各自で配置）
│   ├── clotho-moment/clap/      # 音声特徴量 (.npz)
│   ├── clotho-moment/clap_text/ # テキスト特徴量 (.npz)
│   ├── castella/clap/
│   └── castella/clap_text/
├── src/                     # メインコード
│   ├── train.py
│   ├── evaluate.py
│   ├── refine_boundary.py   # C補正（境界後処理）
│   └── wbf_fusion.py        # アンサンブル
├── results_pretraining/     # Stage1 出力
└── results/                 # Stage2 出力（seed=2023）
```

---

## 実行環境

```bash
source /data/kokuryumaru/.venv/bin/activate
```

> すべての `python` コマンドはこの venv を有効化した状態で実行してください。

---

## Stage 1：事前学習（Clotho-Moment）

```bash
cd /data/kokuryumaru/Datascience2026_cgdetr/src

python train.py --config ../config_pretraining.yml
```

- 出力先：`results_pretraining/`
- ベストモデル：`results_pretraining/best.ckpt`（val R1@0.7 が更新されたときのみ保存）
- 学習スケジュール：200エポック、Warmup 10ep + Cosine Annealing LR

---

## Stage 2：ファインチューニング（CASTELLA）

### 単一 seed（seed=2023）

```bash
cd /data/kokuryumaru/Datascience2026_cgdetr/src

python train.py \
  --config ../config.yml \
  --resume ../results_pretraining/best.ckpt
```

- 出力先：`results/`（`config.yml` の `results_dir` で変更可能）
- ベストモデル：`results/best.ckpt`

### 複数 seed（5seed の例）

seed=2023 の学習完了後、残りの seed を順次実行します。

```bash
cd /data/kokuryumaru/Datascience2026_cgdetr

SEEDS=(1 11 101 2024)
RESUME=$(pwd)/results_pretraining/best.ckpt

for SEED in "${SEEDS[@]}"; do
  CONFIG_TMP=$(pwd)/config_seed${SEED}.yml

  # seed と results_dir だけ書き換えた一時 config を生成
  sed -e "s/^seed:.*/seed: ${SEED}/" \
      -e "s/^results_dir:.*/results_dir: results_seed${SEED}/" \
      config.yml > $CONFIG_TMP

  cd src
  python train.py --config $CONFIG_TMP --resume $RESUME
  cd ..
done
```

---

## 推論（テスト・val 提出ファイルの生成）

```bash
cd /data/kokuryumaru/Datascience2026_cgdetr/src

# val 評価
python evaluate.py \
  --config ../config.yml \
  --model_path ../results/best.ckpt \
  --split val

# test 推論
python evaluate.py \
  --config ../config.yml \
  --model_path ../results/best.ckpt \
  --split test
```

出力：`results/submission_{val,test}.jsonl`（`pred_saliency_scores` を含む）

複数 seed の場合は config と model_path を差し替えて同様に実行します。

```bash
cd /data/kokuryumaru/Datascience2026_cgdetr

for SEED in 1 11 101 2024; do
  cd src
  python evaluate.py \
    --config ../config_seed${SEED}.yml \
    --model_path ../results_seed${SEED}/best.ckpt \
    --split test
  cd ..
done
```

---

## 後処理①：C補正（境界補正）

モデルが出力する `pred_saliency_scores` を使い、Top-1 予測の境界を ±radius 秒の範囲で再探索します。  
再学習不要で適用できます。

```bash
cd /data/kokuryumaru/Datascience2026_cgdetr

for SEED in 2023 1 11 101 2024; do
  # seed=2023 は results/ を使用、それ以外は results_seed${SEED}/
  RESDIR=$([ "$SEED" = "2023" ] && echo "results" || echo "results_seed${SEED}")

  python src/refine_boundary.py \
    -i ${RESDIR}/submission_test.jsonl \
    -o ${RESDIR}/submission_test_refined.jsonl \
    --alpha 0.03 --beta 0.5 --radius 5
done
```

| オプション | 最適値 | 説明 |
|-----------|--------|------|
| `--alpha` | `0.03` | 移動量へのペナルティ（大きいほど元の境界に近い値を選ぶ） |
| `--beta`  | `0.5`  | 長さ変化へのペナルティ |
| `--radius`| `5`    | 探索範囲（秒） |

> **注意**：`--smooth_window` は使わないこと（-1.34pt の悪化を確認済み）。

---

## 後処理②：5seed Hybrid Top-1 アンサンブル

複数 seed の予測を統合します。座標は最高スコア候補をそのまま使用し、スコアを合意数で補正します（座標の加重平均は R1@0.7 を約 -7pt 悪化させるため使用しません）。

```bash
cd /data/kokuryumaru/Datascience2026_cgdetr

# C補正なし版
python src/wbf_fusion.py \
  -i results/submission_test.jsonl \
     results_seed1/submission_test.jsonl \
     results_seed11/submission_test.jsonl \
     results_seed101/submission_test.jsonl \
     results_seed2024/submission_test.jsonl \
  -o results/submission_test_5seed_hybrid.jsonl \
  --strategy hybrid --top_k_in 1 --iou_thr 0.5

# C補正あり版（refine_boundary.py 適用後のファイルを使用）
python src/wbf_fusion.py \
  -i results/submission_test_refined.jsonl \
     results_seed1/submission_test_refined.jsonl \
     results_seed11/submission_test_refined.jsonl \
     results_seed101/submission_test_refined.jsonl \
     results_seed2024/submission_test_refined.jsonl \
  -o results/submission_test_5seed_hybrid_refined.jsonl \
  --strategy hybrid --top_k_in 1 --iou_thr 0.5
```

---

## 後処理③：チームアンサンブル（任意）

`src/` モデルと `private/higashino/` モデル（CG-DETR 別実装）はアーキテクチャが異なるためアンサンブル効果が期待できます。

```bash
cd /data/kokuryumaru/Datascience2026_cgdetr

python src/wbf_fusion.py \
  -i results/submission_test_5seed_hybrid.jsonl \
     private/higashino/predictions/higashino_5seed_hybrid_top1.jsonl \
  -o results/submission_test_team_ensemble.jsonl \
  --strategy hybrid --top_k_in 1 --iou_thr 0.5
```

---

## スコア評価

提出ファイルに `pred_saliency_scores` が含まれている場合は、評価前に除去してください。

```bash
cd /data/kokuryumaru/Datascience2026_cgdetr

# saliency_scores を除去（評価ツールが未対応のため）
python3 -c "
import json
with open('results/submission_test_5seed_hybrid.jsonl') as f, \
     open('/tmp/eval_input.jsonl', 'w') as out:
    for line in f:
        d = json.loads(line)
        d.pop('pred_saliency_scores', None)
        out.write(json.dumps(d) + '\n')
"

# 評価
cd src
python -m standalone_eval.eval \
  --submission_path /tmp/eval_input.jsonl \
  --gt_path ../data/castella_test_release.jsonl \
  --save_path /tmp/metrics.json \
  --not_verbose

# 結果確認
python3 -c "
import json, pprint
pprint.pprint(json.load(open('/tmp/metrics.json'))['brief'])
"
```

---

## 全体フロー

```
features/ （CLAP特徴量）
    │
    ▼
[Stage 1] 事前学習
  python train.py --config config_pretraining.yml
    │ results_pretraining/best.ckpt
    ▼
[Stage 2] ファインチューニング × 5seed
  python train.py --config config.yml --resume ...
    │ results_seed*/best.ckpt
    ▼
[推論] evaluate.py × 5seed
    │ results_seed*/submission_test.jsonl  ← pred_saliency_scores 含む
    ▼
[C補正] refine_boundary.py × 5seed        ← alpha=0.03, beta=0.5, radius=5
    │ results_seed*/submission_test_refined.jsonl
    ▼
[5seed Hybrid Top-1] wbf_fusion.py        ← strategy=hybrid, top_k_in=1
    │ results/submission_test_5seed_hybrid_refined.jsonl
    ▼
[チームアンサンブル] wbf_fusion.py        ← + higashino predictions（任意）
    │
    ▼
[評価] standalone_eval.eval
```

---

## 注意点

- `train.py` / `evaluate.py` は `src/` ディレクトリから実行してください。スクリプト内部で `os.chdir()` によりプロジェクトルートに移動するため、`data/` や `features/` などの相対パスが正しく解決されます。
- config に書かれている `lighthouse-cgdetr` は旧ディレクトリ名です。実際のパスは `Datascience2026_cgdetr` です。
- `private/higashino/code/` は CG-DETR の別実装（Sim-DETR 等の実験コード）です。`src/` とは互換性がないため混在させないでください。詳細は `private/higashino/README.md` を参照してください。


-------------------------------------------------------------------------------------------------
## M2D-CLAPへの置き換え

共有した抽出済みの特徴量データ（ `m2d_features.zip` ）を使用する場合は、「追加パッケージ～特徴量抽出」の手順はスキップし、解凍したデータを配置するだけで使用可能です。

### M2D-CLAPの使用
- 取得したM2D-CLAP特徴量を任意の場所（ `Datascience2026_cgdetr/features/castella/m2d_clap` など）においてください。
- `config.yml` の `a_feat_type` 、 `t_feat_type` を `m2dclap` に変更してください。

### M2D-CLAPの作成：追加パッケージのインストール
```bash
pip install yt-dlp
pip install --upgrade timm
pip install sentence_transformers nnAudio
```

### M2D-CLAPリポジトリのcloneとチェックポイントのダウンロード
公開しているM2D-CLAPの学習済みモデルをダウンロードします。
```bash
# 自分の作業ディレクトリに変更してください
M2D_DIR=/data/your_username/m2d

git clone https://github.com/nttcslab/m2d.git $M2D_DIR
cd $M2D_DIR
wget https://github.com/nttcslab/m2d/releases/download/v0.5.0/m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip
unzip m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025.zip
```

### 音声の準備
```bash
# 自分の作業ディレクトリ
YOUR_DIR=/data/your_username
cd $YOUR_DIR
git clone https://github.com/h-munakata/CASTELLA-audio.git $YOUR_DIR
git clone https://github.com/line/CASTELLA.git $YOUR_DIR

# バックグラウンドでダウンロードを実行
nohup python CASTELLA-audio/script/download_audio.py CASTELLA/json/en/train.json > download_log.txt 2>&1 &

# 進捗確認
tail download_log.txt
```

### 特徴量抽出
ダウンロード完了後に実行してください。
```bash
python src/extract_m2d_features.py --split train
python src/extract_m2d_features.py --split val
python src/extract_m2d_features.py --split test
```
