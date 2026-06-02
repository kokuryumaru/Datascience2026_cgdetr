# higashino branch (feature/higashino-develop)

> 2026-05-29 時点での higashino の改良結果と、チームアンサンブル用予測ファイル。

> [!IMPORTANT]
> **アーキテクチャの違いに注意**:
> - team リポ (`Datascience2026-d`) の `src/` = **QD-DETR ベース**
> - higashino の改良 = **CG-DETR ベース**(`private/higashino/code/` 配下に分離保管)
>
> higashino の CG-DETR コードは **team の `src/`(QD-DETR)とは別アーキ**なので、上書きせず `private/higashino/code/` に置いている。
> チームアンサンブル時には **higashino の予測 jsonl を `private/higashino/predictions/` から取得**して融合する。

## 達成スコア(CASTELLA test、CG-DETR ベース)

| 構成 | R1@0.5 | **R1@0.7** | mAP |
|---|---|---|---|
| CG-DETR ベース(Lighthouse 派生) | 33.93 | 20.86 | 16.89 |
| + B班 Saliency 修正(saliency全relevant_windows) | 34.74 | 22.49 | 17.40 |
| + C班 saliency 境界補正(α=0.03) | 35.34 | 23.90 | 18.33 |
| **+ 5seed hybrid Top-1 + C補正** | **36.38** | **25.39** | 17.14 |

→ **higashino 単独最終: R1@0.7 = 25.39**(ベース +4.53pt)

## ファイル構成(higashino branch)

```
Datascience2026-d/  (team リポ)
├── src/
│   ├── refine_boundary.py    ★ higashino 追加(汎用、QD-DETRでも使える)
│   ├── wbf_fusion.py         ★ higashino 追加(汎用、アンサンブル用)
│   └── (既存の QD-DETR コード、変更なし)
└── private/higashino/
    ├── README.md             (本ファイル)
    ├── code/                 ★ CG-DETR 用改良コード(参考)
    │   ├── dataset.py        B班修正版(複数window対応)
    │   ├── evaluate.py       pred_saliency_scores を submission に保存
    │   ├── cg_detr.py        Sim-DETR(QGR+GLB)追加版(検証のみ、本番未採用)
    │   ├── cg_detr_transformer.py  Sim-DETR QGR 追加(同上)
    │   └── train.py          Cosine LR スケジューラ追加(検証のみ、本番未採用)
    ├── config/
    │   ├── config_higashino.yml   higashino の Stage2 FT 基本設定
    │   └── config_planB_cosine.yml Plan B(Cosine LR、検証用)
    ├── scripts/
    │   ├── multi_seed_train.sh    B班修正版 4 seed 順次学習
    │   └── multi_seed_planB.sh    Plan B 4 seed 順次学習(未実行)
    └── predictions/          ★ チームアンサンブル用予測
        ├── higashino_seed2023.jsonl   B班修正、seed=2023
        ├── higashino_seed1.jsonl      seed=1
        ├── higashino_seed11.jsonl     seed=11
        ├── higashino_seed101.jsonl    seed=101
        ├── higashino_seed2024.jsonl   seed=2024
        └── higashino_5seed_hybrid_top1.jsonl  5seed 統合済(C補正前)
```

## 採用した改良(本番=25.39 達成構成)

### 1. B班 Saliency 修正(=実装の見落とし修正)
- 場所: `private/higashino/code/dataset.py` の `get_saliency_labels_sub_as_query`
- 変更: 単一window前提だった関数を、複数window対応に拡張
- 理由: CASTELLA は 1音声に最大5個 relevant_windows あるのに、ベースラインは先頭1個のみ使用していた(=学習信号の80%損失)
- 効果(CG-DETR上): 20.86 → 22.49 (+1.63pt)

### 2. C班 saliency 境界補正(=推論時の後処理、再学習不要)
- 場所: `src/refine_boundary.py`(team リポにも追加)
- 原理: モデルが内部で計算する saliency_scores を後処理で使い、Top-1 予測の境界を ±5秒範囲で再探索
- スコア式: `contrast - α*move_penalty - β*length_penalty`
- 最適ハイパラ: **α=0.03, β=0.5, radius=5**(grid search で確定)
- 効果(CG-DETR上): 単独 → +2.67pt
- **QD-DETR でも使える汎用ツール**(saliency_scores を出力するモデルなら適用可)

### 3. 5seed hybrid Top-1 アンサンブル
- 場所: `src/wbf_fusion.py`(team リポにも追加、`--strategy hybrid --top_k_in 1`)
- 戦略: **座標は最高スコア cand のものを採用、スコアは合意度で集計**
- **重要発見**: 元の WBF(=座標加重平均)は **moment retrieval で逆効果 -7pt**。座標を平均すると IoU閾値 0.7 を割って大幅悪化。
- 効果(CG-DETR上): 5seed 統合で R1@0.7 = 23.90 → 25.39 (+1.49pt)

## 試したが採用しなかった改良(失敗事例)

| 改良 | 結果 | 教訓 |
|---|---|---|
| Savitzky-Golay 平滑化 | -1.34pt 悪化 | CG-DETR の saliency は既にスムーズ、追加平滑化は情報損失 |
| Sim-DETR (QGR + GLB) | -1.93pt 悪化 | 論文値 +3.5pt(QVHighlights VMR)はドメイン依存、CG-DETR では逆効果 |
| Plan B (Cosine LR Annealing) | -1.78pt 悪化 | StepLR 一定で既に最適化済み、追加減衰が過剰最適化を崩す |

**根本パターン**: 「CG-DETR + B班修正 + CASTELLA」は既によくチューニングされており、論文値ベースの追加改良は**過剰最適化**を崩す方向に働きがち。

## チームアンサンブルへの提案

higashino の予測は **CG-DETR ベース**、team の標準は **QD-DETR**。**アーキ多様性** が確保されているので、アンサンブル効果が出やすい。

### 推奨融合戦略(`src/wbf_fusion.py` を使う場合)

```bash
# higashino の単独最終 + 他メンバーの予測を hybrid Top-1 で統合
python src/wbf_fusion.py \
  -i private/higashino/predictions/higashino_5seed_hybrid_top1.jsonl \
     private/<other_member>/predictions/<theirs>.jsonl \
     ... \
  -o private/team_ensemble/team_final.jsonl \
  --strategy hybrid --top_k_in 1 --iou_thr 0.5
```

### 重要: 座標加重平均 (WBF) はモーメント検索で逆効果

```
# ❌ これは使わない(座標がボヤけて R1@0.7 が大幅悪化)
python src/wbf_fusion.py ... --strategy wbf

# ✓ これを使う(hybrid Top-1、座標は最高スコア cand のまま、スコアは合意度)
python src/wbf_fusion.py ... --strategy hybrid --top_k_in 1
```

### さらに精度を上げたい場合: C補正を上乗せ

```bash
# (1) 統合前: 各メンバーの予測に C補正を適用してから統合
python src/refine_boundary.py \
  -i private/higashino/predictions/higashino_seed2023.jsonl \
  -o /tmp/higashino_seed2023_refined.jsonl \
  --alpha 0.03 --beta 0.5 --radius 5

# (2) 各メンバーの refined を hybrid Top-1 で統合
python src/wbf_fusion.py \
  -i /tmp/higashino_seed2023_refined.jsonl \
     /tmp/<other>_refined.jsonl \
     ... \
  -o final_team_ensemble.jsonl \
  --strategy hybrid --top_k_in 1
```

ただし、各メンバーのモデルが `pred_saliency_scores` を出していないと C補正は適用できない(saliency依存)。
QD-DETR は saliency_scores を出すか確認必要(=B班、C班の元発見は QD-DETRベース)。

## 評価方法

```bash
# 単独評価(saliency 除去してから)
python -c "
import json
with open('private/higashino/predictions/higashino_5seed_hybrid_top1.jsonl') as f, \
     open('/tmp/clean.jsonl', 'w') as out:
    for line in f:
        d = json.loads(line)
        d.pop('pred_saliency_scores', None)
        out.write(json.dumps(d) + '\n')
"

python -m standalone_eval.eval \
  --submission_path /tmp/clean.jsonl \
  --gt_path data/castella_test_release.jsonl \
  --save_path /tmp/metrics.json --not_verbose

cat /tmp/metrics.json | python -m json.tool | head -10
```

## チェックポイント(best.ckpt)について

GitHub の 100MB 制限のため、**`best.ckpt`(各145MB × 5seed = 725MB)は本リポには含めない**。
チームアンサンブルには submission jsonl で十分。

ckpt が必要な場合は、サーバー `tlab002:/home/higashino/Datascience2026_cgdetr/results_v2_btfix*/best.ckpt` に保管。

## 参考文献

- Moon+ 2023 [CG-DETR (arXiv:2311.08835)](https://arxiv.org/abs/2311.08835)
- B班(同コンペ内)Saliency 全relevant_windows 修正の発見
- C班(同コンペ内)saliency_score 境界補正の発見
- Solovyev+ 2021 [Weighted Box Fusion (arXiv:1910.13302)](https://arxiv.org/abs/1910.13302) — WBFの原典(本実装は moment retrieval 向けに hybrid Top-1 に改造)

## 学習データセット

- 事前学習: Clotho-Moment(`data/clotho_moment_*_release.jsonl`)
- ファインチューニング: CASTELLA(`data/castella_*_release.jsonl`)
- データは team リポと共通(`data/` 配下)
