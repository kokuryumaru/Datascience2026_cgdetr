# Datascience2026_cgdetr
実践的データサイエンス演習の用のリポジトリ
事前学習とファインチューニングのデータセット等は各自でへんこうしてください
また、ハイパーパラメータはいくつか適当に変更しているので調整お願いいたします

## 実行方法

### Stage 1: 事前学習（Clotho-Moment）

```bash
cd ../Datascience2026_cgdetr/src
python train.py --config ../config_pretraining.yml
```

### Stage 2: ファインチューニング（CASTELLA）

```bash
cd ../Datascience2026_cgdetr/src
python train.py --config ../config.yml --resume ../results_pretraining/best.ckpt
```

## 注意点

- `config.yml` / `config_pretraining.yml` 内のパスはすべて**プロジェクトルートからの相対パス**。`train.py` がconfig のあるディレクトリに自動で `os.chdir()` するため、`src/` 以下から実行しても正しく解決される。
- `data/` と `features/` はすでにプロジェクト内に配置済みのため、パスの変更は不要。
- configコメントに書かれている `lighthouse-cgdetr` というディレクトリ名は旧名称。実際のパスは `Datascience2026_cgdetr`。
