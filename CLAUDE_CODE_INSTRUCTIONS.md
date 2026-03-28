# H200サーバー上での作業指示書 (Claude Code用)

## あなたの役割

あなたはH200 GPUサーバー上で動作するClaude Codeです。
このプロジェクトは二足歩行ロボットの脚設計の比較実験です。
3つの異なる脚モデル（通常膝 vs 逆関節 vs 両方向膝）をRLで訓練し、
エネルギー効率・安定性・ロバスト性を定量比較します。

## 絶対に守ること

1. **ホスト環境を汚さない。** 全ての作業はDockerコンテナ内で行う。
2. **ホスト上にパッケージをインストールしない。** `pip install`等はコンテナ内のみ。
3. **既存のファイル・ディレクトリを変更しない。** プロジェクトディレクトリ内でのみ作業する。
4. **GPU使用状況を確認してから訓練を開始する。** 他のユーザーのジョブがないか`nvidia-smi`で確認。

## 作業手順

### Phase 0: 環境確認

```bash
# GPU確認
nvidia-smi

# Docker確認
docker --version
docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# プロジェクトディレクトリ確認
ls -la ~/bipedal_leg_platform/
```

### Phase 1: コンテナ構築

```bash
cd ~/bipedal_leg_platform

# Dockerfileでビルド
docker build -t bipedal-rl -f Dockerfile .

# ビルドに失敗した場合:
# 1. Dockerfileのベースイメージを確認（nvcr.io/nvidia/isaac-sim:4.5.0が利用可能か）
# 2. 利用不可なら、代替ベースイメージを使用:
#    FROM nvidia/cuda:12.4.1-devel-ubuntu22.04
#    そしてIsaac Labをpipでインストール:
#    pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics
#    pip install isaaclab
# 3. Isaac Labが使えない場合、legged_gym + Isaac Gym (preview)で代替
# 4. 最終手段: MuJoCo + Gymnasium + Stable-Baselines3でCPU訓練（遅いが動く）
```

### Phase 2: コンテナ起動と検証

```bash
# GPU 0を使ってコンテナ起動
docker run --gpus '"device=0"' -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --shm-size=16g \
  --name bipedal-train-0 \
  bipedal-rl bash

# コンテナ内で検証
python scripts/validate_models.py
# → 3モデル全て ✅ が出ることを確認
```

### Phase 3: 段階的訓練

**重要: 必ずPhase 1訓練（簡易報酬）から始める。いきなりPhase 2に行かない。**

```bash
# Phase 1: 最小限の報酬で「歩く」を確認（各10-20分）
python scripts/train.py --model A --phase 1 --num_envs 4096 --max_iterations 2000
# → reward curveが上昇し、ロボットが前進することを確認
# → 確認方法: TensorBoardのreward plotを見る
# → もし歩かない場合: README.mdのトラブルシューティングを参照

# Phase 1が成功したら、Phase 2へ
python scripts/train.py --model A --phase 2 --num_envs 4096 --max_iterations 15000
python scripts/train.py --model B --phase 2 --num_envs 4096 --max_iterations 15000
python scripts/train.py --model C --phase 2 --num_envs 4096 --max_iterations 20000
```

### Phase 4: 並列実行（GPUが複数使える場合）

```bash
# 別ターミナルで別GPUのコンテナを起動
docker run --gpus '"device=1"' -it --rm \
  -v $(pwd):/workspace -w /workspace \
  --shm-size=16g --name bipedal-train-1 bipedal-rl bash

# 各コンテナで別モデルを訓練
# GPU 0: Model A
# GPU 1: Model B
# GPU 2: Model C
# GPU 3: パラメータ探索 or 予備
```

### Phase 5: 評価

```bash
# CoT比較（最重要）
python scripts/evaluate_cot.py --all_models

# 荷重テスト
python scripts/evaluate_payload.py --all_models --masses 5 10 20

# 外乱テスト
python scripts/evaluate_robustness.py --all_models --forces 50 100 150

# Model C固有: 膝方向分析
python scripts/analyze_knee_direction.py --model C
```

### Phase 6: 動画

```bash
python scripts/render_video.py --model A --output results/videos/model_a_walk.mp4
python scripts/render_video.py --model B --output results/videos/model_b_walk.mp4
python scripts/render_video.py --model C --output results/videos/model_c_walk.mp4
python scripts/render_video.py --model C --payload 20 --output results/videos/model_c_payload20kg.mp4
```

## フレームワーク選択の優先順位

H200上の環境によって使えるフレームワークが異なる可能性がある。
以下の優先順位で試すこと：

### 優先度1: Isaac Lab + rsl_rl（推奨）
- 最も高速（4096並列環境、GPU物理）
- legged_gymの後継、最新のAPI
- セットアップ: `pip install isaaclab`

### 優先度2: legged_gym + Isaac Gym (Preview) + rsl_rl
- Isaac Labが使えない場合の代替
- Isaac Gym Previewはlegacy softwareだが十分動く
- セットアップ: IsaacGymを手動ダウンロード → `pip install -e .`

### 優先度3: MuJoCo + MJX (JAX) + Gymnasium
- NVIDIA依存なし、H200のCUDAだけで動く
- GPU加速はJAX/MJX経由
- legged_gymほどの並列数は出ないが、十分な速度
- セットアップ: `pip install mujoco jax[cuda12] gymnasium`

### 優先度4: MuJoCo + Stable-Baselines3（最終手段）
- CPU主体、遅いが確実に動く
- 並列数を減らして(256-512)実行
- セットアップ: `pip install mujoco stable-baselines3 gymnasium`

## 報酬関数のチューニングが必要な場合

以下の順序で調整すること：

1. `forward_vel`の重みを上げる（1.0 → 2.0）— 前に進む動機を強化
2. `energy`の重みを下げる（-0.001 → -0.0005）— 動かないことへの誘惑を減らす
3. `alive`の重みを上げる（0.5 → 1.0）— 転倒ペナルティを強化
4. `torso_height_var`を調整 — 高さ変動の許容範囲を広げる

**重みを変更したら必ず記録すること。** 最終的にどの重みで3モデルを比較したかが重要。

## 成功基準

以下が達成できれば実験成功：

1. ✅ 3モデル全てが平地で安定歩行（目標速度±0.2m/s）
2. ✅ CoTの定量比較データ（Model B/CがModel Aより低いことを示す）
3. ✅ 荷重テストの結果（20kgまで）
4. ✅ 歩行動画3本以上
5. ✅ Model Cの膝方向分析（後ろ膝を選好することを示す）

## 結果の保存場所

```
results/
├── checkpoints/
│   ├── model_a_phase2_final.pt
│   ├── model_b_phase2_final.pt
│   └── model_c_phase2_final.pt
├── metrics/
│   ├── cot_comparison.json
│   ├── payload_test.json
│   ├── robustness_test.json
│   └── knee_direction_analysis.json
├── videos/
│   ├── model_a_walk.mp4
│   ├── model_b_walk.mp4
│   ├── model_c_walk.mp4
│   └── model_c_payload20kg.mp4
├── plots/
│   ├── cot_comparison.png
│   ├── training_curves.png
│   └── knee_direction_histogram.png
└── summary.md  ← 実験結果の要約（YC応募用データの元）
```
