# Bipedal Leg Platform — RL Comparison Experiment

## 目的

YC応募用のデモデータを作る。
「逆関節＋テンセグリティ膝＋弾性エネルギー貯蔵の脚モジュールは、通常の脚より効率的で安価に製造できる」
を定量的に示す。

## 背景（なぜこの実験をやるのか）

### 学術的根拠
- **MIT Haberland & Kim (2015)**: 後ろ膝（逆関節）は前膝より大多数の設計でエネルギー効率が良い。
  理由は股関節のトルク・動きが少なくて済むため。
- **Rubenson et al. (2011)**: ダチョウは人間より50%低い代謝コストで走る。
  主因はTMP関節（つま先付け根）での弾性エネルギー貯蔵（人間比+120%）。
- **Wen et al. (2025)**: テンセグリティ膝で膝モーターなしの歩行サイクルを実証。
  四節リンクの特異点ロックにより、バネの自己回復で歩行可能。

### ビジネス仮説
- 膝モーター削除 → アクチュエータ数17%削減 → BOM $1,000-$4,000/台削減
- つま先バネ（SEA）追加 → エネルギー効率大幅改善 → バッテリー寿命延長
- 両方向膝 → 歩行時は逆関節で効率的、作業時は前膝で人間的動作

---

## 比較する3モデル

| | Model A | Model B | Model C |
|---|---|---|---|
| ファイル | `model_a_forward_knee.xml` | `model_b_reverse_knee.xml` | `model_c_bidirectional_knee.xml` |
| 名前 | 通常膝（ベースライン） | 逆関節のみ | 両方向膝（提案手法） |
| 膝の方向 | 前のみ（0°〜120°） | 後ろのみ（0°〜120°） | 両方（-90°〜120°） |
| 膝モーター | **あり** | **なし（バネ stiffness=40）** | **なし（バネ stiffness=35）** |
| つま先バネ | なし | あり（stiffness=25） | あり（stiffness=25） |
| アクチュエータ数 | 12（6/脚） | 10（5/脚） | 10（5/脚） |
| 胴体質量 | 10kg | 10kg | 10kg |
| 総質量 | 22.84kg | 22.74kg | 22.92kg |
| 設計思想 | 人間型（標準ヒューマノイド） | ダチョウ型（MIT論文の検証） | 柴田提案（前後両用） |

### モデル設計の重要な違い

**Model B/Cの膝バネについて：**
- MuJoCoの`joint stiffness`で表現。物理的にはテンセグリティユニットの自己回復力に相当。
- `springref=5`（度）：伸展位から5度曲がった位置が自然長。伸展時にほぼロック。
- これにより膝モーターなしでも体重支持が可能（論文⑧の再現）。

**Model B/Cのつま先バネについて：**
- `toe_joint`に`stiffness=25`を設定。ダチョウのTMP関節に相当。
- 着地時に弾性エネルギーを貯蔵し、蹴り出し時に解放（論文②の知見）。
- `toe_act`（gear=30）で能動的な制御も可能（StaccaToe論文⑩の共駆動機構に相当）。

**Model Cの両方向膝について：**
- `range="-90 120"`：負が前膝（人間型）、正が後ろ膝（ダチョウ型）。
- RLに自由に学習させ、歩行時にどちらの方向を選択するか観察する。
- 仮説：歩行時は後ろ膝を、しゃがみ等のタスクでは前膝を自然に使い分ける。

---

## 報酬関数の設計

### 設計原則
1. **3モデル共通の報酬関数を使う**（公平な比較のため）
2. **エネルギーペナルティを含める**（CoTの差を訓練中に反映）
3. **過度に複雑にしない**（デバッグ可能性を維持）

### 報酬の構成

```
R_total = w1 * R_forward_vel      # 前進速度追従
       + w2 * R_alive              # 生存報酬
       + w3 * R_torso_upright      # 胴体水平維持
       + w4 * P_energy             # エネルギー消費ペナルティ（負）
       + w5 * P_joint_acc          # 関節加速度ペナルティ（負）
       + w6 * P_torso_height_var   # 重心上下動ペナルティ（負）
       + w7 * P_action_rate        # アクション変化率ペナルティ（負）
       + w8 * P_foot_slip          # 足滑りペナルティ（負）
```

### 各報酬項目の詳細

#### R_forward_vel（前進速度追従）
```python
target_vel = 1.0  # m/s
R_forward_vel = exp(-4.0 * (v_x - target_vel)^2)
```
- 目標速度に近いほど1.0に近づく指数型報酬
- 速すぎても遅すぎてもペナルティ
- w1 = 1.0（最重要報酬）

#### R_alive（生存報酬）
```python
R_alive = 1.0  # 毎ステップ定数
```
- 転倒せず立っていることへの報酬
- w2 = 0.5

#### R_torso_upright（胴体水平維持）
```python
R_torso_upright = dot(torso_up_vector, world_up_vector)
# 完全水平で1.0、傾くほど減少
```
- 「腰を水平に保つ」というKPIに直結
- w3 = 0.5

#### P_energy（エネルギー消費ペナルティ）
```python
P_energy = -sum(|torque_i * angular_vel_i|) for all actuators
```
- **これがCoTの差を生む核心的な報酬項目**
- Model B/Cは膝モーターがないので構造的にこの値が小さくなる
- w4 = 0.001（小さいが、長期的に蓄積して差を生む）
- **注意**: 大きくしすぎると「動かない」が最適解になる

#### P_joint_acc（関節加速度ペナルティ）
```python
P_joint_acc = -sum(joint_acceleration_i^2)
```
- 滑らかな動きを促進
- w5 = 2.5e-7

#### P_torso_height_var（重心上下動ペナルティ）
```python
P_torso_height_var = -(z - z_target)^2
# z_target = 0.85（初期高さ）
```
- 上下に弾みすぎる歩行を抑制
- w6 = 1.0

#### P_action_rate（アクション変化率ペナルティ）
```python
P_action_rate = -sum((action_t - action_{t-1})^2)
```
- 制御入力のジッター防止
- w7 = 0.01

#### P_foot_slip（足滑りペナルティ）
```python
P_foot_slip = -sum(|foot_vel_xy| * is_contact)
# 接地中の足の水平速度にペナルティ
```
- 自然な歩行の促進
- w8 = 0.5

### 重みのチューニング方針

上記の重みはlegged_gym / rsl_rlの標準的な値に基づく初期値。
以下の順序でチューニング：

1. まず`w1`（前進速度）と`w2`（生存）だけで「とにかく歩く」ポリシーを得る
2. `w4`（エネルギー）を追加して効率的な歩行に誘導
3. 残りのペナルティを追加して歩行品質を改善
4. 重みの感度分析（各重みを2倍/半分にして結果の変化を確認）

---

## 評価項目と計測方法

### 1. Cost of Transport (CoT)
```
CoT = E_total / (m * g * d)
```
- E_total: 全アクチュエータの仕事量の総和（∫|τ·ω|dt）
- m: ロボット総質量
- g: 重力加速度（9.81）
- d: 移動距離
- 計測条件: 平地、目標速度1.0 m/s、30秒間
- **期待結果**: Model B/C < Model A（MIT論文の予測通り）

### 2. 上部荷重耐性
- 胴体上にmass=5/10/20kgの追加質量を配置
- 同じ歩行タスクで転倒せず歩けるかを評価
- 成功率とCoTの変化を記録
- **期待結果**: 3モデルとも対応可能だが、Model B/CはCoT増加が小さい

### 3. 外乱ロバスト性
- 歩行中に横方向の力（50N, 100N, 150N）を0.1秒間印加
- 転倒せず回復できる最大力を記録
- **期待結果**: Model Cが最もロバスト（両方向の膝で回復手段が多い）

### 4. Model C固有: 膝方向の使い分け
- 訓練後のポリシーを再生し、膝関節角度の時系列を記録
- 歩行中に正（後ろ膝）と負（前膝）のどちらを使うか分析
- **期待結果**: 歩行ではほぼ後ろ膝を使用

---

## H200サーバーでの実行手順

### 前提条件
- H200 GPU × 4台利用可能（ただし1台で基本的に十分）
- Docker/コンテナ環境を使用（**ホスト環境は絶対に汚さない**）
- NVIDIA Container Toolkit がインストール済み

### Step 0: コンテナの準備

```bash
# プロジェクトをサーバーにコピー
scp -r bipedal_leg_platform/ user@h200-server:~/

# コンテナをビルド＆起動
cd ~/bipedal_leg_platform
docker build -t bipedal-rl -f Dockerfile .
docker run --gpus '"device=0"' -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  --name bipedal-train \
  bipedal-rl bash
```

### Step 1: モデル検証（5分）

```bash
python scripts/validate_models.py
# 3モデルとも ✅ が出ることを確認
```

### Step 2: 段階的訓練（GPU 0）

```bash
# Phase 1: まず前進速度+生存報酬だけで「歩く」を確認（各10-20分）
python scripts/train.py --model A --phase 1 --num_envs 4096 --max_iterations 2000

# 歩いたらPhase 2: 全報酬で本番訓練（各1-2時間）
python scripts/train.py --model A --phase 2 --num_envs 4096 --max_iterations 15000
python scripts/train.py --model B --phase 2 --num_envs 4096 --max_iterations 15000
python scripts/train.py --model C --phase 2 --num_envs 4096 --max_iterations 15000
```

### Step 3: 並列実験（GPU 0-3を活用する場合）

```bash
# GPU 0: Model A 本番訓練
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --model A --phase 2 &

# GPU 1: Model B 本番訓練
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --model B --phase 2 &

# GPU 2: Model C 本番訓練
CUDA_VISIBLE_DEVICES=2 python scripts/train.py --model C --phase 2 &

# GPU 3: パラメータ探索（エネルギー重みの感度分析）
CUDA_VISIBLE_DEVICES=3 python scripts/sweep_energy_weight.py &
```

### Step 4: 評価（各数分）

```bash
python scripts/evaluate_cot.py --all_models
python scripts/evaluate_payload.py --all_models --masses 5 10 20
python scripts/evaluate_robustness.py --all_models --forces 50 100 150
```

### Step 5: 動画レンダリング

```bash
python scripts/render_video.py --model A --output results/model_a_walk.mp4
python scripts/render_video.py --model B --output results/model_b_walk.mp4
python scripts/render_video.py --model C --output results/model_c_walk.mp4
python scripts/render_video.py --model C --payload 20 --output results/model_c_payload20kg.mp4
```

---

## ディレクトリ構造

```
bipedal_leg_platform/
├── README.md                          # このファイル
├── CLAUDE_CODE_INSTRUCTIONS.md        # H200上のClaude Code用指示書
├── Dockerfile                         # コンテナ定義
├── requirements.txt                   # Python依存関係
├── models/
│   ├── model_a_forward_knee.xml       # Model A: 通常膝（ベースライン）
│   ├── model_b_reverse_knee.xml       # Model B: 逆関節のみ
│   └── model_c_bidirectional_knee.xml # Model C: 両方向膝（提案手法）
├── configs/
│   ├── base_config.py                 # 共通設定（報酬関数含む）
│   ├── model_a_config.py              # Model A固有設定
│   ├── model_b_config.py              # Model B固有設定
│   └── model_c_config.py              # Model C固有設定
├── scripts/
│   ├── validate_models.py             # モデル検証
│   ├── train.py                       # 訓練メインスクリプト
│   ├── evaluate_cot.py                # CoT計測
│   ├── evaluate_payload.py            # 荷重テスト
│   ├── evaluate_robustness.py         # 外乱テスト
│   ├── sweep_energy_weight.py         # エネルギー重みの感度分析
│   └── render_video.py               # 動画レンダリング
└── results/                           # 訓練結果・評価データ（gitignore）
    ├── checkpoints/
    ├── metrics/
    └── videos/
```

---

## 技術リスク分析と対策

この実験には5つの主要リスクがある。優先度順に記載。
**H200上で作業する際は、ここを最初に読んで対策を事前に把握すること。**

### リスク1（最大）：Model B/Cが「歩かない」

**問題の本質：**
膝モーターなし（バネのみ）で二足歩行が成立するかは、論文⑧（Wen et al. 2025）で
低速歩行のみ実証されている。MuJoCoの`joint stiffness`によるバネ近似が
実際のテンセグリティ構造と同じ挙動を示す保証はない。

**具体的な失敗モード：**
- バネ stiffness が高すぎる → 膝が曲がらず棒立ちで転倒
- バネ stiffness が低すぎる → 膝が折れて崩壊
- 股関節だけで膝を間接制御する連携をRLが発見できない
- 逆関節の脚振り出し（swing phase）で足が地面に引っかかる

**対策（優先度順）：**
1. **バネ定数sweepを最優先で実行。** stiffness=10,20,30,40,50,60,80 の7パターンを
   GPU 1台で並列実行（各Phase 1のみ、各10-20分、合計2時間程度）。
   「歩ける stiffness の範囲」を特定してからPhase 2に進む。
2. **springref（自然角度）も同時にsweep。** 0°, 5°, 10°, 15°の4パターン。
   stiffness × springref の組み合わせで最適領域を探索。
3. **damping も調整候補。** 現在の1.0が高すぎる可能性あり。0.3, 0.5, 1.0, 2.0を試す。
4. **最終手段：半パッシブ膝モデルを用意する。**
   完全に歩かない場合、膝に弱いアクチュエータ（gear=20、本来の120の1/6）を追加した
   Model B'/C'を作る。ストーリーは「膝モーターの出力を83%削減できる」に修正。
   完全削除より弱いが、コスト削減の主張は維持可能。
   MJCF修正方法：Model B/Cの`<actuator>`セクションに以下を追加：
   ```xml
   <motor name="r_knee_assist" joint="r_knee" gear="20" ctrlrange="-1 1"/>
   <motor name="l_knee_assist" joint="l_knee" gear="20" ctrlrange="-1 1"/>
   ```

**判断基準：**
- Phase 1で2000イテレーション後に前進距離 > 0.5m → 成功、Phase 2へ
- 前進距離 < 0.1m → パラメータ調整が必要
- 全パラメータで前進距離 < 0.1m → 半パッシブ膝に切り替え

---

### リスク2（高）：報酬関数のバランス崩壊

**問題の本質：**
二足歩行のRL訓練は報酬設計に極めて敏感。重みのバランスが崩れると
学習が収束しないか、望ましくない挙動が最適解になる。

**具体的な失敗モード：**
- **「動かないのが最適」問題：** energy ペナルティが大きすぎて、
  立ったまま微動だにしないのが最高報酬になる。
  兆候：reward は高いが forward_vel 成分がほぼ0。
- **「跳ねる」問題：** forward_vel 報酬が大きすぎて、歩行ではなく
  連続ジャンプで前進する不自然な歩容が生まれる。
  兆候：torso_height_var が非常に大きい、足の接地時間が極端に短い。
- **「一歩目で転倒」問題：** 初期姿勢からの最初の一歩が最も難しく、
  Phase 1ですら立つことができない。
  兆候：episode length が極端に短い（< 50ステップ）。
- **「足踏み」問題：** 転倒を避けるために足踏みだけして前進しない。
  兆候：alive 報酬は高いが distance ≈ 0。

**対策（段階的に）：**
1. Phase 1では energy=0, foot_slip=0 にしてある。まず「とにかく前に進む何か」を学習。
2. Phase 1で歩かない場合の調整順序：
   - alive の重みを 0.5 → 1.0 → 2.0 に上げる（「立ってるだけで偉い」）
   - forward_vel の重みを 1.0 → 2.0 に上げる
   - torso_height_var のペナルティを -0.5 → -0.2 に下げる（高さ変動を許容）
   - init_noise を 0.02 → 0.005 に下げる（初期条件を安定させる）
3. Phase 1で歩けたらPhase 2に移行。Phase 2で品質が下がる場合：
   - energy を -0.001 → -0.0005 → -0.0002 と段階的に追加
   - action_rate を -0.01 → -0.005 に下げる
4. **重みを変更したら必ず `run_config.json` に記録する。**
   最終的にどの重みで3モデルを比較したかが再現性のために必須。

**判断基準：**
- 3モデル全てが同じ報酬重みで歩行できる → そのまま比較
- Model Aは歩くがB/Cが歩かない → リスク1の対策を先に実施
- 全モデル歩かない → 報酬関数の根本的な見直しが必要（下記参照）

**報酬関数の根本的見直しが必要な場合：**
unitree_rl_gym の g1_config.py の報酬重みをそのまま使う方針に切り替える。
あのconfigは実機転移まで検証済みなので、二足歩行で動くことが保証されている。
カスタム脚モデルに合わせて関節名だけ変更すれば使える。

---

### リスク3（中）：Model A vs B/Cの比較が「不公平」に見える

**問題の本質：**
MJCFモデル設計で意図せず差を入れてしまい、CoTの差が脚設計ではなく
他の要因に起因する可能性がある。

**具体的な懸念点：**
- **足の形状が違う：** Model A は box型足（平ら）、Model B/C は capsule型足
  （曲面）+ 分離したtoe。接地面積・接地安定性が異なる。
- **質量配分が違う：** Model B/C は大腿が重く（3.8kg vs 3.0kg）、
  脛が軽い（1.2kg vs 2.0kg）。これは意図的（ダチョウの近位質量集中）だが、
  重心位置の差が歩行ダイナミクスに影響する。
- **アクチュエータ数が違う：** Model B/C は10個、Model Aは12個。
  エネルギーペナルティの計算で構造的に有利。これは意図通りだが、
  「モーターが少ないから当然」という反論が可能。

**対策：**
1. 質量配分の差は「逆関節設計のパッケージ全体」として報告する。
   MIT論文も質量配分を含めた設計空間で比較している。
2. **感度分析として追加実験を検討：**
   - Model A の質量配分を B/C に揃えたバージョン（足の形は box のまま）
   - Model B に膝アクチュエータを追加したバージョン（バネ + モーター併用）
   これで「質量配分の効果」と「膝モーター削除の効果」を分離できる。
3. エネルギーペナルティについては、CoTの計算でアクチュエータ数に
   正規化しない（総エネルギーで比較する）ことで公平性を担保。
   「少ないモーターで同じ距離を歩ける＝効率的」が主張。
4. **結果の報告では、差の原因を正直に分析する。**
   「CoT改善の内訳：X%が膝モーター削除、Y%が質量配分、Z%が弾性エネルギー貯蔵」
   のように分解できれば最も説得力が高い。

---

### リスク4（中）：Model Cが前膝を使わない

**問題の本質：**
歩行タスクでは後ろ膝の方が効率的（MIT論文の結論）なので、
RLが前膝モードを使わないのは技術的に正しい結果。
しかしYCのデモで「両方向に曲がる」を見せたい場合、
歩行動画だけでは片方向しか使わない。

**対策：**
1. **歩行タスクで後ろ膝のみ使用 → これ自体が重要な結果。**
   「RLが自由に学習した結果、自然に後ろ膝を選択した」は
   MIT論文の知見をRL訓練で独立に再現したことを意味する。
2. **前膝モードのデモは別タスクで作る（余裕がある場合）：**
   - 「しゃがんで地面のものを拾う」タスク → 前膝が必要
   - 「低い天井の下を通過する」タスク → 前膝が必要
   - これらは追加の報酬関数設計が必要なので、Day 3の余裕がある場合のみ
3. **YCのピッチでは以下のように説明する：**
   「歩行効率の比較実験で、RLは自由に膝方向を選べる設計にしたが、
   自然に後ろ膝（ダチョウ型）を選択した。これはMIT論文の知見と一致する。
   将来的にはタスクに応じて前膝モードに切り替えるMoE制御を実装する。」

---

### リスク5（低〜中）：フレームワークの環境構築失敗

**問題の本質：**
Isaac Lab / legged_gym のDockerセットアップで依存関係が噛み合わない可能性。
特にNVIDIA Container ToolkitのバージョンやCUDAバージョンの不一致。

**フォールバック順序（CLAUDE_CODE_INSTRUCTIONS.md にも記載）：**

| 優先度 | フレームワーク | 訓練速度 | セットアップ難易度 |
|---|---|---|---|
| 1 | Isaac Lab + rsl_rl | 最速（4096並列） | 高 |
| 2 | legged_gym + Isaac Gym Preview | 速い（4096並列） | 中 |
| 3 | MuJoCo + MJX (JAX) | 中程度（GPU加速） | 中 |
| 4 | MuJoCo + Stable-Baselines3 | 遅い（CPU主体、64並列） | 低 |

**各フレームワークの注意点：**
- **Isaac Lab:** `nvcr.io/nvidia/isaac-sim:4.5.0` イメージが利用可能か確認。
  不可の場合、pip install `isaaclab` を試す。Isaac Sim 5.x はまだ不安定な場合あり。
- **legged_gym:** Isaac Gym Preview はlegacyソフトウェアで NVIDIA公式ダウンロードが必要。
  コンテナ内でダウンロードURLが有効か確認。
- **MuJoCo + MJX:** JAXのCUDA対応版が必要。`pip install jax[cuda12]`。
  H200のCUDA 12.xで動くはず。
- **SB3:** 確実に動くが、num_envs が64程度に制限される。
  訓練時間が10-100倍かかる。Phase 1の検証用としては十分。

**判断基準：**
- 各フレームワークのセットアップに30分以上かかる場合、次のオプションに移る
- 最低限SB3が動けば実験自体は可能（時間はかかるが結果は出る）

---

### 最悪シナリオへの対応計画

**全モデルが全く歩かない場合（確率: 低）：**
1. unitree_rl_gym の G1 config をそのまま使い、G1が歩くことをまず確認
2. G1のMJCFを少しずつ改造してカスタム脚に近づける（漸進的アプローチ）
3. 改造の各ステップで歩行が維持されることを確認しながら進める

**Model B/Cだけが歩かない場合（確率: 中）：**
1. 半パッシブ膝（リスク1の対策4）に切り替え
2. 膝モーターの gear を 120→20 に下げたモデルで比較
3. ストーリーを「膝モーター出力83%削減」に修正

**時間が足りない場合（確率: 中）：**
1. Day 1で Model A vs B のPhase 2比較だけ完了させる（最低限のデータ）
2. Model C は Phase 1結果のみでもYCのビジョンとしては提示可能
3. 動画は最低1本（最も良い結果のモデル）あればYC応募は可能

---

## トラブルシューティング

### 「歩かない」場合
1. Phase 1（前進+生存のみ）で歩くか確認
2. 歩かない → `R_alive`の重みを上げる、`P_energy`の重みを下げる
3. 転倒する → 胴体の`healthy_z_range`を広げる、`R_torso_upright`の重みを上げる
4. その場で足踏み → `R_forward_vel`の重みを上げる

### Model B/Cの膝が動かない場合
1. バネの`stiffness`が高すぎる可能性 → 半分に下げる（40→20, 35→17）
2. `springref`の値を調整（5→10にすると自然長がより曲がった位置になる）
3. 膝の`damping`を下げる（1.0→0.5）

### Model Cが前膝を使わない場合
- 歩行タスクでは後ろ膝だけ使うのが正常（それ自体が結果）
- 前膝テストは別タスク（しゃがみ指令）で実施

### 訓練が不安定な場合
1. `num_envs`を2048に下げる
2. learning rateを半分にする（3e-4 → 1.5e-4）
3. `max_iterations`を増やす（15000 → 30000）

---

## 期待される結果（仮説）

| メトリクス | Model A | Model B | Model C |
|---|---|---|---|
| CoT（低いほど良い） | 1.0 (baseline) | 0.6-0.8 | 0.6-0.8 |
| 20kg荷重時のCoT増加率 | +30-50% | +15-30% | +15-30% |
| 最大耐外乱力 | 80-120N | 80-120N | 100-150N |
| アクチュエータ数 | 12 | 10 | 10 |
| 推定製造コスト削減 | baseline | -$1,000〜$4,000 | -$1,000〜$4,000 |

これらの仮説が実験で支持されれば、YC応募の核心的なデータになる。

---

## ライセンス・参考文献

### 使用するOSSライブラリ
- Isaac Gym / Isaac Lab (NVIDIA, BSD-3)
- legged_gym (ETH Zurich, BSD-3)
- rsl_rl (ETH Zurich, BSD-3)
- MuJoCo (Google DeepMind, Apache 2.0)
- unitree_rl_gym (Unitree, BSD-3)

### 主要参考論文
1. Haberland & Kim (2015) - 膝方向と二足走行効率
2. Rubenson et al. (2011) - ダチョウvs人間の関節力学
3. Wen et al. (2025) - テンセグリティ膝
4. Radosavovic et al. (2024) - 実世界ヒューマノイド歩行とRL
5. Rudin et al. (2022) - Learning to Walk in Minutes (legged_gym)
