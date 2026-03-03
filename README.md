# MIRAI: Metacognitive Intent-Reflective Action Integration Model
## Setup
```bash
git clone https://github.com/Azuma413/MIRAI.git
cd MIRAI
uv sync
uv pip install torch torchvision
uv run python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

## 階層型生成ロボット基盤モデル アーキテクチャ仕様書
### 1. アーキテクチャの基本理念
本アーキテクチャは、人間の認知プロセスにおける「System 2（意識的な熟考・プランニング）」と「System 1（無意識的な身体操作）」の分業を模倣します。
 - System 2（統合モデル）: 潜在空間内で「この行動をとれば世界はどうなるか」を再帰的にシミュレーションし、最適な**意図（潜在Action）**を練り上げます。
 - System 1（行動生成モデル）: 与えられた意図と現在の物理的制約に基づき、具体的な**身体運動（Action Chunk）**を生成します。

### 2. データ表現空間
 - 観測画像 (I)
 - 観測状態 (s): CLIP等の事前学習済みVision Encoderにより観測画像から抽出された、埋め込みベクトル
    - 現状態 (s_t)
    - 補助状態 (s_aux)
    - 予測状態 (s_pred)
    - 目標状態 (s_goal)
 - Action Chunk (A)
 - 潜在Action (z_a): 実際のAction Chunkを、圧縮した「意図」を表す埋め込みベクトル。
 - タスク指示テキスト (T)
 - サブタスク指示テキスト (T_sub)
 - 内部的思考 (z)
 - 確信度 (p_halt)

### 3. モジュール構成と仕様
**統合モデル (System 2)**: 言語推論能力を保持した単一のDenseなVLM（Qwen3.5-2B等を想定）をベースとします。自己回帰（AR）とMaskGIT的な双方向・一括予測のハイブリッド処理を行います。
 - 機能1: サブタスク分解
    - 入力: タスク指示テキスト (T)
    - 出力: サブタスク指示テキストの系列 (T_sub_1, T_sub_2, ... , T_sub_n) ，自己回帰
 - 機能2: タスク達成状態の予測
    - 入力: 現状態 (s_t)、サブタスク指示テキスト (T_sub)
    - 出力: タスク達成状態 (s_goal) ，一括生成
 - 機能3: 潜在Action生成
    - 入力: 内部的思考 (z)，サブタスク指示テキスト (T_sub)，潜在Action (z_a)，現状態 (s_t)，予測状態 (s_pred)，達成状態 (s_goal)
    - 出力: 更新されたz_a，一括予測，確信度 (p_halt, halt headによるスカラー出力)
 - 機能4: 次状態予測
    - 入力: 現状態 (s_t)，潜在Action (z_a)
    - 出力: 予測状態 (s_pred)，一括生成，内部的思考 (z)

1→2→3→4の順に繰り返す． 3の確信度が低いと4に移行．高ければその時点のz_aを最終潜在Actionとして出力．1→2→4→3の順でも良いかも．

3の確信度がかなり低い場合は1や2に戻るのもあり．

**行動生成モデル (System 1)**: Flow Matchingベースの小規模なTransformer。
 - 入力: ガウスノイズ、現在状態 (s_t)、補助的なサブ状態 (s_aux, プロプリオセプション，サブカメラ画像から取得した状態等)
 - 条件付け: 統合モデルから渡された最終潜在Action (z_a) を、Cross-AttentionやAdaLN-Zeroを用いて強力に条件付け。
 - 出力: Action Chunk (A)

将来的にはz_aを保持し，複数Step分利用することも考える (memoryVLAを参照)

### 4. 学習パラダイム
**Deep Supervisionによる探索の内面化**:
 - 学習時に機能2と3のループを固定回数（N回）展開（Unroll）します。
 - 最終出力だけでなく、ループの中間ステップの予測状態 s_pred や潜在Action z_a に対しても損失を計算し、ネットワーク全体に逆伝播させます。これにより、モデルは「失敗状態からどう軌道修正すべきか」という探索アルゴリズムそのものを重みの中に学習します。
 - 確信度 p_halt は，真のz_aと予測したz_aのcos類似度などで学習

**Co-trainingによる言語能力の維持**:
 - ロボットの軌道データ（状態遷移・行動）の学習バッチに、一般的なVQAや画像キャプション等の言語タスクデータを一定割合で混ぜて学習（Co-training）させ、VLMの汎用的な推論能力の忘却を防ぎます。

**Offline RLへの拡張**:
 - 下位のFlow Matchingモデルに対して、Offline RL（Diffusion-QL等）を適用することで、OODからでも復帰動作を生成できるようにします。

### 参考文献
- Action Chunking with Transformer
- Tiny Recursive Model
- FPO
- V-JEPA, I-JEPA
- Qwen3.5
- EO-1
- pi0
- AdaLM-Zero