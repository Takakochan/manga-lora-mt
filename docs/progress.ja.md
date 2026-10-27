# 進捗まとめ

2026-10-27 時点の、実験の経過と結果をまとめます。手順そのものは [aws.ja.md](aws.ja.md)、結果の表は
[README](../README.md#results) にもあります。

## これまでの流れ

| 日付 | 内容 |
|---|---|
| 2026-10-23 | 実験コード一式（データ作成、Nebius での予測、LoRA の学習と予測、評価）を作成 |
| 2026-10-24 | Kimi-K3 のベースラインを実行。AWS の手順書と `scripts/aws_setup.sh` を追加 |
| 2026-10-27 | AWS で GPU インスタンスを用意し、Qwen3-1.7B と Qwen3-1.7B + LoRA の本実験を実行 |

## GPU 環境の選択

- Mac（Intel）には torch を入れられず、Ubuntu マシンの GPU は Intel 製のため、CUDA を前提とした
  `scripts/lora.py` は動きません。
- Google Colab（無料版の T4）でも本実験は動く見込みでしたが、切断や時間制限があるため、AWS EC2 を
  使うことにしました。Kimi-K3 は Nebius Token Factory の API で動かすので、GPU は不要です。
- インスタンスは `g6.2xlarge`（NVIDIA L4、GPU メモリ 24 GB）です。

## AWS の準備でやったこと

1. 使えるリージョンが1つに限られた project なので、すべてのリソースをそのリージョンに作成。
   手順書の `us-east-1` の例と「別のリージョンを試す」という記述を直しました。
2. GPU の上限（Running On-Demand G and VT instances）が 0 だったので、8 vCPU に引き上げを申請。
   申請はサポートケースに回され（`CASE_OPENED`）、同じ日のうちに承認されました。
3. 月 $100 の予算アラート（50% と 80% でメール通知）を作成。
4. キーペアと、自分の IP からの SSH だけを許可するセキュリティグループを作成。
5. インスタンスを起動。最初は g6.2xlarge と g5.2xlarge の在庫不足
   （`InsufficientInstanceCapacity`）が AZ ごとに続きましたが、AZ を指定せずに試し直して
   起動できました。

## つまずいた点と対処

| 症状 | 原因 | 対処 |
|---|---|---|
| `TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'` | transformers 5 で `warmup_ratio` が廃止された | `warmup_steps=0.05` に変更（0〜1 の小数は全ステップに対する割合として扱われる） |
| tmux の中で `uv: command not found` | uv の入る `~/.local/bin` が tmux の PATH に無い | 実行前に `export PATH=$HOME/.local/bin:$PATH` |
| `aws login` が「Profile 'default' is already configured」で止まる | `default` に古いアカウントのアクセスキーが残っていた | 別のプロファイル名で `aws login --profile <名前>` |
| 場所が変わると SSH できない | セキュリティグループが接続元の IP だけを許可している | 新しい IP を許可し、古い IP を削除 |

## 本実験の結果

209 ページすべて、5 fold の leave-one-book-out です。差はベースモデルとの chrF++ の差で、ページ単位の
対応ありブートストラップ（1000 回）の 95% 区間です。

| run | フォーマットエラー | chrF++ | BLEU | 秒/ページ | 差 |
|---|---|---|---|---|---|
| Qwen3-1.7B | 27 | 20.1 | 7.7 | 0.8 | |
| Qwen3-1.7B + LoRA | 13 | 24.1 | 10.6 | 1.4 | +4.0 [+1.7, +6.3] |
| Kimi-K3 | 0 | 35.6 | 16.0 | 0.9 | 未比較 |

| run | balloon_dream | boureisougi | rasetugari | tencho_isoro | tojime_no_siora |
|---|---|---|---|---|---|
| Qwen3-1.7B | 22.7 | 17.3 | 19.8 | 20.1 | 19.8 |
| Qwen3-1.7B + LoRA | 27.0 | 23.4 | 22.1 | 26.3 | 21.7 |
| Kimi-K3 | 38.7 | 35.9 | 33.0 | 36.0 | 34.4 |

- LoRA で chrF++ が 4.0 ポイント上がり、95% 区間は 0 を含みません。5 冊すべてで向上しました。
- フォーマットエラー（JSON が壊れる、行数が合わない）が 27 ページから 13 ページに半減しました。
- Kimi-K3 とはまだ約 11 ポイントの差があります。
- LoRA の予測が遅いのは、アダプタを重みに統合せずに `transformers` で生成しているためと考えられます。
  配信速度は Week 3 に vLLM で改めて測ります。

## 実行時間と費用

- LoRA の学習は 1 fold（155〜175 ページ、3 エポック）あたり約 1.6 分、5 fold で約 8 分でした。
- インスタンスを起動してから停止するまで（セットアップ、スモークテスト、本実験すべて）は約 26 分でした。
  GPU の料金は $1 程度です。
- 停止中もディスク（100 GB gp3）の料金が月 $8 前後かかります。

## 次にやること

1. Kimi-K3 の予測ファイル（`out/mt/preds/Kimi-K3/`）を評価する PC にそろえ、3 つを対応ありで比較する。
2. 比較実験（README の Plan の Week 2）: コンテキストなし（`--context False`）、rank 8 と 32、
   学習データ 25% と 50%、Qwen3-0.6B と 4B。
3. 誤りの分析（30 行を抽出して、名前、敬称、口調、効果音を確認）。
4. Week 3: vLLM（`--enable-lora`）での配信と、ページあたりの遅延と費用の測定。
5. 実験が終わったら、インスタンスを Terminate してディスクの料金を止める。
