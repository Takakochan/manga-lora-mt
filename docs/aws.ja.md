# AWS での GPU 実験の実行手順

[English](aws.md)

`scripts/lora.py` による Qwen3 の学習と推論には NVIDIA GPU が必要です。この手順では AWS EC2 の
GPU インスタンス1台で実行し、予測結果を `scripts/eval.py` 用に手元へ戻します。Kimi-K3 の実行と
評価には GPU は不要で、どの PC でも実行できます。

## 1. 事前準備（初回のみ）

1. **GPU クォータの申請。** 新しい AWS アカウントでは、クォータを上げるまで GPU インスタンスを
   起動できません。AWS コンソールで **Service Quotas → Amazon EC2 → Running On-Demand G and VT
   instances** を開き、**8**（単位は vCPU）を申請します。クォータはリージョンごとなので、使う
   リージョン（例: `us-east-1`）で申請してください。承認には数時間〜数日かかることがあります。
2. **クレジットの確認。** **Billing → Credits** で、クレジットが Amazon EC2 に使えることを
   確認します。
3. **キーペアの作成。** **EC2 → Key pairs** でキーペア（`.pem`）を作成し、接続元の PC に保存
   します。その後 `chmod 400 your-key.pem` を実行します。

## 2. インスタンスの起動

**EC2 → Launch instance** で以下を設定します。

| 項目 | 値 |
|---|---|
| AMI | PyTorch 入りの Ubuntu 22.04 **Deep Learning AMI**（"Deep Learning AMI" で検索）。NVIDIA ドライバー込み |
| インスタンスタイプ | `g6.2xlarge`（L4、GPU 24 GB、RAM 32 GB）。`g5.2xlarge`（A10G）でも可 |
| キーペア | 手順1で作成したもの |
| ネットワーク | SSH は **My IP** からのみ許可 |
| ストレージ | 100 GB gp3 |

`VcpuLimitExceeded` で失敗した場合は、クォータがまだ承認されていません。
`InsufficientInstanceCapacity` の場合は、別のアベイラビリティゾーンかリージョンを試してください。

## 3. 接続とセットアップ

```
ssh -i your-key.pem ubuntu@<インスタンスのパブリック DNS>
tmux new -s lora
```

SSH が切れても処理が続くよう、`tmux` の中で作業します。再接続は `tmux attach -t lora` です。

リポジトリをクローンします。非公開リポジトリの場合は、`gh auth login` を実行するか、GitHub の
個人アクセストークンをパスワードとして HTTPS でクローンしてください。

```
git clone https://github.com/Takakochan/manga-lora-mt.git
cd manga-lora-mt
scripts/aws_setup.sh
```

`scripts/aws_setup.sh` は GPU の確認、uv のインストール、`finetune` extra の同期、データの
ダウンロード、短いスモークテスト学習を行います。最後に `smoke test passed` と表示されれば成功です。

## 4. 実験の実行

```
uv run scripts/lora.py predict                               # Qwen3-1.7B（ベース）、全5分割
uv run scripts/lora.py train                                 # アダプター5個
uv run scripts/lora.py predict --adapters Qwen3-1.7B-lora    # Qwen3-1.7B + LoRA
```

学習中に GPU メモリが足りなくなった場合は、`train` に `--batch-size 1 --grad-accum 8` を追加します。

README の Plan にあるアブレーションも同じように実行できます。例:

```
uv run scripts/lora.py train --rank 8 --run Qwen3-1.7B-lora-r8
uv run scripts/lora.py predict --adapters Qwen3-1.7B-lora-r8
```

## 5. 結果を手元に戻す

評価を行う PC のリポジトリ内で実行します。

```
scp -i your-key.pem -r ubuntu@<パブリック DNS>:manga-lora-mt/out/mt/preds/ out/mt/
scp -i your-key.pem -r ubuntu@<パブリック DNS>:manga-lora-mt/out/mt/adapters/ out/mt/
```

アダプターは、後で再利用する場合（vLLM での配信など）にのみ必要です。

## 6. 評価

`eval.py` を実行する PC の `out/mt/preds/` に、すべての実行の予測が揃っている必要があります。
Kimi-K3 の予測は git に含まれていないため、実行した PC から `out/mt/preds/Kimi-K3/` をコピーするか、
`.env` に `NEBIUS_API_TOKEN` を設定して再生成します。

```
uv run scripts/data.py
uv run scripts/predict_api.py
```

その後:

```
uv run scripts/eval.py Qwen3-1.7B,Qwen3-1.7B-lora,Kimi-K3
```

## 7. 課金を止める

- 作業を中断するときはインスタンスを **Stop** します。停止中は GPU の料金はかかりませんが、
  100 GB のディスク料金は発生します。
- 作業が終わり、結果をコピーしたら **Terminate** します。ディスクとその中身はすべて削除されます。
- 最初の作業の後に **Billing → Bills** を確認し、料金がクレジットから引かれていることを確かめて
  ください。
