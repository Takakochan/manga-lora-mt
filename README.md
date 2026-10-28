# manga-lora-mt

Does a small Qwen model, LoRA-tuned on manga dialogue, translate Japanese manga pages better
than the same model untuned, and how close does it get to a large hosted model (Kimi-K3)?

The setting comes from Fukidashi, a manga
translation app I worked on in a team. This repository is a separate, individual experiment
and contains no Fukidashi code. It reuses one idea from the app: translate a page together with
the previous page's text as context.

## Design

- Task: translation only. The input is the gold Japanese lines of a page plus the previous
  page's Japanese lines. The output is JSON with one translation per line. Using gold text keeps
  detection and OCR errors out of the comparison.
- Data: [OpenMantra](https://github.com/mantra-inc/open-mantra-dataset), 5 books, 209 pages,
  1592 lines with professional English translations. Evaluation is leave-one-book-out: 5 folds,
  each trains on 4 books and tests on the unseen one, so every page is scored exactly once by a
  model that never saw its book.
- Hyperparameters are fixed before looking at test scores (rank 16, alpha 32, lr 2e-4, 3 epochs,
  all linear layers). Greedy decoding everywhere.
- Metrics: chrF++ (main) and BLEU over all lines, format errors (invalid JSON or wrong line count
  score as empty), seconds per page. Each run is compared to a baseline with a paired bootstrap
  over pages (1000 resamples, 95% interval).

| run | what |
|---|---|
| `Qwen3-1.7B` | base model, zero-shot, same prompt |
| `Qwen3-1.7B-lora` | one LoRA adapter per fold |
| `Kimi-K3` | hosted model on Nebius Token Factory, same prompt |
| ablations | `--context False`, `--rank` 8/32, `--fraction` 0.25/0.5, Qwen3-0.6B and 4B |

## Running

`scripts/data.py`, `scripts/predict_api.py` and `scripts/eval.py` run anywhere.
`scripts/lora.py` needs a CUDA GPU and the `finetune` extra. `predict_api.py` needs
`NEBIUS_API_TOKEN` in `.env`.

```
uv sync --python 3.12 --extra finetune
uv run scripts/data.py                                  # out/mt/pages.jsonl
uv run scripts/lora.py train --holdout balloon_dream --epochs 0.1   # smoke test
uv run scripts/predict_api.py                           # Kimi-K3 -> out/mt/preds/Kimi-K3/
uv run scripts/lora.py predict                          # base Qwen3-1.7B, all folds
uv run scripts/lora.py train                            # 5 adapters -> out/mt/adapters/
uv run scripts/lora.py predict --adapters Qwen3-1.7B-lora
uv run scripts/eval.py Qwen3-1.7B,Qwen3-1.7B-lora,Kimi-K3
```

On a fresh GPU instance (e.g. AWS `g6.2xlarge` with the Ubuntu Deep Learning AMI),
`scripts/aws_setup.sh` installs uv, syncs the `finetune` extra, downloads the data and runs the
smoke test. The full procedure is in [`docs/aws.md`](docs/aws.md)
([日本語](docs/aws.ja.md)).

With 8 to 12 GB of GPU memory, train with `--batch-size 1 --grad-accum 8`. GPUs without bf16
fall back to fp16 automatically.

## Plan

1. Week 1: base and Kimi-K3 runs, smoke-train one fold, then all 5 folds. First table.
2. Week 2: ablations (context, rank, training-set size learning curve, model size), error
   analysis on 30 sampled lines (names, honorifics, tone, sfx), optional COMET.
3. Week 3: serve the adapter with vLLM (`--enable-lora`), measure latency and cost per page
   against Kimi-K3, write up results.

## Results

Main runs as of 2026-10-28. The Qwen runs used one AWS `g6.2xlarge` (NVIDIA L4). The delta is
chrF++ vs `Qwen3-1.7B` with a 95% paired bootstrap over pages.

| run | pages | format errors | chrF++ | BLEU | sec/page | chrF++ delta |
|---|---|---|---|---|---|---|
| `Qwen3-1.7B` | 209 | 27 | 20.1 | 7.7 | 0.8 | |
| `Qwen3-1.7B-lora` | 209 | 13 | 24.1 | 10.6 | 1.4 | +4.0 [+1.7, +6.3] |
| `Kimi-K3` | 209 | 1 | 35.3 | 16.0 | 1.1 | +15.2 [+13.3, +17.2] |

chrF++ per held-out book:

| run | balloon_dream | boureisougi | rasetugari | tencho_isoro | tojime_no_siora |
|---|---|---|---|---|---|
| `Qwen3-1.7B` | 22.7 | 17.3 | 19.8 | 20.1 | 19.8 |
| `Qwen3-1.7B-lora` | 27.0 | 23.4 | 22.1 | 26.3 | 21.7 |
| `Kimi-K3` | 38.4 | 36.1 | 33.4 | 35.8 | 33.1 |

LoRA improves chrF++ by 4.0 points over the base model, with a 95% interval that excludes zero,
and improves every held-out book. It also halves the format errors (27 to 13 pages). Kimi-K3 is
still about 11 points ahead. The LoRA run is slower per page because the adapter is applied
unmerged with `transformers` generation; serving speed is measured separately in week 3.

The Kimi-K3 rows are from a rerun on 2026-10-28, kept so that every run can be paired page by page.
The first run on 2026-10-24 scored chrF++ 35.6 with 0 format errors and per-book chrF++ 38.7, 35.9,
33.0, 36.0 and 34.4; the API is not fully deterministic even with greedy decoding.

Gap to Kimi-K3 (chrF++ delta vs `Kimi-K3`, 95% paired bootstrap over pages):

| run | chrF++ | delta vs `Kimi-K3` |
|---|---|---|
| `Qwen3-4B-lora` | 28.4 | -6.9 [-9.0, -5.0] |
| `Qwen3-4B` | 27.2 | -8.1 [-9.8, -6.5] |
| `Qwen3-1.7B-lora` | 24.1 | -11.2 [-13.0, -9.4] |

Kimi-K3 is ahead on every held-out book. The smallest gap is on tencho_isoro and
tojime_no_siora (4.0 points for `Qwen3-4B-lora`), the largest on boureisougi (11.6 points).

Absolute scores are low because the OpenMantra references are loose and often lowercase
("don't you have aspirations?" vs Kimi-K3's "You really have no dream"). This affects every run
equally, so the differences between runs are what matter.

### Ablations

Run on 2026-10-27 with the same folds, prompt and hyperparameters, changing one thing at a time.
The delta is chrF++ vs `Qwen3-1.7B` unless a row names another baseline.

| run | change | format errors | chrF++ | BLEU | chrF++ delta |
|---|---|---|---|---|---|
| `Qwen3-1.7B-noctx` | no previous-page context | 29 | 21.2 | 8.2 | +1.1 [-0.8, +3.1] |
| `Qwen3-1.7B-lora-noctx` | no context, train and test | 15 | 24.8 | 10.9 | +4.7 [+2.7, +6.9] |
| | vs `Qwen3-1.7B-lora` | | | | +0.7 [-1.2, +2.4] |
| `Qwen3-1.7B-lora-r8` | rank 8 (alpha 32) | 16 | 23.5 | 10.4 | +3.4 [+1.0, +5.7] |
| `Qwen3-1.7B-lora-r32` | rank 32 (alpha 32) | 15 | 23.4 | 10.1 | +3.3 [+1.5, +5.3] |
| `Qwen3-1.7B-lora-f25` | 25% of the training pages | 31 | 20.6 | 7.9 | +0.5 [-1.5, +2.7] |
| `Qwen3-1.7B-lora-f50` | 50% of the training pages | 27 | 21.5 | 8.6 | +1.4 [-0.9, +3.7] |
| `Qwen3-0.6B` | smaller base | 114 | 5.1 | 0.6 | -15.0 [-17.0, -12.9] |
| `Qwen3-0.6B-lora` | smaller base + LoRA | 32 | 17.2 | 6.2 | -2.9 [-5.0, -0.7] |
| `Qwen3-4B` | larger base | 15 | 27.2 | 11.8 | +7.1 [+5.3, +8.9] |
| `Qwen3-4B-lora` | larger base + LoRA | 11 | 28.4 | 13.8 | +8.3 [+6.3, +10.4] |
| | vs `Qwen3-4B` | | | | +1.2 [-0.9, +3.1] |

chrF++ per held-out book:

| run | balloon_dream | boureisougi | rasetugari | tencho_isoro | tojime_no_siora |
|---|---|---|---|---|---|
| `Qwen3-1.7B-noctx` | 24.4 | 19.2 | 19.4 | 22.1 | 20.3 |
| `Qwen3-1.7B-lora-noctx` | 24.5 | 23.8 | 23.3 | 27.4 | 25.2 |
| `Qwen3-1.7B-lora-r8` | 23.5 | 22.5 | 21.7 | 26.5 | 23.3 |
| `Qwen3-1.7B-lora-r32` | 25.7 | 19.9 | 22.1 | 26.7 | 22.3 |
| `Qwen3-1.7B-lora-f25` | 22.1 | 21.8 | 19.1 | 24.7 | 15.9 |
| `Qwen3-1.7B-lora-f50` | 20.5 | 19.2 | 22.4 | 23.4 | 21.7 |
| `Qwen3-0.6B` | 4.9 | 4.9 | 8.2 | 1.9 | 5.0 |
| `Qwen3-0.6B-lora` | 17.6 | 18.0 | 16.0 | 20.3 | 14.4 |
| `Qwen3-4B` | 29.9 | 24.1 | 24.9 | 28.3 | 28.2 |
| `Qwen3-4B-lora` | 29.2 | 24.5 | 27.1 | 31.8 | 29.1 |

- The previous-page context does not help: without it, both the base model and the LoRA model
  score the same or slightly higher, and the paired difference between the two LoRA runs
  includes zero.
- Rank 8, 16 and 32 give similar gains with overlapping intervals.
- Training-set size matters most for LoRA: with 25% or 50% of the roughly 170 training pages the
  gain is not distinguishable from zero and format errors barely drop.
- Model size matters most overall. LoRA rescues Qwen3-0.6B (114 to 32 format errors, chrF++ 5.1 to
  17.2). The untuned Qwen3-4B already beats Qwen3-1.7B + LoRA, and LoRA on top of 4B adds +1.2
  with an interval that includes zero. `Qwen3-4B-lora` is the best Qwen run at 28.4, about 7
  points behind Kimi-K3.
- Seconds per page are not comparable across runs: pages with broken output often generate up to
  the 1024-token limit.

### 結果（日本語）

2026-10-27 時点の本実験の結果です。Qwen は AWS の `g6.2xlarge`（NVIDIA L4）で実行しました。
LoRA によって chrF++ はベースモデルの 20.1 から 24.1 に上がり（差 +4.0、95% 区間 [+1.7, +6.3]）、
5冊すべてで向上しました。フォーマットエラーも27ページから13ページに半減しました。一方、
Kimi-K3（chrF++ 35.3、フォーマットエラー1件）とはまだ11ポイントほどの差があります。
OpenMantra の参照訳は意訳寄りで小文字も多いため絶対値は低めですが、どの実行にも同じように
影響するので、重要なのは実行間の差です。

比較実験（Ablations の表）では、次のことが分かりました。

- 前のページのコンテキストは効いていません。コンテキストなしでもベース、LoRA ともに同じか
  わずかに高く、LoRA 同士の差（+0.7、[-1.2, +2.4]）は 0 を含みます。
- rank 8、16、32 の効果はほぼ同じです。
- 学習データの量は重要です。約170ページの 25% や 50% では効果がはっきりせず、フォーマット
  エラーもほとんど減りません。
- 最も効くのはモデルサイズです。Qwen3-0.6B は LoRA でフォーマットエラーが 114 から 32 に減り、
  chrF++ も 5.1 から 17.2 に上がりました。Qwen3-4B は LoRA なしでも 1.7B + LoRA を上回り、
  4B に LoRA を足した差（+1.2、[-0.9, +3.1]）は 0 を含みます。Qwen の最高は `Qwen3-4B-lora` の
  28.4 で、Kimi-K3 とは約7ポイントの差です。

Kimi-K3 は 2026-10-28 に作り直した予測の値で、全 run とページ単位で対応を取って比較しています
（2026-10-24 の初回は chrF++ 35.6、フォーマットエラー0件。greedy でも API の出力は完全には一定
ではありません）。Kimi-K3 との差は Qwen3-4B-lora が -6.9 [-9.0, -5.0]、Qwen3-4B が
-8.1 [-9.8, -6.5]、Qwen3-1.7B-lora が -11.2 [-13.0, -9.4] で、5冊すべてで Kimi-K3 が上でした。

## License

The code is MIT (see `LICENSE`). The OpenMantra data is CC BY-NC 4.0: it is downloaded at run
time and not stored here, and adapters trained on it are for non-commercial use. Cite:

```
@article{hinami_aaai_2021,
    author={Ryota Hinami and Shonosuke Ishiwatari and Kazuhiko Yasuda and Yusuke Matsui},
    title={Towards Fully Automated Manga Translation},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2021}
}
```
