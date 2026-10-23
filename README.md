# manga-lora-mt

Does a small Qwen model, LoRA-tuned on manga dialogue, translate Japanese manga pages better
than the same model untuned, and how close does it get to a large hosted model (Kimi-K3)?

The setting comes from [Fukidashi](https://github.com/IlyaGusev/fukidashi), a manga
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

With 8 to 12 GB of GPU memory, train with `--batch-size 1 --grad-accum 8`. GPUs without bf16
fall back to fp16 automatically.

## Plan

1. Week 1: base and Kimi-K3 runs, smoke-train one fold, then all 5 folds. First table.
2. Week 2: ablations (context, rank, training-set size learning curve, model size), error
   analysis on 30 sampled lines (names, honorifics, tone, sfx), optional COMET.
3. Week 3: serve the adapter with vLLM (`--enable-lora`), measure latency and cost per page
   against Kimi-K3, write up results.

## Results

Not run yet.

## Data license

OpenMantra is CC BY-NC 4.0. It is downloaded at run time and not stored here. Adapters trained
on it are for non-commercial use. Cite:

```
@article{hinami_aaai_2021,
    author={Ryota Hinami and Shonosuke Ishiwatari and Kazuhiko Yasuda and Yusuke Matsui},
    title={Towards Fully Automated Manga Translation},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    year={2021}
}
```
