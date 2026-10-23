# CLAUDE.md

## Git

- Do not force push.
- Create a new commit. Do not amend a commit.

## Conventions

- Python 3.12+, strict mypy, Ruff with `line-length = 100`. `make check`, `make test`.
- Use async code for API calls.
- Put all imports at the top of the file.
- Do not write code comments. Use clear names and named helpers. Keep only tool directives.

## Project

LoRA fine-tuning experiment for manga translation: Qwen vs Qwen + LoRA vs Kimi-K3, on
OpenMantra with leave-one-book-out folds.

- `src/manga_lora/mt.py`: pages with the previous page as context, splits, the prompt, parsing
  (`BadOutput`), chrF++/BLEU and paired bootstrap.
- `src/manga_lora/settings.py`: settings via pydantic-settings, `MANGA_LORA_<FIELD>` plus
  `NEBIUS_API_TOKEN` from `.env`. `src/manga_lora/nebius.py`: the API client.
- `scripts/data.py` writes `out/mt/pages.jsonl`. `scripts/predict_api.py` runs a Nebius model.
  `scripts/lora.py train|predict` trains and runs Qwen + LoRA (CUDA, `finetune` extra).
  `scripts/eval.py` prints the comparison. Predictions go to `out/mt/preds/<run>/`, adapters to
  `out/mt/adapters/<run>/<book>/`.
- Training and GPU inference run on an Ubuntu machine with a GPU. The Mac (Intel) cannot
  install torch.
