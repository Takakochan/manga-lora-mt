import json
import random
import time
from pathlib import Path
from typing import Any

import fire
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from manga_lora.mt import (
    OUT,
    BadOutput,
    Example,
    Prediction,
    answer,
    books,
    load_examples,
    messages,
    parse_translations,
    run_dir,
    split,
    write_jsonl,
)

BASE = "Qwen/Qwen3-1.7B"
IGNORE = -100
Batch = dict[str, torch.Tensor]


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def holdouts(holdout: str) -> list[str]:
    return books(load_examples()) if holdout == "all" else [holdout]


def load_tokenizer(model: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model: str) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(model, dtype=dtype()).to(device())


def prompt_ids(tokenizer: PreTrainedTokenizerBase, example: Example, context: bool) -> list[int]:
    ids: list[int] = tokenizer.apply_chat_template(
        messages(example, context=context),
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=False,
    )
    return ids


def encode(
    tokenizer: PreTrainedTokenizerBase, example: Example, context: bool
) -> dict[str, list[int]]:
    prompt = prompt_ids(tokenizer, example, context)
    target = tokenizer(answer(example) + tokenizer.eos_token, add_special_tokens=False)
    return {
        "input_ids": prompt + target["input_ids"],
        "labels": [IGNORE] * len(prompt) + target["input_ids"],
    }


def pad(rows: list[list[int]], value: int) -> torch.Tensor:
    width = max(len(r) for r in rows)
    return torch.tensor([r + [value] * (width - len(r)) for r in rows])


def collator(pad_id: int) -> Any:
    def collate(features: list[dict[str, list[int]]]) -> Batch:
        ids = [f["input_ids"] for f in features]
        return {
            "input_ids": pad(ids, pad_id),
            "labels": pad([f["labels"] for f in features], IGNORE),
            "attention_mask": pad([[1] * len(i) for i in ids], 0),
        }

    return collate


def subsample(examples: list[Example], fraction: float, seed: int) -> list[Example]:
    keep = max(1, round(len(examples) * fraction))
    return random.Random(seed).sample(examples, keep) if keep < len(examples) else examples


def adapter_dir(run: str, holdout: str) -> Path:
    return OUT / "adapters" / run / holdout


def train_fold(
    model_name: str,
    run: str,
    holdout: str,
    context: bool,
    hyper: dict[str, Any],
) -> None:
    train_set = subsample(split(load_examples(), holdout)[0], hyper["fraction"], hyper["seed"])
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=hyper["rank"],
        lora_alpha=hyper["alpha"],
        lora_dropout=hyper["dropout"],
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    peft_model.print_trainable_parameters()
    out = adapter_dir(run, holdout)
    half = dtype()
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=hyper["epochs"],
        per_device_train_batch_size=hyper["batch_size"],
        gradient_accumulation_steps=hyper["grad_accum"],
        learning_rate=hyper["lr"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="no",
        bf16=half == torch.bfloat16,
        fp16=half == torch.float16,
        report_to="none",
        seed=hyper["seed"],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=[encode(tokenizer, e, context) for e in train_set],
        data_collator=collator(tokenizer.pad_token_id),
    )
    start = time.monotonic()
    trainer.train()
    peft_model.save_pretrained(str(out))
    log = {
        "base": model_name,
        "holdout": holdout,
        "train_pages": len(train_set),
        "context": context,
        "minutes": (time.monotonic() - start) / 60,
        "hyper": hyper,
        "history": trainer.state.log_history,
    }
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"saved adapter for {holdout} to {out}", flush=True)


def train(
    holdout: str = "all",
    model: str = BASE,
    run: str | None = None,
    context: bool = True,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    epochs: float = 3,
    batch_size: int = 4,
    grad_accum: int = 2,
    lr: float = 2e-4,
    fraction: float = 1.0,
    seed: int = 0,
) -> None:
    name = run or model.split("/")[-1] + "-lora"
    hyper = {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "lr": lr,
        "fraction": fraction,
        "seed": seed,
    }
    for book in holdouts(holdout):
        train_fold(model, name, book, context, hyper)


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch: list[Example],
    context: bool,
    max_new_tokens: int,
) -> list[str]:
    prompts = [prompt_ids(tokenizer, e, context) for e in batch]
    width = max(len(p) for p in prompts)
    ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(p)) + p for p in prompts])
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts])
    with torch.no_grad():
        out = model.generate(
            input_ids=ids.to(model.device),
            attention_mask=mask.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    texts: list[str] = tokenizer.batch_decode(out[:, width:], skip_special_tokens=True)
    return texts


def to_prediction(example: Example, text: str, seconds: float) -> Prediction:
    hyp: list[str] = []
    error = None
    try:
        hyp = parse_translations(text, len(example["ja"]))
    except BadOutput as e:
        error = f"BadOutput: {e}"
    return {
        "book": example["book"],
        "page": example["page"],
        "ref": example["en"],
        "hyp": hyp,
        "error": error,
        "seconds": seconds,
    }


def predict_fold(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    holdout: str,
    run: str,
    context: bool,
    batch_size: int,
    max_new_tokens: int,
) -> None:
    _, test_set = split(load_examples(), holdout)
    preds: list[Prediction] = []
    for i in range(0, len(test_set), batch_size):
        batch = test_set[i : i + batch_size]
        start = time.monotonic()
        texts = generate(model, tokenizer, batch, context, max_new_tokens)
        seconds = (time.monotonic() - start) / len(batch)
        preds += [to_prediction(e, t, seconds) for e, t in zip(batch, texts, strict=True)]
        print(f"{holdout:20} {len(preds):3}/{len(test_set)}", flush=True)
    write_jsonl(run_dir(run) / f"{holdout}.jsonl", preds)
    errors = sum(p["error"] is not None for p in preds)
    print(f"{holdout}: {errors} format errors in {len(preds)} pages", flush=True)


def predict(
    holdout: str = "all",
    model: str = BASE,
    adapters: str | None = None,
    run: str | None = None,
    context: bool = True,
    batch_size: int = 8,
    max_new_tokens: int = 1024,
) -> None:
    name = run or adapters or model.split("/")[-1]
    tokenizer = load_tokenizer(model)
    tokenizer.padding_side = "left"
    base = load_model(model).eval()
    for book in holdouts(holdout):
        tuned = (
            PeftModel.from_pretrained(base, str(adapter_dir(adapters, book))) if adapters else base
        )
        predict_fold(tuned, tokenizer, book, name, context, batch_size, max_new_tokens)
        if isinstance(tuned, PeftModel):
            base = tuned.unload()


if __name__ == "__main__":
    fire.Fire({"train": train, "predict": predict})
