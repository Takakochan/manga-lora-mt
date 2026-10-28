import random
from typing import Any

import fire

from manga_lora.mt import OUT, load_examples, load_predictions, page_key, write_jsonl

RUNS = "Kimi-K3,Qwen3-4B-lora,Qwen3-1.7B-lora"


def main(runs: str | list[str] = RUNS, n: int = 30, seed: int = 0) -> None:
    names = runs.split(",") if isinstance(runs, str) else runs
    preds = {name: {page_key(p): p for p in load_predictions(name)} for name in names}
    candidates: list[dict[str, Any]] = []
    for e in load_examples():
        key = page_key(e)
        if all(preds[name][key]["error"] is None for name in names):
            for i, ja in enumerate(e["ja"]):
                candidates.append(
                    {
                        "book": e["book"],
                        "page": e["page"],
                        "line": i + 1,
                        "ja": ja,
                        "ref": e["en"][i],
                        **{name: preds[name][key]["hyp"][i] for name in names},
                    }
                )
    picked = sorted(
        random.Random(seed).sample(candidates, n), key=lambda r: (r["book"], r["page"], r["line"])
    )
    write_jsonl(OUT / "sample.jsonl", picked)
    print(f"sampled {n} of {len(candidates)} lines to {OUT / 'sample.jsonl'}")


if __name__ == "__main__":
    fire.Fire(main)
