import json
from pathlib import Path
from typing import Any

import fire

from manga_lora.mt import (
    OUT,
    Example,
    Prediction,
    aligned_hyp,
    build_layouts,
    chrf,
    download_annotation,
    load_examples,
    load_predictions,
    page_key,
)

RUNS = "Kimi-K3,Qwen3-4B-lora,Qwen3-4B,Qwen3-1.7B-lora,Qwen3-1.7B"
TEMPLATE = Path(__file__).with_name("viewer.html")


def translation(pred: Prediction | None) -> dict[str, Any] | None:
    if pred is None:
        return None
    return {"lines": aligned_hyp(pred), "error": pred["error"], "chrf": round(chrf([pred]), 1)}


def page_rows(
    examples: list[Example], runs: dict[str, dict[tuple[str, int], Prediction]]
) -> list[dict[str, Any]]:
    layouts = build_layouts(download_annotation())
    return [
        {
            "book": e["book"],
            "page": e["page"],
            "ja": e["ja"],
            "en": e["en"],
            **layouts[e["book"], e["page"]],
            "runs": {name: translation(preds.get(page_key(e))) for name, preds in runs.items()},
        }
        for e in examples
    ]


def embed(data: object) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def main(runs: str | list[str] = RUNS, out: str = str(OUT / "viewer.html")) -> None:
    names = runs.split(",") if isinstance(runs, str) else runs
    loaded = {name: {page_key(p): p for p in load_predictions(name)} for name in names}
    for name, preds in loaded.items():
        print(f"{name:20} {len(preds):4} pages")
    pages = page_rows(load_examples(), loaded)
    data = {"runs": names, "pages": pages}
    Path(out).write_text(TEMPLATE.read_text().replace("__DATA__", embed(data)))
    print(f"wrote {len(pages)} pages to {out}")


if __name__ == "__main__":
    fire.Fire(main)
