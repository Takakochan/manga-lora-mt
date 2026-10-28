import json

import fire

from manga_lora.mt import (
    OUT,
    Prediction,
    bleu,
    bootstrap_delta,
    chrf,
    load_predictions,
    paired,
)

COLUMNS = [
    "run",
    "pages",
    "format_errors",
    "chrf++",
    "bleu",
    "sec_per_page",
    "paired_pages",
    "delta_vs_base",
]


def as_list(value: str | list[str]) -> list[str]:
    return value.split(",") if isinstance(value, str) else value


def score(
    run: str, preds: list[Prediction], baseline: list[Prediction], lowercase: bool
) -> dict[str, object]:
    delta, low, high = bootstrap_delta(preds, baseline, lowercase=lowercase)
    return {
        "run": run,
        "pages": len(preds),
        "format_errors": sum(p["error"] is not None for p in preds),
        "chrf++": chrf(preds, lowercase),
        "bleu": bleu(preds, lowercase),
        "sec_per_page": sum(p["seconds"] for p in preds) / len(preds),
        "paired_pages": len(paired(preds, baseline)[0]),
        "delta_vs_base": f"{delta:+.1f} [{low:+.1f}, {high:+.1f}]",
    }


def per_book(preds: list[Prediction], lowercase: bool) -> dict[str, float]:
    by_book: dict[str, list[Prediction]] = {}
    for p in preds:
        by_book.setdefault(p["book"], []).append(p)
    return {book: chrf(rows, lowercase) for book, rows in sorted(by_book.items())}


def print_table(rows: list[dict[str, object]]) -> None:
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        cells = [f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]) for c in COLUMNS]
        print("| " + " | ".join(cells) + " |")


def main(runs: str | list[str], baseline: str | None = None, lowercase: bool = False) -> None:
    names = as_list(runs)
    base_name = baseline or names[0]
    base = load_predictions(base_name)
    loaded = {name: load_predictions(name) for name in names}
    rows = [score(name, preds, base, lowercase) for name, preds in loaded.items()]
    case = ", lowercased" if lowercase else ""
    print(f"chrF++ delta vs {base_name}, 95% paired bootstrap over pages{case}\n")
    print_table(rows)
    books = {name: per_book(preds, lowercase) for name, preds in loaded.items()}
    print("\nchrF++ per held-out book\n")
    titles = sorted({b for scores in books.values() for b in scores})
    print("| run | " + " | ".join(titles) + " |")
    print("|" + "---|" * (len(titles) + 1))
    for name, scores in books.items():
        print(f"| {name} | " + " | ".join(f"{scores.get(t, 0.0):.1f}" for t in titles) + " |")
    summary = {"lowercase": lowercase, "runs": rows, "books": books}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
