import json

import fire

from manga_lora.mt import OUT, Prediction, bleu, bootstrap_delta, chrf, load_predictions

COLUMNS = ["run", "pages", "format_errors", "chrf++", "bleu", "sec_per_page", "delta_vs_base"]


def as_list(value: str | list[str]) -> list[str]:
    return value.split(",") if isinstance(value, str) else value


def score(run: str, preds: list[Prediction], baseline: list[Prediction]) -> dict[str, object]:
    delta, low, high = bootstrap_delta(preds, baseline)
    return {
        "run": run,
        "pages": len(preds),
        "format_errors": sum(p["error"] is not None for p in preds),
        "chrf++": chrf(preds),
        "bleu": bleu(preds),
        "sec_per_page": sum(p["seconds"] for p in preds) / len(preds),
        "delta_vs_base": f"{delta:+.1f} [{low:+.1f}, {high:+.1f}]",
    }


def per_book(preds: list[Prediction]) -> dict[str, float]:
    by_book: dict[str, list[Prediction]] = {}
    for p in preds:
        by_book.setdefault(p["book"], []).append(p)
    return {book: chrf(rows) for book, rows in sorted(by_book.items())}


def print_table(rows: list[dict[str, object]]) -> None:
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in rows:
        cells = [f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]) for c in COLUMNS]
        print("| " + " | ".join(cells) + " |")


def main(runs: str | list[str], baseline: str | None = None) -> None:
    names = as_list(runs)
    base_name = baseline or names[0]
    base = load_predictions(base_name)
    loaded = {name: load_predictions(name) for name in names}
    rows = [score(name, preds, base) for name, preds in loaded.items()]
    print(f"chrF++ delta vs {base_name}, 95% paired bootstrap over pages\n")
    print_table(rows)
    books = {name: per_book(preds) for name, preds in loaded.items()}
    print("\nchrF++ per held-out book\n")
    titles = sorted({b for scores in books.values() for b in scores})
    print("| run | " + " | ".join(titles) + " |")
    print("|" + "---|" * (len(titles) + 1))
    for name, scores in books.items():
        print(f"| {name} | " + " | ".join(f"{scores.get(t, 0.0):.1f}" for t in titles) + " |")
    (OUT / "summary.json").write_text(json.dumps({"runs": rows, "books": books}, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
