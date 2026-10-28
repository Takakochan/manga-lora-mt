import json
import random
import re
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypedDict

from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF

DATASET_URL = (
    "https://raw.githubusercontent.com/mantra-inc/open-mantra-dataset/"
    "59341ad922284a8be219b7784feb5a79779a1447/"
)
ANNOTATION_URL = DATASET_URL + "annotation.json"
OUT = Path("out/mt")
PAGES = OUT / "pages.jsonl"

FENCE = re.compile(r"^```[\w-]*\s*(.*?)\s*(?:```)?$", re.DOTALL)

SYSTEM = "You translate manga dialogue from Japanese into natural, idiomatic {lang}."

PROMPT = """\
Translate each numbered line of this manga page into {lang}.
Keep the same number of lines, in the same order. Match each character's tone.
{context}
Lines:
{lines}

Answer with JSON only, in this shape:
{{"translations": ["...", "..."]}}
"""

CONTEXT = """
Text from the previous page, for consistent names, terms and tone:
{lines}
"""


class BadOutput(ValueError):
    pass


class Example(TypedDict):
    book: str
    page: int
    ja: list[str]
    en: list[str]
    context: list[str]


class Layout(TypedDict):
    image: str
    boxes: list[list[int]]


class Prediction(TypedDict):
    book: str
    page: int
    ref: list[str]
    hyp: list[str]
    error: str | None
    seconds: float


def build_examples(annotation: list[dict[str, Any]]) -> list[Example]:
    examples: list[Example] = []
    for book in annotation:
        previous: list[str] = []
        for index, page in enumerate(book["pages"]):
            ja = [t["text_ja"] for t in page["text"]]
            if ja:
                examples.append(
                    {
                        "book": book["book_title"],
                        "page": index,
                        "ja": ja,
                        "en": [t["text_en"] for t in page["text"]],
                        "context": previous,
                    }
                )
                previous = ja
    return examples


def build_layouts(annotation: list[dict[str, Any]]) -> dict[tuple[str, int], Layout]:
    layouts: dict[tuple[str, int], Layout] = {}
    for book in annotation:
        for index, page in enumerate(book["pages"]):
            if page["text"]:
                layouts[book["book_title"], index] = {
                    "image": DATASET_URL + page["image_paths"]["ja"],
                    "boxes": [[t["x"], t["y"], t["w"], t["h"]] for t in page["text"]],
                }
    return layouts


def download_annotation() -> list[dict[str, Any]]:
    with urllib.request.urlopen(ANNOTATION_URL) as response:
        annotation: list[dict[str, Any]] = json.loads(response.read())
    return annotation


def download_examples() -> list[Example]:
    return build_examples(download_annotation())


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_examples() -> list[Example]:
    return read_jsonl(PAGES)


def books(examples: list[Example]) -> list[str]:
    return sorted({e["book"] for e in examples})


def split(examples: list[Example], holdout: str) -> tuple[list[Example], list[Example]]:
    if holdout not in books(examples):
        raise ValueError(f"unknown book {holdout!r}, expected one of {books(examples)}")
    train = [e for e in examples if e["book"] != holdout]
    test = [e for e in examples if e["book"] == holdout]
    return train, test


def numbered(lines: list[str]) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, 1))


def messages(example: Example, lang: str = "English", context: bool = True) -> list[dict[str, str]]:
    previous = example["context"] if context else []
    ctx = CONTEXT.format(lines="\n".join(f"- {t}" for t in previous)) if previous else ""
    return [
        {"role": "system", "content": SYSTEM.format(lang=lang)},
        {
            "role": "user",
            "content": PROMPT.format(lang=lang, context=ctx, lines=numbered(example["ja"])),
        },
    ]


def answer(example: Example) -> str:
    return json.dumps({"translations": example["en"]}, ensure_ascii=False)


def strip_fence(text: str) -> str:
    text = text.strip()
    fenced = FENCE.match(text)
    return fenced.group(1) if fenced else text


def parse_translations(text: str, expected: int) -> list[str]:
    try:
        data = json.loads(strip_fence(text))
    except json.JSONDecodeError as e:
        raise BadOutput(f"invalid JSON after {len(text)} chars: {e}") from e
    lines = data.get("translations") if isinstance(data, dict) else None
    if not isinstance(lines, list):
        raise BadOutput("answer has no translations list")
    if len(lines) != expected:
        raise BadOutput(f"expected {expected} lines, got {len(lines)}")
    return [str(line) for line in lines]


def run_dir(run: str) -> Path:
    return OUT / "preds" / run


def load_predictions(run: str) -> list[Prediction]:
    return [p for path in sorted(run_dir(run).glob("*.jsonl")) for p in read_jsonl(path)]


def aligned_hyp(pred: Prediction) -> list[str]:
    return pred["hyp"] if len(pred["hyp"]) == len(pred["ref"]) else [""] * len(pred["ref"])


def corpus_lines(preds: list[Prediction]) -> tuple[list[str], list[str]]:
    hyps = [h for p in preds for h in aligned_hyp(p)]
    refs = [r for p in preds for r in p["ref"]]
    return hyps, refs


def chrf(preds: list[Prediction]) -> float:
    hyps, refs = corpus_lines(preds)
    return float(CHRF(word_order=2).corpus_score(hyps, [refs]).score)


def bleu(preds: list[Prediction]) -> float:
    hyps, refs = corpus_lines(preds)
    return float(BLEU().corpus_score(hyps, [refs]).score)


def page_key(row: Example | Prediction) -> tuple[str, int]:
    return row["book"], row["page"]


def paired(a: list[Prediction], b: list[Prediction]) -> tuple[list[Prediction], list[Prediction]]:
    by_page = {page_key(p): p for p in b}
    shared = [p for p in a if page_key(p) in by_page]
    return shared, [by_page[page_key(p)] for p in shared]


def bootstrap_delta(
    a: list[Prediction], b: list[Prediction], samples: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    a, b = paired(a, b)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        picks = [rng.randrange(len(a)) for _ in a]
        deltas.append(chrf([a[i] for i in picks]) - chrf([b[i] for i in picks]))
    deltas.sort()
    return chrf(a) - chrf(b), deltas[int(0.025 * samples)], deltas[int(0.975 * samples) - 1]
