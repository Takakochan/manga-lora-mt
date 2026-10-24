from typing import Any

import pytest

from manga_lora.mt import (
    BadOutput,
    Prediction,
    bootstrap_delta,
    build_examples,
    chrf,
    messages,
    parse_translations,
    split,
)


def line(ja: str, en: str) -> dict[str, str]:
    return {"text_ja": ja, "text_en": en}


ANNOTATION: list[dict[str, Any]] = [
    {
        "book_title": "a",
        "pages": [
            {"text": [line("おはよう", "good morning")]},
            {"text": []},
            {"text": [line("行くぞ", "let's go"), line("待って", "wait")]},
        ],
    },
    {"book_title": "b", "pages": [{"text": [line("はい", "yes")]}]},
]


def pred(page: int, hyp: list[str], ref: list[str]) -> Prediction:
    return {"book": "a", "page": page, "ref": ref, "hyp": hyp, "error": None, "seconds": 1.0}


def test_examples_skip_empty_pages_and_carry_previous_text() -> None:
    examples = build_examples(ANNOTATION)
    assert [(e["book"], e["page"]) for e in examples] == [("a", 0), ("a", 2), ("b", 0)]
    assert examples[1]["context"] == ["おはよう"]
    assert examples[2]["context"] == []


def test_split_holds_out_one_book() -> None:
    train, test = split(build_examples(ANNOTATION), "b")
    assert {e["book"] for e in train} == {"a"}
    assert [e["ja"] for e in test] == [["はい"]]
    with pytest.raises(ValueError):
        split(build_examples(ANNOTATION), "c")


def test_prompt_numbers_lines_and_can_drop_context() -> None:
    example = build_examples(ANNOTATION)[1]
    with_context = messages(example)[1]["content"]
    assert "1. 行くぞ\n2. 待って" in with_context
    assert "- おはよう" in with_context
    assert "おはよう" not in messages(example, context=False)[1]["content"]


def test_parse_translations_checks_shape() -> None:
    assert parse_translations('```json\n{"translations": ["a", "b"]}\n```', 2) == ["a", "b"]
    assert parse_translations('```json{"translations": ["a", "b"]}```', 2) == ["a", "b"]
    assert parse_translations('```\n{"translations": ["a", "b"]}', 2) == ["a", "b"]
    for bad in ('{"translations": ["a"]}', "not json", "```", '{"lines": []}'):
        with pytest.raises(BadOutput):
            parse_translations(bad, 2)


def test_wrong_line_count_scores_as_empty() -> None:
    good = [pred(0, ["good morning"], ["good morning"])]
    misaligned = [pred(0, ["good", "morning"], ["good morning"])]
    assert chrf(good) == pytest.approx(100)
    assert chrf(misaligned) == 0


def test_bootstrap_delta_brackets_the_observed_gain() -> None:
    refs = [["let's go to the station"], ["wait for me"], ["good morning"]]
    better = [pred(i, r, r) for i, r in enumerate(refs)]
    worse = [pred(i, ["hello"], r) for i, r in enumerate(refs)]
    delta, low, high = bootstrap_delta(better, worse, samples=200)
    assert delta > 0
    assert low <= delta <= high
