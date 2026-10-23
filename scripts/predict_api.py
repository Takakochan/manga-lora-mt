import asyncio
import time
from typing import Any, cast

import fire
from openai.types.chat import ChatCompletionMessageParam

from manga_lora.mt import (
    Example,
    Prediction,
    books,
    load_examples,
    messages,
    parse_translations,
    run_dir,
    write_jsonl,
)
from manga_lora.nebius import client
from manga_lora.settings import settings


async def translate(example: Example, model: str, context: bool) -> str:
    chat = cast(list[ChatCompletionMessageParam], messages(example, context=context))
    response = await client().chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=4096,
        messages=chat,
        response_format={"type": "json_object"},
        reasoning_effort="none",
        extra_body={"chat_template_kwargs": {"thinking": False, "enable_thinking": False}},
    )
    return response.choices[0].message.content or ""


async def predict(
    example: Example, model: str, context: bool, sem: asyncio.Semaphore
) -> Prediction:
    async with sem:
        start = time.monotonic()
        hyp: list[str] = []
        error = None
        try:
            async with asyncio.timeout(settings.step_timeout):
                text = await translate(example, model, context)
            hyp = parse_translations(text, len(example["ja"]))
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
    print(f"{example['book']:20} {example['page']:3} {error or 'ok'}", flush=True)
    return {
        "book": example["book"],
        "page": example["page"],
        "ref": example["en"],
        "hyp": hyp,
        "error": error,
        "seconds": time.monotonic() - start,
    }


async def run_all(model: str, run: str, context: bool, concurrency: int) -> None:
    client()
    examples = load_examples()
    sem = asyncio.Semaphore(concurrency)
    preds: list[Any] = await asyncio.gather(*(predict(e, model, context, sem) for e in examples))
    for book in books(examples):
        write_jsonl(run_dir(run) / f"{book}.jsonl", [p for p in preds if p["book"] == book])


def main(
    model: str = settings.api_model,
    run: str | None = None,
    context: bool = True,
    concurrency: int = 8,
) -> None:
    name = run or model.split("/")[-1] + ("" if context else "-noctx")
    asyncio.run(run_all(model, name, context, concurrency))


if __name__ == "__main__":
    fire.Fire(main)
