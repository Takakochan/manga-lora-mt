from collections import Counter

import fire

from manga_lora.mt import PAGES, download_examples, write_jsonl


def main() -> None:
    examples = download_examples()
    write_jsonl(PAGES, examples)
    pages = Counter(e["book"] for e in examples)
    lines = Counter[str]()
    for e in examples:
        lines[e["book"]] += len(e["ja"])
    for book in sorted(pages):
        print(f"{book:20} {pages[book]:4} pages {lines[book]:5} lines")
    print(f"wrote {len(examples)} pages to {PAGES}")


if __name__ == "__main__":
    fire.Fire(main)
