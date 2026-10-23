.PHONY: test check

test:
	uv run pytest

check:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts
	uv run mypy src tests scripts
