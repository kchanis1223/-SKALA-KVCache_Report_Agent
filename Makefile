.PHONY: setup run test lint format
setup:
	uv sync --locked
run:
	uv run skala-agent --output outputs/report.md
test:
	uv run pytest
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
format:
	uv run ruff check --fix src tests
	uv run ruff format src tests
