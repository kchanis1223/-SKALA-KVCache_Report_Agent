.PHONY: setup run test test-unit lint format
setup:
	uv sync --locked
run:
	uv run skala-agent --verbose
test:
	uv run pytest
test-unit:
	uv run pytest -m "not live"
lint:
	uv run ruff check src tests app.py
	uv run ruff format --check src tests app.py
format:
	uv run ruff check --fix src tests app.py
	uv run ruff format src tests app.py
