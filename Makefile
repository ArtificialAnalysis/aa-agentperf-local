.PHONY: sync format lint test rustcore test-rust test-rustcore

sync:
	uv sync --locked

format:
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check

test:
	uv run pytest

rustcore:
	uv sync --locked --extra rust

test-rust:
	cargo test --locked -p agentperf-local-rustcore --no-default-features

test-rustcore: rustcore
	uv run pytest
