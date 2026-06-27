# HackerRank Orchestrate — Multi-Modal Evidence Review
# Uses `uv` (https://docs.astral.sh/uv/). Falls back to plain python + pip is
# documented in code/README.md for graders without uv.

.PHONY: install run eval test clean

install:        ## Create the env and install deps (provisions Python 3.12)
	uv sync

run:            ## Generate output.csv for the test set (dataset/claims.csv)
	uv run python code/main.py

eval:           ## Evaluate on sample_claims.csv and write the report
	uv run python code/evaluation/main.py

test:           ## Run the unit-test suite
	uv run pytest

clean:          ## Remove generated caches/artifacts (keeps .vlm_cache.json)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
