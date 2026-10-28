.PHONY: install train test check
install:
	python -m pip install -e ".[dev]"
train:
	python -m movie_recs train
test:
	python -m pytest -q
check:
	python -m ruff check .
	python -m ruff format --check .
