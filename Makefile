.PHONY: install test lint audit check check-data check-full-data

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

check:
	python scripts/check_project.py

check-data:
	python scripts/check_project.py --with-data

check-full-data:
	python scripts/check_project.py --full-data

audit:
	python scripts/audit_dataset.py --input data/sample/vivqax_sample.jsonl --output artifacts/audit/sample_audit.json
