.PHONY: install test lint audit check check-data check-full-data

install:
	python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	python -m pip install -e ".[dev,verification,demo,report]"

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
