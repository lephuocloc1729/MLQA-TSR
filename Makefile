.PHONY: help setup check-data qdrant-up qdrant-down preprocess index retrieve task1 task2 assistant eval demo ci-test test verify clean

help:
	@echo "Traffic Legal VLM - common commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup       Install pinned Python dependencies"
	@echo "  make check-data  Check expected raw VLSP data paths"
	@echo "  make ci-test     Run lightweight schema tests used by CI"
	@echo "  make test        Run local unit tests with the macOS readline workaround"
	@echo ""
	@echo "Services:"
	@echo "  make qdrant-up   Start Qdrant with docker compose"
	@echo "  make qdrant-down Stop Qdrant"
	@echo ""
	@echo "Pipeline placeholders:"
	@echo "  make preprocess  Build processed LawDB articles"
	@echo "  make index       Build the retrieval index"
	@echo "  make eval        Evaluate data/outputs/dev_predictions.jsonl"
	@echo "  make demo        Start the Streamlit demo"

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

check-data:
	bash scripts/check_data.sh

qdrant-up:
	docker compose up -d

qdrant-down:
	docker compose down

preprocess:
	python -m src.data_utils --mode preprocess

index:
	python -m src.retrieval --mode index

retrieve:
	python -m src.retrieval --mode retrieve

task1:
	python -m src.pipeline --mode task1

task2:
	python -m src.pipeline --mode task2

assistant:
	python -m src.pipeline --mode assistant

eval:
	bash scripts/evaluate.sh

demo:
	python -m streamlit run app/streamlit_app.py

ci-test:
	python -c 'import sys, types; sys.modules.setdefault("readline", types.ModuleType("readline")); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_schemas.py"], plugins=[]))'

test:
	python -c 'import sys, types; sys.modules.setdefault("readline", types.ModuleType("readline")); import pytest; raise SystemExit(pytest.main(["-q", "tests"], plugins=[]))'

verify: ci-test
	python -m pip check
	git diff --check

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
