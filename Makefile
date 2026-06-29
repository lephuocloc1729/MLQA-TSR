.PHONY: help setup check-data check-data-all qdrant-up qdrant-down preprocess index index-examples index-week2 retrieve benchmark-b2 benchmark-b3 benchmark-b4 benchmark-week2-smoke benchmark-w3-real adapter-diagnostic qlora-dry-run qlora-smoke20 task1 task2 assistant eval demo ci-test test verify release-check clean

SMOKE_LIMIT ?= 5

help:
	@echo "Traffic Legal VLM - common commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup       Install pinned Python dependencies"
	@echo "  make check-data  Check expected raw VLSP data paths"
	@echo "  make check-data-all  Check train, public, and private data paths"
	@echo "  make ci-test     Run lightweight schema tests used by CI"
	@echo "  make test        Run local unit tests with the macOS readline workaround"
	@echo ""
	@echo "Services:"
	@echo "  make qdrant-up   Start Qdrant with docker compose"
	@echo "  make qdrant-down Stop Qdrant"
	@echo ""
	@echo "Pipeline:"
	@echo "  make preprocess       Build processed LawDB articles"
	@echo "  make index            Build the LawDB text retrieval index"
	@echo "  make index-examples   Build the week-2 train-example index"
	@echo "  make index-week2      Build both week-2 retrieval indexes"
	@echo "  make benchmark-b2     Run B2 Text-RAG smoke benchmark"
	@echo "  make benchmark-b3     Run B3 fused-RAG smoke benchmark"
	@echo "  make benchmark-b4     Run B4 few-shot-RAG smoke benchmark"
	@echo "  make benchmark-w3-real  Run W3 non-mock configs with SMOKE_LIMIT"
	@echo "  make adapter-diagnostic  Run W4 QLoRA adapter diagnostic with SMOKE_LIMIT"
	@echo "  make qlora-dry-run    Validate QLoRA config without loading model weights"
	@echo "  make qlora-smoke20    GPU-only 20-sample QLoRA smoke command"
	@echo "  make demo             Start the Streamlit demo"
	@echo "  make release-check    Run final release preflight checks"

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

check-data:
	bash scripts/check_data.sh

check-data-all:
	bash scripts/check_data.sh --all

qdrant-up:
	docker compose up -d

qdrant-down:
	docker compose down

preprocess:
	python -m src.data_utils --mode preprocess

index:
	bash scripts/index.sh law

index-examples:
	bash scripts/index.sh examples

index-week2:
	bash scripts/index.sh week2

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

benchmark-b2:
	bash scripts/evaluate.sh run-experiment configs/experiments/w2_b2_text_rag.yaml $(SMOKE_LIMIT)

benchmark-b3:
	bash scripts/evaluate.sh run-experiment configs/experiments/w2_b3_fused_rag.yaml $(SMOKE_LIMIT)

benchmark-b4:
	bash scripts/evaluate.sh run-experiment configs/experiments/w2_b4_few_shot_rag.yaml $(SMOKE_LIMIT)

benchmark-week2-smoke: benchmark-b2 benchmark-b3 benchmark-b4

benchmark-w3-real:
	bash scripts/evaluate.sh run-w3-real $(SMOKE_LIMIT)

adapter-diagnostic:
	bash scripts/evaluate.sh adapter-diagnostic $(SMOKE_LIMIT)

qlora-dry-run:
	python -m src.train_qlora --config configs/qlora.yaml --dry-run

qlora-smoke20:
	python -m src.train_qlora --config configs/qlora.yaml --max-samples 20

demo:
	bash scripts/demo.sh

ci-test:
	python -c 'import sys, types; sys.modules.setdefault("readline", types.ModuleType("readline")); import pytest; raise SystemExit(pytest.main(["-q", "tests/test_schemas.py"], plugins=[]))'

test:
	python -c 'import sys, types; sys.modules.setdefault("readline", types.ModuleType("readline")); import pytest; raise SystemExit(pytest.main(["-q", "tests"], plugins=[]))'

verify: ci-test
	python -m pip check
	git diff --check

release-check: verify
	python -m src.evaluate --help >/dev/null
	python -m src.submission --help >/dev/null
	bash scripts/check_data.sh --all
	git status --ignored --short

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
