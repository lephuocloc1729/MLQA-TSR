.PHONY: setup qdrant-up qdrant-down preprocess index task1 task2 eval demo test clean

setup:
	pip install -r requirements.txt

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
	python -m src.evaluate

demo:
	streamlit run app/streamlit_app.py

test:
	pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
