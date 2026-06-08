.PHONY: dataset train index eval serve dashboard test clean help

help:
	@echo "DriftShield Makefile targets:"
	@echo "  dataset   - Run dataset construction & augmentation splits"
	@echo "  train     - Run BioBERT sequence classifier training pipeline"
	@echo "  index     - Generate guidelines embeddings and build FAISS index"
	@echo "  eval      - Run comparative validation evaluation pipeline"
	@echo "  serve     - Start the FastAPI backend server on port 8000"
	@echo "  dashboard - Launch the local Gradio user dashboard interface"
	@echo "  test      - Run test suites via pytest"
	@echo "  clean     - Remove python caches and temporary folders"

dataset:
	python data/build_dataset.py

train:
	python models/train.py

index:
	python rag/build_index.py

eval:
	python evaluation/run_evaluation.py

serve:
	uvicorn api.main:app --reload --port 8000

dashboard:
	python app/gradio_app.py

test:
	pytest tests/ -v --cov=.

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
