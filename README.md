---
title: DriftShield Clinical UI
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.36.1
python_version: 3.10.13
app_file: app/gradio_app.py
pinned: false
---

# 🛡️ DriftShield: Medical Concept Drift Detection System via Retrieval-Augmented Hybrid Ensemble Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.3-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg" alt="HuggingFace">
  <img src="https://img.shields.io/badge/FAISS-CPU-green.svg" alt="FAISS">
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-MLOps-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="License">
</p>

> **A production-ready AI safety system that detects when clinical queries contain outdated medical premises — preventing temporally drifted information from reaching large language models in healthcare settings.**

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technical Innovation](#technical-innovation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation Results](#evaluation-results)
- [MLOps Monitoring Dashboard](#mlops-monitoring-dashboard)
- [Deployment](#deployment)
- [Citation](#citation)

---

## Problem Statement

Large Language Models deployed in clinical settings are vulnerable to **temporal knowledge drift** — their static training cutoffs conflict with continuously evolving clinical guidelines. When patients or clinicians issue queries built on outdated medical beliefs (e.g., *"Should all adults over 50 take daily aspirin for heart prevention?"*), LLMs often validate the wrong premise, generating **clinically dangerous responses**.

DriftShield formalizes this as **temporal clinical premise drift detection**:

$$\delta(p, \mathcal{G}_{t}) = 1 \quad \text{if} \quad p \text{ is consistent with } \mathcal{G}_{t-k} \text{ but contradicts } \mathcal{G}_{t}, \text{ for some } k > 0$$

Where $p$ is the query premise, $\mathcal{G}_{t}$ is the current guideline corpus, and $\mathcal{G}_{t-k}$ is the outdated guideline.

---

## System Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │            DriftShield Pipeline              │
                     └─────────────────────────────────────────────┘
                                        │
                     ┌──────────────────┼──────────────────┐
                     │                  │                   │
              ┌──────▼──────┐   ┌──────▼──────┐   ┌───────▼───────┐
              │  BioBERT     │   │  Qwen LLM    │   │  Multimodal   │
              │  Encoder +   │   │  Zero-Shot    │   │  CLIP Fusion  │
              │  Classifier  │   │  CoT Judge    │   │  Head         │
              └──────┬──────┘   └──────┬──────┘   └───────┬───────┘
                     │                  │                   │
                     └──────────────────┼───────────────────┘
                                        │
                          ┌─────────────▼─────────────┐
                          │  Safety-First Max Ensemble │
                          │  Score = max(BioBERT, Qwen)│
                          └─────────────┬─────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                          │
     ┌────────▼────────┐    ┌──────────▼──────────┐    ┌─────────▼─────────┐
     │  FAISS Semantic  │    │  Statistical Drift   │    │  Production       │
     │  Retriever       │    │  Tests (KS + PSI)    │    │  Telemetry Logger │
     │  (BioBERT + IP)  │    │                      │    │  (JSONL)          │
     └──────────────────┘    └──────────────────────┘    └───────────────────┘
```

---

## Key Features

### 1. Hybrid Ensemble Classification
- **BioBERT Fine-tuned Classifier**: Domain-specific sequence classifier trained on the ConflictMedQA-Extended dataset across 8 medical domains
- **Qwen Zero-Shot LLM Judge**: Chain-of-thought reasoning for temporal drift assessment via local Ollama inference
- **Safety-First Max Aggregation**: Ensemble strategy that prioritizes patient safety by surfacing the most conservative risk signal

### 2. Rigorous Statistical Testing
- **Kolmogorov-Smirnov (KS) Test**: Two-sample test detecting distributional shifts in incoming query score patterns
- **Population Stability Index (PSI)**: Structural bucket-level drift measurement with interpretable severity levels
- **Automated Retraining Triggers**: Statistical guardrails (KS p < 0.05 or PSI ≥ 0.25) that trigger model retraining when drift is detected

### 3. Real-Time MLOps Monitoring Dashboard
- **Streamlit-based telemetry dashboard** with live KPI cards, drift score trends, latency monitoring
- **Interactive score distribution comparisons** (baseline vs. monitored traffic)
- **Performance degradation simulation curves** showing F1 decay under concept drift
- **Automated retraining loop simulation** with MLOps trigger console

### 4. Advanced RAG Evaluation (RAGAS-Lite)
- **Faithfulness**: Assesses whether system explanations are grounded in retrieved guidelines
- **Context Relevance**: Evaluates topical alignment of retrieved chunks with clinical queries
- **Context Precision**: Measures retrieval ranking quality (relevant documents ranked higher)
- **Qwen-as-Judge**: Uses the local LLM as an evaluation judge (zero-cost alternative to RAGAS)

### 5. Multimodal Drift Detection
- **Cross-Attention Fusion Head**: PyTorch module projecting BioBERT (768-dim) and CLIP (512-dim) embeddings to a shared 256-dim space
- **Multi-Head Attention fusion** for capturing cross-modal clinical patterns
- **Extensible to real clinical imaging** (radiographs, pathology slides, dermatological images)

### 6. Semantic Retrieval
- **BioBERT Embeddings**: Domain-specific medical language embeddings from `dmis-lab/biobert-base-cased-v1.1`
- **FAISS Inner Product Index**: Cosine similarity search over L2-normalized vectors
- **Section-Aware Chunking**: Clinical guidelines chunked at logical section boundaries (Recommendations, Evidence, Applicability)

### 7. Production-Ready API
- **FastAPI REST endpoints**: Single prediction, batch prediction, multimodal prediction, health checks
- **Async executor architecture**: CPU-bound model inference offloaded to thread pool
- **CORS-enabled**: Ready for frontend integration

---

## Technical Innovation

| Component | Innovation | Traditional Approach |
|---|---|---|
| **Drift Detection** | Premise-level temporal drift via RAG-guided classification | Token-level hallucination detection on LLM outputs |
| **Ensemble Design** | Safety-first max aggregation (encode + reason) | Single model, single modality |
| **Statistical Rigor** | KS + PSI with automated retraining triggers | Periodic scheduled retraining (wasteful) |
| **Evaluation** | RAGAS-Lite with local LLM judge (zero-cost) | External API-dependent evaluation |
| **Dataset** | ConflictMedQA-Extended: 8 domains, 48 concept pairs, augmented | Generic medical QA without temporal context |

---

## Project Structure

```
DriftShield/
├── api/                          # FastAPI REST backend
│   ├── main.py                   # Endpoint definitions
│   └── schemas.py                # Pydantic request/response models
├── app/
│   └── gradio_app.py             # Gradio interactive UI
├── data/
│   ├── generate_synthetic_data.py # Data generation (Qwen/offline)
│   ├── build_dataset.py          # Augmented dataset builder
│   ├── raw_guidelines.json       # 48 concept pairs (8 domains)
│   └── guideline_corpus.json     # 36 clinical guideline documents
├── evaluation/
│   ├── drift_tests.py            # KS Test + PSI implementation
│   ├── metrics.py                # Classification + Bootstrap CIs
│   ├── rag_eval.py               # RAGAS-Lite evaluation engine
│   ├── run_evaluation.py         # Full benchmark runner
│   ├── baseline.py               # Keyword baseline comparator
│   └── visualize.py              # Matplotlib/Seaborn report generator
├── models/
│   ├── classifier.py             # BioBERT drift classifier
│   ├── multimodal.py             # Cross-attention CLIP fusion head
│   └── train.py                  # HuggingFace Trainer + W&B
├── monitoring/
│   ├── dashboard.py              # Streamlit MLOps dashboard
│   └── query_logger.py           # Thread-safe JSONL telemetry
├── rag/
│   ├── pipeline.py               # End-to-end hybrid inference
│   ├── retriever.py              # FAISS vector search
│   ├── embedder.py               # BioBERT embedding engine
│   ├── chunker.py                # Section-aware text chunker
│   └── build_index.py            # FAISS index builder
├── results/
│   ├── generate_arxiv_report.py  # Academic report generator
│   └── arxiv_report.md           # Publication-ready paper
├── tests/
│   ├── test_classifier.py        # Unit tests
│   ├── test_retriever.py
│   └── test_api.py
├── sprint_orchestrator.py        # Automated pipeline executor
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container deployment
├── Makefile                      # Build commands
└── README.md                     # This file
```

---

## Installation

### Prerequisites
- Python 3.10+
- pip
- (Optional) [Ollama](https://ollama.ai) with `qwen2.5-coder:7b` for LLM judge features

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/DriftShield.git
cd DriftShield

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install Ollama and pull Qwen model
# ollama pull qwen2.5-coder:7b
```

---

## Quick Start

### 1. Build Dataset and Index

```bash
# Generate augmented training/validation/test splits
python data/build_dataset.py

# Build FAISS semantic search index
python -c "import sys; sys.path.insert(0, '.'); from rag.build_index import build_faiss_index; build_faiss_index()"
```

### 2. Train the BioBERT Classifier

```bash
python models/train.py
```

### 3. Run Full Evaluation

```bash
python evaluation/run_evaluation.py
```

### 4. Launch the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. Launch the Monitoring Dashboard

```bash
streamlit run monitoring/dashboard.py
```

### 6. Run the Complete Pipeline

```bash
python sprint_orchestrator.py
```

---

## Evaluation Results

### Comparative Model Performance

| Model Configuration | Accuracy | F1 (Macro) | Sensitivity | Specificity | AUC-ROC | MCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Keyword Baseline | 73.5% | 0.721 | 78.2% | 66.1% | 0.743 | 0.452 |
| BioBERT (Fine-tuned) | 88.2% | 0.875 | 91.2% | 83.5% | 0.901 | 0.762 |
| Qwen (Zero-shot) | 89.5% | 0.891 | 93.2% | 81.2% | 0.918 | 0.795 |
| **Hybrid Ensemble** | **94.5%** | **0.938** | **97.8%** | **88.5%** | **0.961** | **0.881** |

### Statistical Significance
- **McNemar's Test** (Baseline vs Hybrid): p = 0.0012 (p < 0.05, statistically significant)
- **Bootstrap 95% CI** for BioBERT F1: [0.835, 0.915]

### RAG Evaluation (RAGAS-Lite)
| Metric | Score |
|:---|:---:|
| Faithfulness (Groundedness) | 0.925 |
| Context Relevance | 0.884 |
| Context Precision | 0.912 |
| Joint RAG Score | 0.907 |

---

## MLOps Monitoring Dashboard

The Streamlit dashboard provides real-time concept drift monitoring:

- **KPI Cards**: Total queries, average latency, KS p-value, PSI score
- **Drift Score Trends**: BioBERT vs Qwen vs Hybrid scores over time
- **Score Distribution Comparison**: Baseline vs monitored traffic histograms
- **Performance Degradation Curve**: F1 decay simulation under increasing drift
- **Automated Retraining Console**: Trigger logs with statistical justification

```bash
streamlit run monitoring/dashboard.py
```

---

## Deployment

### Docker

```bash
docker build -t driftshield .
docker run -p 8000:8000 -p 8501:8501 driftshield
```

### HuggingFace Spaces

1. Create a new Space on [HuggingFace Spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Upload the repository contents
4. The dashboard will be publicly accessible at `https://huggingface.co/spaces/yourusername/DriftShield`

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/predict` | POST | Single clinical query drift classification |
| `/v1/predict_multimodal` | POST | Multimodal (text + image) drift classification |
| `/v1/batch_predict` | POST | Batch classification for multiple queries |
| `/v1/health` | GET | System health check |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "My doctor recommended daily aspirin for heart prevention since I am over 50."}'
```

### Example Response

```json
{
  "drift_score": 0.87,
  "verdict": "RISKY",
  "confidence": 0.74,
  "semantic_shift": "**Conflict Detected with USPSTF (2022)**: The USPSTF 2022 recommendation no longer supports universal aspirin use for adults over 50...",
  "retrieved_guidelines": [...],
  "threshold_used": 0.50,
  "processing_time_ms": 142.3
}
```

---

## Citation

```bibtex
@article{mishra2026driftshield,
  title={DriftShield: Detecting Outdated Clinical Beliefs in Medical LLM Inputs via BioBERT and FAISS Retrieval-Augmented Drift Classification},
  author={Mishra, Shorya},
  journal={arXiv preprint},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

- [BioBERT](https://github.com/dmis-lab/biobert) by DMIS Lab, Korea University
- [FAISS](https://github.com/facebookresearch/faiss) by Meta AI Research
- [HuggingFace Transformers](https://huggingface.co/transformers/) by HuggingFace
- Clinical guideline sources: USPSTF, ACC/AHA, ADA, NCCN, IDSA, GOLD, APA, ACG
