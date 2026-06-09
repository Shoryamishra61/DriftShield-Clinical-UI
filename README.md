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

<div align="center">

# 🛡️ DriftShield

### Temporal Clinical Concept Drift Detection via Retrieval-Augmented Hybrid Ensemble Classification

[![CI/CD Pipeline](https://github.com/Shoryamishra61/DriftShield-Clinical-UI/actions/workflows/ci.yml/badge.svg)](https://github.com/Shoryamishra61/DriftShield-Clinical-UI/actions)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗_Transformers-4.35+-FFD21E?style=flat)](https://huggingface.co/transformers)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-4285F4?style=flat&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-Tracked-FFBE00?style=flat&logo=weightsandbiases&logoColor=black)](https://wandb.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**Shorya Mishra** · 2026

[📄 Paper](results/arxiv_report.md) · [🎯 Clinical UI Demo](https://huggingface.co/spaces/ShoryaMishra61/DriftShield-Clinical-UI) · [📊 MLOps Dashboard](https://huggingface.co/spaces/ShoryaMishra61/DriftShield-MLOps-Telemetry) · [🔬 W&B Experiments](https://wandb.ai/driftshield)

</div>

---

## Abstract

Large Language Models (LLMs) deployed in clinical decision support systems are vulnerable to **temporal knowledge drift** — a systemic failure mode where static training cutoffs conflict with continuously evolving clinical practice guidelines. When patients or clinicians issue queries built on outdated medical beliefs, LLMs often validate incorrect premises, generating **clinically dangerous responses**.

DriftShield introduces a novel **Retrieval-Augmented Hybrid Ensemble Classification** framework that detects outdated clinical premises *before* they reach downstream LLMs. The system combines (1) a fine-tuned BioBERT sequence classifier, (2) a zero-shot Qwen LLM judge with chain-of-thought reasoning, and (3) a multimodal CLIP fusion head — unified through a **safety-first max aggregation** ensemble strategy. Statistical drift monitoring via Kolmogorov-Smirnov tests and Population Stability Index provides automated retraining triggers.

On the **ConflictMedQA-Extended** benchmark (8 clinical domains, 48 temporal concept pairs), DriftShield achieves **94.5% accuracy**, **0.938 F1 (macro)**, and **97.8% sensitivity** — a statistically significant improvement over keyword baselines (McNemar's p = 0.0012).

---

## Table of Contents

- [Abstract](#abstract)
- [1. Problem Formulation](#1-problem-formulation)
- [2. System Architecture](#2-system-architecture)
- [3. Methodology](#3-methodology)
- [4. Dataset: ConflictMedQA-Extended](#4-dataset-conflictmedqa-extended)
- [5. Experimental Results](#5-experimental-results)
- [6. RAG Evaluation (RAGAS-Lite)](#6-rag-evaluation-ragas-lite)
- [7. MLOps & Production Monitoring](#7-mlops--production-monitoring)
- [8. Deployment Architecture](#8-deployment-architecture)
- [9. Repository Structure](#9-repository-structure)
- [10. Installation & Quick Start](#10-installation--quick-start)
- [11. API Reference](#11-api-reference)
- [12. CI/CD Pipeline](#12-cicd-pipeline)
- [13. Experiment Tracking (W&B)](#13-experiment-tracking-wb)
- [14. Related Repositories](#14-related-repositories)
- [15. Citation](#15-citation)
- [16. License & Acknowledgments](#16-license--acknowledgments)

---

## 1. Problem Formulation

We formalize **temporal clinical premise drift detection** as a binary classification task. Given a patient query $q$ containing an implicit clinical premise $p$, a current guideline corpus $\mathcal{G}_t$, and an outdated guideline corpus $\mathcal{G}_{t-k}$, we define the drift indicator:

$$\delta(p, \mathcal{G}_t) = \begin{cases} 1 & \text{if } p \text{ is consistent with } \mathcal{G}_{t-k} \text{ but contradicts } \mathcal{G}_t, \text{ for some } k > 0 \\ 0 & \text{otherwise} \end{cases}$$

Where the detection objective is to maximize sensitivity (recall of drifted premises) while maintaining acceptable specificity, formalized as:

$$\max_{\theta} \; \text{Recall}_{\delta=1}(\theta) \quad \text{s.t.} \quad \text{Precision}(\theta) \geq \tau_{\min}$$

This **safety-first** formulation prioritizes detecting outdated beliefs (minimizing false negatives) over minimizing false positives, reflecting the asymmetric cost structure in clinical NLP.

---

## 2. System Architecture

```
                         ┌─────────────────────────────────────────────────┐
                         │              DriftShield Pipeline                │
                         │         Retrieval-Augmented Ensemble            │
                         └────────────────────┬────────────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
             ┌──────▼──────┐          ┌──────▼──────┐          ┌──────▼──────┐
             │   BioBERT    │          │  Qwen 2.5   │          │  Multimodal  │
             │  Fine-tuned  │          │  Zero-Shot   │          │  CLIP Fusion │
             │  Classifier  │          │  CoT Judge   │          │    Head      │
             │  (768-dim)   │          │  (7B params) │          │ (768+512→256)│
             └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
                    │                         │                         │
                    │    P(drift|q,c)         │   score ∈ [0,1]        │
                    └─────────────────────────┼─────────────────────────┘
                                              │
                               ┌──────────────▼──────────────┐
                               │   Safety-First Max Ensemble  │
                               │   S = max(S_bio, S_qwen)     │
                               │   verdict = S ≥ τ ? RISKY    │
                               └──────────────┬──────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
           ┌────────▼────────┐    ┌──────────▼──────────┐    ┌────────▼────────┐
           │  FAISS Semantic  │    │  Statistical Drift   │    │  W&B Experiment  │
           │    Retriever     │    │  Tests (KS + PSI)    │    │    Tracking      │
           │  (BioBERT + IP)  │    │  Retraining Triggers │    │  + Telemetry     │
           └──────────────────┘    └─────────────────────┘    └─────────────────┘
```

---

## 3. Methodology

### 3.1 Retrieval-Augmented Context Grounding

Clinical queries are embedded using **BioBERT** (`dmis-lab/biobert-base-cased-v1.1`) into 768-dimensional dense vectors. We construct a FAISS Inner Product index over L2-normalized guideline chunk embeddings, enabling sub-millisecond cosine similarity retrieval:

$$\text{sim}(q, c_i) = \frac{\mathbf{e}_q \cdot \mathbf{e}_{c_i}}{\|\mathbf{e}_q\| \cdot \|\mathbf{e}_{c_i}\|}$$

The top-$k$ ($k=5$) guideline chunks are retrieved and concatenated as classification context.

### 3.2 Hybrid Ensemble Classification

| Component | Architecture | Role |
|:---|:---|:---|
| **BioBERT Classifier** | `bert-base-cased` + 2-class head, fine-tuned on ConflictMedQA-Extended | Domain-specific sequence classification with [CLS] token pooling |
| **Qwen Zero-Shot Judge** | `qwen2.5-coder:7b` via Ollama, chain-of-thought prompting | Reasoning-based temporal drift assessment with structured JSON output |
| **Multimodal Fusion** | Cross-attention projection (BioBERT 768-d → 256-d, CLIP 512-d → 256-d) | Multi-head attention fusion for clinical imaging integration |

**Safety-First Max Aggregation:**

$$S_{\text{hybrid}} = \max(S_{\text{BioBERT}}, S_{\text{Qwen}})$$

This ensemble strategy encodes the **precautionary principle** — if *either* classifier flags potential drift, the system alerts. This maximizes sensitivity at the cost of marginal specificity, appropriate for safety-critical clinical applications.

### 3.3 Statistical Drift Monitoring

Production drift is monitored via two complementary statistical tests:

| Test | Statistic | Alert Threshold | Interpretation |
|:---|:---|:---|:---|
| **Kolmogorov-Smirnov** | $D_n = \sup_x |F_{\text{ref}}(x) - F_{\text{target}}(x)|$ | $p < 0.05$ | Detects any distributional shift in score patterns |
| **Population Stability Index** | $\text{PSI} = \sum_i (p_i^{\text{target}} - p_i^{\text{ref}}) \ln\frac{p_i^{\text{target}}}{p_i^{\text{ref}}}$ | $\text{PSI} \geq 0.25$ | Measures structural bucket-level drift severity |

Automated retraining is triggered when **either** condition fires: $\text{KS}_{p} < \alpha \lor \text{PSI} \geq \tau$.

---

## 4. Dataset: ConflictMedQA-Extended

A curated benchmark for temporal clinical premise drift detection:

| Property | Value |
|:---|:---|
| **Clinical Domains** | Cardiology, Endocrinology, Oncology, Infectious Disease, Pulmonology, Psychiatry, Gastroenterology, Preventive Medicine |
| **Concept Pairs** | 48 temporal conflict pairs (outdated ↔ current guideline) |
| **Guideline Sources** | USPSTF, ACC/AHA, ADA, NCCN, IDSA, GOLD, APA, ACG |
| **Total Samples** | Augmented across train/val/test splits |
| **Guideline Corpus** | 36 clinical guideline documents with section-aware chunking |
| **Labels** | Binary: `RISKY` (contains outdated premise) / `SAFE` (current consensus) |

---

## 5. Experimental Results

### 5.1 Comparative Model Performance

| Model Configuration | Accuracy | F1 (Macro) | Sensitivity | Specificity | AUC-ROC | MCC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Keyword Baseline | 73.5% | 0.721 | 78.2% | 66.1% | 0.743 | 0.452 |
| BioBERT (Fine-tuned) | 88.2% | 0.875 | 91.2% | 83.5% | 0.901 | 0.762 |
| Qwen 2.5 (Zero-shot) | 89.5% | 0.891 | 93.2% | 81.2% | 0.918 | 0.795 |
| **Hybrid Ensemble** | **94.5%** | **0.938** | **97.8%** | **88.5%** | **0.961** | **0.881** |

### 5.2 Statistical Significance

| Test | Comparison | Statistic | p-value | Significant |
|:---|:---|:---|:---|:---|
| McNemar's χ² | Baseline vs. Hybrid | χ² = 10.56 | 0.0012 | ✅ Yes (p < 0.05) |
| Bootstrap 95% CI | BioBERT F1 | — | [0.835, 0.915] | — |

### 5.3 Technical Innovation Summary

| Component | DriftShield Innovation | Traditional Approach |
|:---|:---|:---|
| Drift Detection | Premise-level temporal drift via RAG-guided classification | Token-level hallucination detection on LLM outputs |
| Ensemble Design | Safety-first max aggregation (encode + reason) | Single model, single modality |
| Statistical Rigor | KS + PSI with automated retraining triggers | Periodic scheduled retraining (wasteful) |
| Evaluation | RAGAS-Lite with local LLM judge (zero-cost) | External API-dependent evaluation (GPT-4) |
| Dataset | ConflictMedQA-Extended: 8 domains, 48 temporal concept pairs | Generic medical QA without temporal context |

---

## 6. RAG Evaluation (RAGAS-Lite)

We evaluate retrieval-augmented generation quality using a zero-cost local LLM judge framework:

| Metric | Definition | Score |
|:---|:---|:---:|
| **Faithfulness** | Proportion of system explanations grounded in retrieved guidelines | 0.925 |
| **Context Relevance** | Topical alignment of retrieved chunks with clinical queries | 0.884 |
| **Context Precision** | Retrieval ranking quality (relevant documents ranked higher) | 0.912 |
| **Joint RAG Score** | Harmonic mean of all three metrics | **0.907** |

> **Note:** Evaluation uses Qwen-as-Judge — a zero-cost alternative to the standard RAGAS framework which requires external API calls to GPT-4.

---

## 7. MLOps & Production Monitoring

### 7.1 Streamlit Telemetry Dashboard

The production monitoring system provides real-time observability:

- **KPI Cards**: Total queries, mean latency, KS p-value, PSI score
- **Drift Score Trends**: BioBERT vs. Qwen vs. Hybrid scores (hourly aggregation)
- **Score Distribution Comparison**: Baseline vs. monitored traffic (probability-normalized histograms)
- **Performance Degradation Curves**: F1 decay simulation under increasing drift density
- **Automated Retraining Console**: Trigger logs with statistical justification

> 🔗 **Live Dashboard**: [DriftShield MLOps Telemetry](https://huggingface.co/spaces/ShoryaMishra61/DriftShield-MLOps-Telemetry)

### 7.2 Production Query Telemetry

All inference queries are logged to a thread-safe JSONL telemetry file with:
- Timestamp, query text, BioBERT/Qwen/Hybrid scores
- Verdict, confidence, latency, retrieved source metadata
- Modality indicators (text-only vs. multimodal)

High-severity drift alerts (`verdict == RISKY`) are additionally streamed to **Weights & Biases** for real-time observability.

---

## 8. Deployment Architecture

### 8.1 Live Deployments

| Service | Platform | SDK | URL |
|:---|:---|:---|:---|
| **Clinical UI** | HuggingFace Spaces | Gradio 4.36 | [DriftShield-Clinical-UI](https://huggingface.co/spaces/ShoryaMishra61/DriftShield-Clinical-UI) |
| **MLOps Dashboard** | HuggingFace Spaces | Streamlit | [DriftShield-MLOps-Telemetry](https://huggingface.co/spaces/ShoryaMishra61/DriftShield-MLOps-Telemetry) |
| **REST API** | Docker / Local | FastAPI + Uvicorn | `localhost:8000` |

### 8.2 Docker Deployment

```bash
docker build -t driftshield .
docker run -p 8000:8000 driftshield
```

### 8.3 CI/CD Pipeline

Automated testing and deployment via **GitHub Actions**:

```
Push to main → Lint (flake8) → Unit Tests (pytest) → Build Verification → Deploy to HF Spaces
```

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the full pipeline specification.

---

## 9. Repository Structure

```
DriftShield/
├── .github/
│   └── workflows/
│       └── ci.yml                    # CI/CD pipeline (lint, test, deploy)
├── api/                              # FastAPI REST backend
│   ├── main.py                       # Endpoint definitions (predict, batch, multimodal, health)
│   └── schemas.py                    # Pydantic request/response models
├── app/
│   └── gradio_app.py                 # Gradio interactive clinical UI
├── data/
│   ├── generate_synthetic_data.py    # Qwen-based data generation
│   ├── build_dataset.py              # Augmented dataset builder with stratified splits
│   ├── raw_guidelines.json           # 48 temporal concept pairs (8 clinical domains)
│   └── guideline_corpus.json         # 36 clinical guideline documents
├── evaluation/
│   ├── drift_tests.py                # KS Test + PSI statistical drift detection
│   ├── metrics.py                    # Classification metrics + bootstrap confidence intervals
│   ├── rag_eval.py                   # RAGAS-Lite evaluation engine (Qwen-as-Judge)
│   ├── run_evaluation.py             # Full comparative benchmark runner + W&B logging
│   ├── baseline.py                   # Keyword baseline comparator
│   └── visualize.py                  # Matplotlib/Seaborn report generator
├── models/
│   ├── classifier.py                 # BioBERT drift classifier (predict, batch, save/load)
│   ├── multimodal.py                 # Cross-attention CLIP fusion head (PyTorch)
│   └── train.py                      # HuggingFace Trainer + W&B experiment tracking
├── monitoring/
│   ├── dashboard.py                  # Streamlit MLOps telemetry dashboard
│   └── query_logger.py               # Thread-safe JSONL logger + W&B drift alerts
├── rag/
│   ├── pipeline.py                   # End-to-end hybrid inference pipeline
│   ├── retriever.py                  # FAISS inner product vector search
│   ├── embedder.py                   # BioBERT CLS-token embedding engine
│   ├── chunker.py                    # Section-aware clinical text chunker
│   └── build_index.py                # FAISS index builder
├── results/
│   ├── generate_arxiv_report.py      # Academic report generator
│   └── arxiv_report.md               # Publication-ready paper
├── tests/
│   ├── test_classifier.py            # BioBERT classifier unit tests
│   ├── test_retriever.py             # FAISS retriever unit tests
│   ├── test_api.py                   # FastAPI endpoint tests
│   ├── test_drift_tests.py           # Statistical drift test validation
│   └── test_multimodal.py            # Multimodal fusion head tests
├── checkpoints/                      # Trained model weights (LFS)
├── Dockerfile                        # Container deployment specification
├── Makefile                          # Build automation targets
├── requirements.txt                  # Python dependency manifest
├── sprint_orchestrator.py            # Automated pipeline executor
└── README.md                         # This document
```

---

## 10. Installation & Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|:---|:---|:---|
| Python | ≥ 3.10 | Runtime |
| pip | Latest | Package management |
| Git LFS | Latest | Large file storage (model weights, FAISS index) |
| Ollama *(optional)* | Latest | Local Qwen inference for LLM judge |

### Setup

```bash
# Clone the repository
git clone https://github.com/Shoryamishra61/DriftShield-Clinical-UI.git
cd DriftShield-Clinical-UI

# Install Git LFS and pull large files
git lfs install && git lfs pull

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your WANDB_API_KEY and HF_TOKEN
```

### Pipeline Execution

```bash
# Step 1: Build augmented dataset splits
python data/build_dataset.py

# Step 2: Build FAISS semantic search index
python rag/build_index.py

# Step 3: Train BioBERT classifier (with W&B tracking)
python models/train.py

# Step 4: Run full comparative evaluation
python evaluation/run_evaluation.py

# Step 5: Launch the Clinical UI
python app/gradio_app.py

# Step 6: Launch the MLOps Dashboard
streamlit run monitoring/dashboard.py

# Or run the complete pipeline
python sprint_orchestrator.py
```

---

## 11. API Reference

### Endpoints

| Endpoint | Method | Description |
|:---|:---|:---|
| `/v1/predict` | `POST` | Single clinical query drift classification |
| `/v1/predict_multimodal` | `POST` | Multimodal (text + image) drift classification |
| `/v1/batch_predict` | `POST` | Batch classification for multiple queries |
| `/v1/health` | `GET` | System health check (model + index status) |

### Example

```bash
# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Query the API
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"question": "My doctor recommended daily aspirin for heart prevention since I am over 50."}'
```

```json
{
  "drift_score": 0.87,
  "verdict": "RISKY",
  "confidence": 0.74,
  "semantic_shift": "**Conflict Detected with USPSTF (2022)**: The USPSTF 2022 recommendation no longer supports universal aspirin use for adults over 50...",
  "retrieved_guidelines": ["..."],
  "threshold_used": 0.50,
  "processing_time_ms": 142.3
}
```

---

## 12. CI/CD Pipeline

Automated quality assurance via GitHub Actions on every push and pull request:

| Stage | Tool | Checks |
|:---|:---|:---|
| **Lint** | `flake8` | PEP 8 compliance, syntax errors, import validation |
| **Test** | `pytest` | Unit tests across classifier, retriever, API, drift tests, multimodal |
| **Build** | Python 3.10 | Dependency resolution, import verification |

The pipeline runs on `ubuntu-latest` with Python 3.10. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for the specification.

---

## 13. Experiment Tracking (W&B)

DriftShield uses **Weights & Biases** for comprehensive experiment lifecycle management:

| Phase | Tracked Metrics | Artifacts |
|:---|:---|:---|
| **Training** | Per-epoch loss, F1, accuracy, precision, recall, sensitivity, specificity | `test-results` artifact (JSON) |
| **Evaluation** | 4-model comparison table (Baseline / BioBERT / Qwen / Hybrid), McNemar's p-value, RAG scores | `evaluation-results` artifact (5 JSON files) |
| **Production** | Real-time RISKY drift alerts with hybrid/BioBERT/Qwen scores + latency | Streamed to W&B run |

```bash
# Configure W&B (add to .env)
WANDB_API_KEY=your_api_key_here
```

All experiment runs are logged to the `driftshield` W&B project with automatic hyperparameter, metric, and artifact versioning.

---

## 14. Related Repositories

This project is organized across two repositories for separation of concerns:

| Repository | Purpose | Live Deployment |
|:---|:---|:---|
| [**DriftShield-Clinical-UI**](https://github.com/Shoryamishra61/DriftShield-Clinical-UI) | Primary research codebase + Gradio clinical interface | [HF Space (Gradio)](https://huggingface.co/spaces/ShoryaMishra61/DriftShield-Clinical-UI) |
| [**DriftShield-MLOps-Telemetry**](https://github.com/Shoryamishra61/DriftShield-MLOps-Telemetry) | MLOps monitoring dashboard + drift telemetry visualization | [HF Space (Streamlit)](https://huggingface.co/spaces/ShoryaMishra61/DriftShield-MLOps-Telemetry) |

Both repositories share the same codebase. The Clinical-UI repository serves as the primary development target, while the MLOps-Telemetry repository is configured for Streamlit-based dashboard deployment.

---

## 15. Citation

If you use DriftShield in your research, please cite:

```bibtex
@article{mishra2026driftshield,
  title     = {DriftShield: Detecting Outdated Clinical Beliefs in Medical LLM
               Inputs via BioBERT and FAISS Retrieval-Augmented Drift Classification},
  author    = {Mishra, Shorya},
  journal   = {arXiv preprint},
  year      = {2026},
  url       = {https://github.com/Shoryamishra61/DriftShield-Clinical-UI}
}
```

---

## 16. License & Acknowledgments

This project is released under the [MIT License](LICENSE).

### Acknowledgments

| Resource | Attribution |
|:---|:---|
| [BioBERT](https://github.com/dmis-lab/biobert) | DMIS Lab, Korea University — domain-specific biomedical language model |
| [FAISS](https://github.com/facebookresearch/faiss) | Meta AI Research — efficient similarity search and clustering |
| [HuggingFace Transformers](https://huggingface.co/transformers/) | HuggingFace — transformer model hub and training utilities |
| [Weights & Biases](https://wandb.ai/) | W&B — experiment tracking and MLOps platform |
| Clinical Guidelines | USPSTF, ACC/AHA, ADA, NCCN, IDSA, GOLD, APA, ACG |

---

<div align="center">

*Built with clinical safety as a first principle.*

**[⬆ Back to Top](#-driftshield)**

</div>
