"""
DriftShield Gradio Dashboard App.

Implements the graphical front-end using Gradio blocks. Features interactive indicators,
gauge charts via Plotly, side-by-side model metrics, and multimodal image uploads.
"""

import gradio as gr
import plotly.graph_objects as go
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Fix for OpenMP/FAISS duplicate library crashes on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to path for local imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from rag.pipeline import DriftShieldPipeline

CLASSIFIER_CHECKPOINT: Path = Path("checkpoints/best_model")
INDEX_DIR: Path = Path("rag/index")

# Lazy loading pipeline wrapper to prevent failure if checkpoints do not exist yet
pipeline: Optional[DriftShieldPipeline] = None

def get_pipeline() -> DriftShieldPipeline:
    """Loads and returns the global pipeline instance, caching it after load."""
    global pipeline
    if pipeline is None:
        pipeline = DriftShieldPipeline.from_checkpoints(CLASSIFIER_CHECKPOINT, INDEX_DIR)
    return pipeline

EXAMPLES: List[List[str]] = [
    ["My doctor said I should take a daily baby aspirin since I turned 50 for heart protection.", "high"],
    ["Should I push for strict HbA1c below 6.5% as my diabetes target? My old doctor said that was the gold standard.", "high"],
    ["I read that platinum-based chemotherapy is always the first-line treatment for advanced lung cancer.", "high"],
    ["The new guidelines recommend against routine aspirin for primary prevention in adults over 60.", "safe"],
    ["My oncologist said immunotherapy is now first-line for NSCLC with high PD-L1 expression.", "safe"],
    ["I understand HbA1c targets should be individualized based on my age and health status.", "safe"],
]

def make_gauge(score: float, verdict: str) -> go.Figure:
    """Generates a Plotly gauge indicator chart showing prediction risk severity."""
    color = "#EF4444" if verdict == "RISKY" else "#22C55E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        title={"text": f"Ensemble Drift Score — {verdict}", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "#DCFCE7"},
                {"range": [30, 60], "color": "#FEF9C3"},
                {"range": [60, 100], "color": "#FEE2E2"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": 50},
        },
        number={"suffix": "%", "font": {"size": 24}},
    ))
    fig.update_layout(height=260, margin=dict(l=15, r=15, t=35, b=15), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def format_guidelines(retrieved: List[Dict[str, Any]]) -> str:
    """Formats retrieved list of guideline metadata maps to clean markdown strings."""
    if not retrieved:
        return "No relevant guidelines found."
    lines = ["### 📋 Retrieved Current Guidelines\n"]
    for i, g in enumerate(retrieved[:3], 1):
        lines.append(f"**[{i}] {g['source_name']} ({g['year']}) — {g['domain'].title()}**")
        lines.append(f"Relevance: `{g['score']:.2%}`")
        lines.append(f"> {g['text'][:300]}{'...' if len(g['text']) > 300 else ''}")
        lines.append("")
    return "\n".join(lines)

def predict(query: str, image_file: Optional[str], threshold: float) -> Tuple[float, str, str, go.Figure, str, str]:
    """Runs query statement analysis through pipeline and structures outputs for dashboard."""
    if not query or not query.strip():
        return 0.0, "—", "Please enter a clinical query.", make_gauge(0.0, "UNKNOWN"), "No input query provided.", ""
    try:
        pipe = get_pipeline()
        original_thresh = pipe.threshold
        pipe.threshold = threshold
        
        if image_file:
            result = pipe.predict_multimodal(query, image_file)
        else:
            result = pipe(query)
            
        pipe.threshold = original_thresh
    except Exception as e:
        return 0.0, "ERROR", f"Failed to run inference pipeline: {e}", make_gauge(0.0, "ERROR"), f"Pipeline error: {e}", ""

    guidelines_md = format_guidelines([
        {"source_name": c.source_name, "year": c.year, "domain": c.domain, "score": c.score, "text": c.text}
        for c in result.retrieved_guidelines
    ])
    
    summary = (
        f"### 🛡️ Decision Summary\n"
        f"- **Ensemble Verdict**: `{result.verdict}`\n"
        f"- **Ensemble Score**: `{result.drift_score:.2%}`\n"
        f"- **Confidence Level**: `{result.confidence:.2%}`\n"
        f"- **Modalities Fused**: `['text', 'image']` if result.is_multimodal else `['text']`"
    )
    
    explanation_md = (
        f"### 🔍 Detailed Medical Analysis & Explanation\n"
        f"{result.semantic_shift}"
    )
    
    return float(result.drift_score), result.verdict, guidelines_md, make_gauge(result.drift_score, result.verdict), summary, explanation_md

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    title="DriftShield — Medical Concept Drift Detector",
) as demo:
    gr.Markdown("""
    # 🛡️ DriftShield: Medical Concept Drift Detector & Monitor
    **Detects outdated clinical beliefs in patient queries and multimodal diagnostic reports before they reach a clinical LLM.**
    
    *BioBERT + FAISS RAG + Qwen Zero-Shot Ensemble | KS & PSI Drift Significance Testing | CLIP Image Fusion*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Clinical Query / Symptom Description (Text)",
                placeholder="e.g., 'My doctor said strict HbA1c below 6.5% is the gold standard for diabetes management'",
                lines=3,
            )
            image_input = gr.Image(
                label="Associated Diagnostic Image (Optional - Simulates CLIP visual embedding)",
                type="filepath",
            )
            threshold_slider = gr.Slider(0.2, 0.8, value=0.5, step=0.05, label="Detection Threshold")
            submit_btn = gr.Button("🔍 Analyze for Drift", variant="primary", size="lg")

        with gr.Column(scale=1):
            gauge_plot = gr.Plot(label="Drift Score Gauge")
            summary_output = gr.Markdown(label="Decision Summary")

    with gr.Row():
        explanation_output = gr.Markdown(label="Detailed Medical Explanation")

    guidelines_output = gr.Markdown(label="Retrieved Current Guidelines")

    gr.Examples(
        examples=[[e[0]] for e in EXAMPLES],
        inputs=query_input,
        label="Example Queries (try 3 outdated + 3 current)",
    )

    gr.Markdown("""
    ---
    **How it works:** DriftShield embeds clinical queries via BioBERT, retrieves the top 5 current guidelines from FAISS,
    performs zero-shot inference using Qwen, and aggregates their scores. Image uploads simulate multimodal CLIP projection.
    
    [GitHub](https://github.com/Shoryamishra61/driftshield) | [Paper](#) | [W&B Dashboard](#)
    """)

    submit_btn.click(
        fn=predict,
        inputs=[query_input, image_input, threshold_slider],
        outputs=[gr.Number(visible=False), gr.Textbox(visible=False), guidelines_output, gauge_plot, summary_output, explanation_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)

