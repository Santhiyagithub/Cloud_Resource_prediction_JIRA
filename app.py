"""
app.py
------
Cloud Resource Prediction — Gradio Web Interface
=================================================
Dataset schema: Alibaba Cluster Trace 2017 (machine_usage)
  Source: https://github.com/alibaba/clusterdata

Loads the trained Random Forest model and serves a prediction UI that:
  - Accepts current + historical CPU, Memory, and Network usage.
  - Returns the predicted next-step CPU utilization (%).
  - Gives an actionable capacity-planning recommendation.

Usage:  python app.py
"""

import os
import gradio as gr
import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[App] Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. Run 'python train.py' first."
    )

# Must match train.py FEATURES list exactly
FEATURES = ["cpu_util", "mem_util", "net_in", "net_out",
            "cpu_lag1", "cpu_lag2", "mem_lag1"]

# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict(cpu_util, mem_util, net_in, net_out,
            cpu_lag1, cpu_lag2, mem_lag1):
    """
    Predict next-step CPU utilization and provide a recommendation.

    Parameters (all floats, %)
    ----------
    cpu_util  : Current CPU utilization
    mem_util  : Current Memory utilization
    net_in    : Current inbound network (normalized 0-100)
    net_out   : Current outbound network (normalized 0-100)
    cpu_lag1  : CPU 1 interval ago
    cpu_lag2  : CPU 2 intervals ago
    mem_lag1  : Memory 1 interval ago
    """
    # Input validation
    for name, val in [("CPU", cpu_util), ("Memory", mem_util),
                      ("CPU lag1", cpu_lag1), ("CPU lag2", cpu_lag2),
                      ("Memory lag1", mem_lag1)]:
        if not (0 <= val <= 100):
            return None, f"Input error: {name} must be 0-100 (got {val})", "Error"

    X = pd.DataFrame(
        [[cpu_util, mem_util, net_in, net_out, cpu_lag1, cpu_lag2, mem_lag1]],
        columns=FEATURES
    )
    pred = float(model.predict(X)[0])
    pred = round(np.clip(pred, 0, 100), 2)

    if pred < 30:
        rec   = ("Low demand detected.\n"
                 "Recommendation: Consider scaling down idle instances "
                 "to reduce costs. Monitor for 2+ consecutive intervals before acting.")
        label = "Low Load"
    elif pred <= 75:
        rec   = ("Moderate demand detected.\n"
                 "Recommendation: Maintain current capacity. "
                 "Set auto-scale triggers at 80% to handle spikes proactively.")
        label = "Moderate Load"
    else:
        rec   = ("High demand detected.\n"
                 "Recommendation: Scale out immediately or investigate "
                 "the source of the spike. Check for runaway processes or traffic surge.")
        label = "High Load - Action Required"

    return pred, rec, label

# ---------------------------------------------------------------------------
# Example scenarios
# ---------------------------------------------------------------------------
EXAMPLES = [
    [20.0, 35.0, 15.0, 10.0, 22.0, 24.0, 36.0],  # Low load
    [60.0, 65.0, 45.0, 38.0, 58.0, 55.0, 63.0],  # Moderate load
    [88.0, 80.0, 72.0, 65.0, 85.0, 82.0, 78.0],  # High load / spike
    [45.0, 55.0, 30.0, 25.0, 47.0, 46.0, 54.0],  # Stable mid-range
]

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(
    title="Cloud Resource Prediction | Alibaba Cluster Schema",
    theme=gr.themes.Soft(primary_hue="indigo"),
) as demo:

    gr.Markdown(
        """
        # Cloud Resource Prediction
        **Predict next-step CPU utilization and get a capacity-planning recommendation.**

        > Dataset schema: [Alibaba Cluster Trace 2017](https://github.com/alibaba/clusterdata)
        > (synthetic data mirroring real schema) | Model: Random Forest Regressor
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Input — Current Resource State")
            cpu_in  = gr.Number(label="Current CPU Utilization (%)",     value=50.0, minimum=0, maximum=100)
            mem_in  = gr.Number(label="Current Memory Utilization (%)",  value=60.0, minimum=0, maximum=100)
            ni_in   = gr.Number(label="Network In (normalized 0-100)",   value=30.0, minimum=0, maximum=100)
            no_in   = gr.Number(label="Network Out (normalized 0-100)",  value=25.0, minimum=0, maximum=100)
            l1_in   = gr.Number(label="CPU — 1 interval ago (%)",        value=50.0, minimum=0, maximum=100)
            l2_in   = gr.Number(label="CPU — 2 intervals ago (%)",       value=50.0, minimum=0, maximum=100)
            ml1_in  = gr.Number(label="Memory — 1 interval ago (%)",     value=60.0, minimum=0, maximum=100)
            btn     = gr.Button("Predict", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Output — Forecast & Recommendation")
            pred_out = gr.Number(label="Predicted Next-Step CPU (%)")
            sev_out  = gr.Textbox(label="Load Level")
            rec_out  = gr.Textbox(label="Capacity-Planning Recommendation", lines=4)

    gr.Markdown("---")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[cpu_in, mem_in, ni_in, no_in, l1_in, l2_in, ml1_in],
        label="Example scenarios (click to load)",
    )

    btn.click(
        fn=predict,
        inputs=[cpu_in, mem_in, ni_in, no_in, l1_in, l2_in, ml1_in],
        outputs=[pred_out, rec_out, sev_out],
    )

    gr.Markdown(
        """
        ---
        | Predicted CPU | Level | Action |
        |---|---|---|
        | < 30 % | Low | Scale down to save cost |
        | 30–75 % | Moderate | Monitor; auto-scale at 80% |
        | > 75 % | High | Scale out or investigate spike |

        *Predictions are based on a model trained on synthetic data mirroring the*
        *Alibaba Cluster Trace 2017 schema. Not validated on live production data.*
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
