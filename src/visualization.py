"""Misc visualization utilities for SUMM-Lens.

This file is intentionally lightweight — primary plotting lives in `analyze.py`.
Here we keep small helpers used in notebooks: side-by-side text rendering,
NLI score distribution plots, length histograms.
"""

from __future__ import annotations

import html
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Re-export the font setup hook so notebooks can call it explicitly.
from analyze import _setup_chinese_font, COLORS  # noqa: F401


# ── Side-by-side rendering for notebooks ───────────────────────────────


def render_side_by_side(
    article: str,
    reference: str,
    predictions: Dict[str, str],
    max_article_chars: int = 2000,
) -> str:
    """Return an HTML string showing article / reference / predictions side-by-side.

    Use in Jupyter via:
        from IPython.display import HTML
        HTML(render_side_by_side(article, reference, {"BART": s1, "Qwen+CoD": s2}))
    """
    a = html.escape(article[:max_article_chars] + ("..." if len(article) > max_article_chars else ""))
    r = html.escape(reference)
    pred_blocks = []
    for name, s in predictions.items():
        pred_blocks.append(
            f"""<div style="border:1px solid #ddd; border-radius:6px; padding:10px;
                 margin-top:8px; background:#fafafa;">
                <b>{html.escape(name)}</b>
                <p style="white-space: pre-wrap; line-height:1.5;">{html.escape(s)}</p>
              </div>"""
        )
    return f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 1100px;">
      <h4 style="margin-bottom:6px;">Source article (truncated)</h4>
      <div style="border:1px solid #ddd; padding:10px; max-height:240px;
                  overflow-y:auto; background:#fff; line-height:1.5;">{a}</div>
      <h4 style="margin-top:14px; margin-bottom:6px;">Reference</h4>
      <div style="border:1px solid #b3d8ff; padding:10px; background:#f6faff;
                  line-height:1.5;">{r}</div>
      <h4 style="margin-top:14px; margin-bottom:6px;">Predictions</h4>
      {''.join(pred_blocks)}
    </div>
    """


# ── NLI score distribution ─────────────────────────────────────────────


def plot_nli_score_histogram(
    scores: List[float],
    output_path: str,
    title: str = "Per-candidate NLI entailment score",
):
    """Histogram of NLI faithfulness scores for diagnostic plots."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=20, color=COLORS[0], edgecolor="white")
    ax.set_xlabel("Mean entailment probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# ── Length histograms ─────────────────────────────────────────────────


def plot_length_distribution(
    summaries_by_model: Dict[str, List[str]],
    output_path: str,
    title: str = "Summary length distribution (words)",
):
    """Compare word-count distributions across models."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, summaries) in enumerate(summaries_by_model.items()):
        lens = [len(s.split()) for s in summaries]
        ax.hist(
            lens,
            bins=25,
            alpha=0.5,
            label=name,
            color=COLORS[i % len(COLORS)],
            edgecolor="white",
        )
    ax.set_xlabel("Word count", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
