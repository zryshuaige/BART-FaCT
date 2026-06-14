"""Plotting and table generation for SUMM-Lens experiment results.

All figures honor a Chinese-friendly font fallback so titles and labels render
correctly on Windows (SimHei), macOS (PingFang/STHeiti), and Linux (Noto CJK).
"""

import json
import logging
import os
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Chinese font fallback ──────────────────────────────────────────────


def _setup_chinese_font() -> Optional[str]:
    """Try a list of CJK fonts and pick the first one available on the system."""
    candidates = [
        "SimHei",
        "Microsoft YaHei",
        "Source Han Sans SC",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "PingFang SC",
        "STHeiti",
        "Hiragino Sans GB",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font] + plt.rcParams.get(
                "font.sans-serif", ["DejaVu Sans"]
            )
            logger.info(f"Using Chinese-capable font: {font}")
            return font
    logger.warning(
        "No CJK font found. Install one of: fonts-noto-cjk (Linux), or "
        "use Microsoft YaHei (Windows default)."
    )
    return None


_setup_chinese_font()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150


# ── Style ──────────────────────────────────────────────────────────────

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4", "#795548"]


# ── Display names ──────────────────────────────────────────────────────

MODEL_DISPLAY_NAMES = {
    "bart-large-cnn": "BART-Large-CNN (2019)",
    "pegasus-cnn_dailymail": "PEGASUS-CNN/DM (2020)",
    "pegasus-arxiv": "PEGASUS-arXiv (2020)",
    "distilbart-cnn-12-6": "DistilBART-CNN (2020)",
    "led-large-arxiv": "LED-arXiv (2020)",
    "qwen2.5-1.5b": "Qwen2.5-1.5B (2024)",
    "summlens-cod": "Qwen2.5 + CoD",
    "summlens-nlr": "Qwen2.5 + NLR",
    "summlens-full": "SUMM-Lens (Full)",
}

ABLATION_SHORT_NAMES = {
    "qwen2.5-1.5b": "Vanilla",
    "summlens-cod": "+CoD",
    "summlens-nlr": "+NLR",
    "summlens-full": "+CoD+NLR",
}


# ── Plot: model comparison (E1) ────────────────────────────────────────


def plot_rouge_comparison(
    results_dict: Dict[str, Dict],
    output_path: str = "./results/figures",
    dataset_name: str = "arxiv",
):
    os.makedirs(output_path, exist_ok=True)

    models = list(results_dict.keys())
    if not models:
        return None

    rouge1_f = [results_dict[m]["rouge"]["rouge1"]["fmeasure"] for m in models]
    rouge2_f = [results_dict[m]["rouge"]["rouge2"]["fmeasure"] for m in models]
    rougeL_f = [results_dict[m]["rouge"]["rougeL"]["fmeasure"] for m in models]
    display = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (values, title) in enumerate(
        [
            (rouge1_f, "ROUGE-1 F1"),
            (rouge2_f, "ROUGE-2 F1"),
            (rougeL_f, "ROUGE-L F1"),
        ]
    ):
        ax = axes[idx]
        bars = ax.bar(range(len(models)), values, color=COLORS[: len(models)])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(display, rotation=35, ha="right", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle(
        f"Model Performance Comparison ({dataset_name.upper()})",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "rouge_comparison.png"), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_path, "rouge_comparison.pdf"), bbox_inches="tight"
    )
    plt.close()

    df = pd.DataFrame(
        {
            "Model": display,
            "ROUGE-1": rouge1_f,
            "ROUGE-2": rouge2_f,
            "ROUGE-L": rougeL_f,
        }
    )
    df.to_csv(os.path.join(output_path, "rouge_comparison.csv"), index=False)
    return df


# ── Plot: ablation (E2) ────────────────────────────────────────────────


def plot_ablation_comparison(
    ablation_results: Dict[str, Dict],
    output_path: str = "./results/figures",
):
    os.makedirs(output_path, exist_ok=True)

    order = ["qwen2.5-1.5b", "summlens-cod", "summlens-nlr", "summlens-full"]
    available = [k for k in order if k in ablation_results]
    if not available:
        logger.warning("No ablation results available for plotting")
        return None

    names = [ABLATION_SHORT_NAMES.get(k, k) for k in available]
    rouge1 = [
        ablation_results[k].get("rouge", {}).get("rouge1", {}).get("fmeasure", 0)
        for k in available
    ]
    rouge2 = [
        ablation_results[k].get("rouge", {}).get("rouge2", {}).get("fmeasure", 0)
        for k in available
    ]
    rougeL = [
        ablation_results[k].get("rouge", {}).get("rougeL", {}).get("fmeasure", 0)
        for k in available
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, title in zip(
        axes,
        [rouge1, rouge2, rougeL],
        ["ROUGE-1", "ROUGE-2", "ROUGE-L"],
    ):
        bars = ax.bar(range(len(names)), vals, color=COLORS[: len(names)])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.2 if max(vals) > 0 else 1.0)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)

    plt.suptitle(
        "SUMM-Lens Module Ablation",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "ablation_comparison.png"), bbox_inches="tight"
    )
    plt.savefig(
        os.path.join(output_path, "ablation_comparison.pdf"), bbox_inches="tight"
    )
    plt.close()

    df = pd.DataFrame(
        {
            "Config": names,
            "CoD": [str(ablation_results[k].get("use_cod", "-")) for k in available],
            "NLR": [str(ablation_results[k].get("use_nlr", "-")) for k in available],
            "ROUGE-1": rouge1,
            "ROUGE-2": rouge2,
            "ROUGE-L": rougeL,
        }
    )
    df.to_csv(os.path.join(output_path, "ablation_comparison.csv"), index=False)
    return df


# ── Plot: faithfulness (BERTScore + repetition) — light proxy ─────────


def plot_faithfulness_metrics(
    results_dict: Dict[str, Dict],
    output_path: str = "./results/figures",
):
    """Plot BERTScore F1 and JS divergence side-by-side as faithfulness proxies."""
    os.makedirs(output_path, exist_ok=True)
    models = list(results_dict.keys())
    if not models:
        return None

    display = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    bert_f = []
    js = []
    for m in models:
        bench = results_dict[m].get("benchmark", {})
        bert_f.append(bench.get("bertscore", {}).get("bertscore_f1", 0))
        js.append(bench.get("js_divergence", {}).get("js_divergence_mean", 0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(range(len(models)), bert_f, color=COLORS[: len(models)])
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(display, rotation=35, ha="right", fontsize=10)
    axes[0].set_ylabel("BERTScore F1", fontsize=11)
    axes[0].set_title("Semantic similarity (higher = better)", fontsize=12, fontweight="bold")

    axes[1].bar(range(len(models)), js, color=COLORS[: len(models)])
    axes[1].set_xticks(range(len(models)))
    axes[1].set_xticklabels(display, rotation=35, ha="right", fontsize=10)
    axes[1].set_ylabel("JS divergence", fontsize=11)
    axes[1].set_title("Distribution distance to reference (lower = better)",
                      fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "faithfulness.png"), bbox_inches="tight")
    plt.savefig(os.path.join(output_path, "faithfulness.pdf"), bbox_inches="tight")
    plt.close()
    return None


# ── LaTeX table ────────────────────────────────────────────────────────


def generate_latex_table(
    results_dict: Dict[str, Dict],
    output_path: str = "./results/figures",
    caption: str = "Summarization performance comparison",
    label: str = "tab:rouge_results",
):
    os.makedirs(output_path, exist_ok=True)
    rows = []
    for name, r in results_dict.items():
        display = MODEL_DISPLAY_NAMES.get(name, name)
        r1 = r["rouge"]["rouge1"]["fmeasure"]
        r2 = r["rouge"]["rouge2"]["fmeasure"]
        rL = r["rouge"]["rougeL"]["fmeasure"]
        rows.append(f"{display} & {r1:.4f} & {r2:.4f} & {rL:.4f} \\\\")

    latex = (
        "\\begin{table}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\begin{tabular}{lccc}\n\\hline\n"
        "Model & ROUGE-1 & ROUGE-2 & ROUGE-L \\\\\n\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n\\end{tabular}\n\\end{table}"
    )
    with open(os.path.join(output_path, "results_table.tex"), "w", encoding="utf-8") as f:
        f.write(latex)
    return latex


# ── CLI ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate SUMM-Lens analysis figures.")
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--output_dir", type=str, default="./results/figures")
    p.add_argument("--dataset", type=str, default="arxiv")
    args = p.parse_args()

    all_results = {}
    for model_name in MODEL_DISPLAY_NAMES.keys():
        path = os.path.join(
            args.results_dir, f"{model_name}_{args.dataset}", "eval_results.json"
        )
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                all_results[model_name] = json.load(f)

    if all_results:
        plot_rouge_comparison(all_results, args.output_dir, args.dataset)
        plot_faithfulness_metrics(all_results, args.output_dir)
        generate_latex_table(all_results, args.output_dir)

        ablation_keys = ["qwen2.5-1.5b", "summlens-cod", "summlens-nlr", "summlens-full"]
        ablation_only = {k: v for k, v in all_results.items() if k in ablation_keys}
        if ablation_only:
            plot_ablation_comparison(ablation_only, args.output_dir)
        print(f"Figures written to {args.output_dir}")
    else:
        print(f"No eval_results.json files found under {args.results_dir}")
