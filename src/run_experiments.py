"""SUMM-Lens — unified experiment runner.

Entry points:

    # Smoke test (5 samples on arxiv, 3 models)
    python src/run_experiments.py --mode quick_test --dataset arxiv

    # 2019→2024 baseline ladder
    python src/run_experiments.py --mode baseline --dataset arxiv --num_test 100

    # 4-config ablation (Vanilla / +CoD / +NLR / +Both)
    python src/run_experiments.py --mode ablation --dataset arxiv --num_test 100

    # baseline + ablation + figures
    python src/run_experiments.py --mode all --dataset arxiv --num_test 100
"""

import argparse
import gc
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

# Ensure HuggingFace mirror BEFORE transformers/datasets import.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from analyze import (
    generate_latex_table,
    plot_ablation_comparison,
    plot_faithfulness_metrics,
    plot_rouge_comparison,
)
from config import ABLATION_MODELS, BASELINE_MODELS, get_available_models
from data_utils import set_seed
from evaluate import evaluate_model
from methods.nli_rerank import NLIReranker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── Single-stage runners ───────────────────────────────────────────────


def _run_models(
    model_list: List[str],
    dataset: str,
    num_test: int,
    output_dir: str,
    *,
    compute_bertscore: bool = True,
    compute_meteor: bool = True,
    shared_reranker: Optional[NLIReranker] = None,
) -> Dict[str, Dict]:
    """Run a list of models sequentially, collecting their results dicts."""
    results: Dict[str, Dict] = {}
    for name in model_list:
        logger.info("=" * 60)
        logger.info(f"Running {name} on {dataset}")
        logger.info("=" * 60)
        try:
            res, _, _ = evaluate_model(
                model_name=name,
                dataset_name=dataset,
                num_test_samples=num_test,
                output_dir=output_dir,
                compute_bertscore=compute_bertscore,
                compute_meteor=compute_meteor,
                reranker=shared_reranker,
            )
            results[name] = res
        except Exception as e:
            logger.error(f"Failed for {name}: {e}", exc_info=True)
            results[name] = {"error": str(e)}
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def run_baseline(dataset: str, num_test: int, output_dir: str, **kwargs) -> Dict:
    """Run the 2019→2024 baseline ladder (no SUMM-Lens variants)."""
    return _run_models(BASELINE_MODELS, dataset, num_test, output_dir, **kwargs)


def run_ablation(
    dataset: str, num_test: int, output_dir: str, **kwargs
) -> Dict:
    """Run the 4-config ablation. Shares a single NLIReranker across runs."""
    # NLR variants share the same NLI checkpoint — load it once.
    reranker = NLIReranker()
    return _run_models(
        ABLATION_MODELS,
        dataset,
        num_test,
        output_dir,
        shared_reranker=reranker,
        **kwargs,
    )


# ── End-to-end ────────────────────────────────────────────────────────


def run_all(
    dataset: str = "arxiv",
    num_test: int = 100,
    output_dir: str = "./results",
    skip_baseline: bool = False,
    skip_ablation: bool = False,
    skip_bertscore: bool = False,
    skip_meteor: bool = False,
):
    set_seed(42)
    run_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    all_results: Dict[str, Dict] = {}

    if not skip_baseline:
        baseline = run_baseline(
            dataset=dataset,
            num_test=num_test,
            output_dir=run_dir,
            compute_bertscore=not skip_bertscore,
            compute_meteor=not skip_meteor,
        )
        all_results.update(baseline)

    if not skip_ablation:
        ablation = run_ablation(
            dataset=dataset,
            num_test=num_test,
            output_dir=run_dir,
            compute_bertscore=not skip_bertscore,
            compute_meteor=not skip_meteor,
        )
        all_results.update(ablation)

    # Figures.
    figs = os.path.join(run_dir, "figures")
    valid = {k: v for k, v in all_results.items() if "error" not in v}
    if valid:
        plot_rouge_comparison(valid, figs, dataset)
        plot_faithfulness_metrics(valid, figs)
        ablation_only = {k: v for k, v in valid.items() if k in ABLATION_MODELS}
        if ablation_only:
            plot_ablation_comparison(ablation_only, figs)
        generate_latex_table(valid, figs)

    # Aggregate summary.
    with open(os.path.join(run_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print compact one-liner per model for quick inspection.
    print("\n" + "=" * 60)
    print(f"Summary on {dataset} ({num_test} samples)")
    print("=" * 60)
    for name, r in all_results.items():
        if "error" in r:
            print(f"  {name:24s}  ERROR: {r['error'][:60]}")
            continue
        rouge = r["rouge"]
        print(
            f"  {name:24s}  R1={rouge['rouge1']['fmeasure']:.4f}  "
            f"R2={rouge['rouge2']['fmeasure']:.4f}  "
            f"RL={rouge['rougeL']['fmeasure']:.4f}"
        )

    logger.info(f"\nDone. Results: {run_dir}")
    return run_dir


# ── CLI ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUMM-Lens experiment runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "baseline", "ablation", "single", "quick_test"],
    )
    parser.add_argument("--dataset", type=str, default="arxiv", choices=["arxiv", "pubmed"])
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="For --mode single: model name from the registry. "
             f"Available: {get_available_models()}",
    )
    parser.add_argument("--skip_bertscore", action="store_true")
    parser.add_argument("--skip_meteor", action="store_true")
    args = parser.parse_args()

    if args.mode == "all":
        run_all(
            dataset=args.dataset,
            num_test=args.num_test,
            output_dir=args.output_dir,
            skip_bertscore=args.skip_bertscore,
            skip_meteor=args.skip_meteor,
        )
    elif args.mode == "baseline":
        run_baseline(
            dataset=args.dataset,
            num_test=args.num_test,
            output_dir=args.output_dir,
            compute_bertscore=not args.skip_bertscore,
            compute_meteor=not args.skip_meteor,
        )
    elif args.mode == "ablation":
        run_ablation(
            dataset=args.dataset,
            num_test=args.num_test,
            output_dir=args.output_dir,
            compute_bertscore=not args.skip_bertscore,
            compute_meteor=not args.skip_meteor,
        )
    elif args.mode == "single":
        if not args.model:
            raise SystemExit("--mode single requires --model <name>")
        evaluate_model(
            model_name=args.model,
            dataset_name=args.dataset,
            num_test_samples=args.num_test,
            output_dir=args.output_dir,
            compute_bertscore=not args.skip_bertscore,
            compute_meteor=not args.skip_meteor,
        )
    elif args.mode == "quick_test":
        # Tiny run that exercises BART (seq2seq), Qwen (causal-LM zero-shot)
        # and SUMM-Lens (causal-LM with both modules) end-to-end.
        logger.info("Quick smoke test on 5 samples...")
        results = _run_models(
            ["bart-large-cnn", "qwen2.5-1.5b", "summlens-full"],
            dataset=args.dataset,
            num_test=5,
            output_dir=os.path.join(args.output_dir, "quick_test"),
            compute_bertscore=False,
            compute_meteor=False,
        )
        for k, v in results.items():
            if "error" in v:
                print(f"  {k}: ERROR {v['error'][:80]}")
            else:
                rouge = v["rouge"]
                print(
                    f"  {k}: R1={rouge['rouge1']['fmeasure']:.4f}  "
                    f"R2={rouge['rouge2']['fmeasure']:.4f}  "
                    f"RL={rouge['rougeL']['fmeasure']:.4f}"
                )
