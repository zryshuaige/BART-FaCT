"""Inference + evaluation pipeline for SUMM-Lens.

Loads any model from the registry, runs the appropriate inference path
(seq2seq direct / causal-LM zero-shot / +CoD / +NLR / +CoD+NLR), and stores
both raw predictions and aggregated metrics under ``results/<run_name>/``.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from benchmark import full_benchmark, compute_rouge, compute_length_stats
from config import ModelConfig, get_device, get_model_config
from data_utils import (
    INPUT_FIELD,
    TARGET_FIELD,
    load_arxiv_dataset,
    load_pubmed_dataset,
    set_seed,
)
from methods.cod import chain_of_density
from methods.llm_summarizer import build_summarizer
from methods.nli_rerank import NLIReranker, nli_rerank
from methods.prompts import get_prompt

logger = logging.getLogger(__name__)


# ── Dataset loading helper ─────────────────────────────────────────────


def _load_test_split(dataset_name: str, num_samples: Optional[int]):
    if dataset_name == "arxiv":
        ds = load_arxiv_dataset()
    elif dataset_name == "pubmed":
        ds = load_pubmed_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    test_key = "test" if "test" in ds else "validation"
    test = ds[test_key]
    if num_samples and len(test) > num_samples:
        test = test.select(range(num_samples))
    return test


# ── Per-sample inference dispatch ──────────────────────────────────────


def _summarize_one(
    summarizer,
    article: str,
    *,
    dataset: str,
    use_cod: bool,
    use_nlr: bool,
    cod_iters: int,
    nlr_candidates: int,
    nlr_temperature: float,
    nlr_top_p: float,
    reranker: Optional[NLIReranker] = None,
) -> str:
    """Run the right inference variant for one article."""
    if use_nlr:
        # NLR with optional CoD inside.
        best, _ = nli_rerank(
            summarizer,
            article,
            dataset=dataset,
            n_candidates=nlr_candidates,
            temperature=nlr_temperature,
            top_p=nlr_top_p,
            reranker=reranker,
            use_cod=use_cod,
            cod_iters=cod_iters,
        )
        return best

    if use_cod and summarizer.is_causal_lm:
        return chain_of_density(
            summarizer, article, dataset=dataset, num_iters=cod_iters
        )

    # Plain zero-shot / seq2seq path.
    if summarizer.is_causal_lm:
        prompt = get_prompt(dataset, article)
        return summarizer.summarize(prompt, n=1, temperature=0.0)[0]
    return summarizer.summarize(article, n=1, temperature=0.0)[0]


# ── Main entry point ──────────────────────────────────────────────────


def evaluate_model(
    model_name: str,
    dataset_name: str = "arxiv",
    num_test_samples: int = 100,
    output_dir: str = "./results",
    device: Optional[torch.device] = None,
    compute_bertscore: bool = True,
    compute_meteor: bool = True,
    reranker: Optional[NLIReranker] = None,
) -> Tuple[Dict, List[str], List[str]]:
    """Run a single model end-to-end and return ``(results, summaries, references)``.

    Falls back to default decoding parameters; ``ModelConfig`` carries the
    inference-time augmentation flags (``use_cod`` / ``use_nlr``).
    """
    set_seed(42)
    device = device or get_device()

    cfg: ModelConfig = get_model_config(model_name)
    logger.info(f"Evaluating {cfg.name} ({cfg.description})")

    summarizer = build_summarizer(cfg, device=device)

    if cfg.use_nlr and reranker is None:
        # Single shared reranker for the whole run; loaded once.
        reranker = NLIReranker(device=device)

    test = _load_test_split(dataset_name, num_test_samples)
    articles = [s[INPUT_FIELD] for s in test]
    references = [s[TARGET_FIELD] for s in test]

    summaries: List[str] = []
    for article in tqdm(articles, desc=f"Generating ({cfg.name})"):
        try:
            out = _summarize_one(
                summarizer,
                article,
                dataset=dataset_name,
                use_cod=cfg.use_cod,
                use_nlr=cfg.use_nlr,
                cod_iters=cfg.cod_iters,
                nlr_candidates=cfg.nlr_candidates,
                nlr_temperature=cfg.nlr_temperature,
                nlr_top_p=cfg.nlr_top_p,
                reranker=reranker,
            )
        except Exception as e:
            logger.warning(f"Generation failed on one sample: {e}")
            out = ""
        summaries.append(out)

    # Free model weights before metric loading.
    del summarizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Metrics.
    rouge = compute_rouge(summaries, references)
    bench = full_benchmark(
        summaries,
        references,
        sources=articles,
        compute_bert=compute_bertscore,
        compute_met=compute_meteor,
    )

    results = {
        "model": cfg.name,
        "hf_path": cfg.hf_path,
        "dataset": dataset_name,
        "num_test_samples": len(articles),
        "is_causal_lm": cfg.is_causal_lm,
        "use_cod": cfg.use_cod,
        "use_nlr": cfg.use_nlr,
        "rouge": rouge,
        "benchmark": bench,
        "pred_length": compute_length_stats(summaries),
        "ref_length": compute_length_stats(references),
    }

    # Persist.
    out_dir = os.path.join(output_dir, f"{cfg.name}_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "input_preview": articles[i][:500] + "...",
                    "reference": references[i],
                    "prediction": summaries[i],
                }
                for i in range(min(50, len(articles)))
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        f"  ROUGE-1={rouge['rouge1']['fmeasure']:.4f}  "
        f"ROUGE-2={rouge['rouge2']['fmeasure']:.4f}  "
        f"ROUGE-L={rouge['rougeL']['fmeasure']:.4f}"
    )
    return results, summaries, references


# ── CLI ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    p = argparse.ArgumentParser(description="Evaluate a single SUMM-Lens model.")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--dataset", type=str, default="arxiv", choices=["arxiv", "pubmed"])
    p.add_argument("--num_test", type=int, default=100)
    p.add_argument("--output_dir", type=str, default="./results")
    p.add_argument("--no_bertscore", action="store_true")
    p.add_argument("--no_meteor", action="store_true")
    args = p.parse_args()

    evaluate_model(
        model_name=args.model,
        dataset_name=args.dataset,
        num_test_samples=args.num_test,
        output_dir=args.output_dir,
        compute_bertscore=not args.no_bertscore,
        compute_meteor=not args.no_meteor,
    )
