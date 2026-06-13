import os
import json
import logging
from typing import List, Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import ModelConfig, MODEL_CONFIGS, get_model_config, get_device, get_bart_fact_config
from data_utils import load_arxiv_dataset, load_pubmed_dataset, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── ROUGE ──────────────────────────────────────────────────────────────

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, Dict[str, float]]:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True
    )
    scores_p = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}
    scores_r = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}
    scores_f = {"rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": []}

    for pred, ref in zip(predictions, references):
        if not pred.strip():
            for key in scores_p:
                scores_p[key].append(0.0)
                scores_r[key].append(0.0)
                scores_f[key].append(0.0)
            continue
        score = scorer.score(ref, pred)
        for k in scores_p:
            scores_p[k].append(score[k].precision)
            scores_r[k].append(score[k].recall)
            scores_f[k].append(score[k].fmeasure)

    return {
        k: {
            "precision": float(np.mean(scores_p[k])),
            "recall": float(np.mean(scores_r[k])),
            "fmeasure": float(np.mean(scores_f[k])),
        }
        for k in scores_p
    }


def compute_length_stats(texts: List[str]) -> Dict[str, float]:
    lengths = [len(text.split()) for text in texts]
    return {
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(np.min(lengths)) if lengths else 0,
        "max_length": int(np.max(lengths)) if lengths else 0,
    }


# ── Generation ─────────────────────────────────────────────────────────

def generate_summaries(
    model,
    tokenizer,
    texts: List[str],
    max_input_length: int,
    max_target_length: int = 256,
    beam_size: int = 4,
    length_penalty: float = 2.0,
    no_repeat_ngram_size: int = 3,
    batch_size: int = 8,
    device=None,
    is_bart_fact: bool = False,
):
    if device is None:
        device = get_device()

    model.eval()
    all_summaries = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating summaries"):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)

        gen_kwargs = {
            "max_length": max_target_length,
            "num_beams": beam_size,
            "length_penalty": length_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "early_stopping": True,
        }

        with torch.no_grad():
            if is_bart_fact:
                from models.hierarchical_structure import batch_detect_boundaries
                boundary_mask = batch_detect_boundaries(
                    batch, tokenizer, max_input_length
                ).to(device)

                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    boundary_mask=boundary_mask,
                    input_texts=batch,
                    **gen_kwargs,
                )
            else:
                outputs = model.generate(**inputs, **gen_kwargs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_summaries.extend(decoded)

    return all_summaries


# ── Model loading ──────────────────────────────────────────────────────

def load_model_and_tokenizer(model_config_or_name, device=None, checkpoint_dir=None):
    if isinstance(model_config_or_name, str):
        model_config = get_model_config(model_config_or_name)
    else:
        model_config = model_config_or_name

    if device is None:
        device = get_device()

    model_path = checkpoint_dir if checkpoint_dir else model_config.hf_path
    tokenizer_path = checkpoint_dir if checkpoint_dir else model_config.hf_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return model, tokenizer


# ── Main evaluation ────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    dataset_name: str = "arxiv",
    max_samples: int = None,
    num_test_samples: int = 100,
    max_input_length: int = None,
    beam_size: int = 4,
    length_penalty: float = 2.0,
    batch_size: int = 8,
    output_dir: str = "./results",
    device=None,
    trained_model=None,
    trained_tokenizer=None,
):
    set_seed(42)
    if device is None:
        device = get_device()

    model_config = get_model_config(model_name)
    if max_input_length is not None:
        max_input_length = min(max_input_length, model_config.max_input_length)
    else:
        max_input_length = model_config.max_input_length

    logger.info(f"Evaluating {model_name} on {dataset_name} (ctx={max_input_length})")

    is_bart_fact = model_config.is_bart_fact

    checkpoint_dir = os.path.join(
        output_dir,
        f"{model_name}_{dataset_name}_ctx{max_input_length}",
    )

    # Load or reuse model
    if trained_model is not None and trained_tokenizer is not None:
        model = trained_model
        tokenizer = trained_tokenizer
        model.eval()
    elif is_bart_fact:
        from models.bart_fact import BARTFaCTForConditionalGeneration
        bart_fact_config = get_bart_fact_config(model_name)
        bart_fact_config.max_input_length = max_input_length
        bart_fact_config.max_target_length = model_config.max_target_length

        if os.path.exists(os.path.join(checkpoint_dir, "bart_fact_config.json")):
            logger.info(f"Loading trained BART-FaCT model from {checkpoint_dir}")
            model = BARTFaCTForConditionalGeneration.from_pretrained(checkpoint_dir)
            model = model.to(device)
            tokenizer = model.tokenizer
        else:
            logger.warning(f"No checkpoint found at {checkpoint_dir}, using untrained model")
            model = BARTFaCTForConditionalGeneration(bart_fact_config)
            model = model.to(device)
            tokenizer = model.tokenizer
    else:
        if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
            logger.info(f"Loading trained model from {checkpoint_dir}")
            model, tokenizer = load_model_and_tokenizer(
                model_config, device, checkpoint_dir=checkpoint_dir
            )
        else:
            logger.warning(f"No checkpoint found at {checkpoint_dir}, using pretrained model")
            model, tokenizer = load_model_and_tokenizer(model_config, device)

    # Load dataset
    if dataset_name == "arxiv":
        ds = load_arxiv_dataset(max_samples=None)
    elif dataset_name == "pubmed":
        ds = load_pubmed_dataset(max_samples=None)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_key = "test" if "test" in ds else "validation"
    test_data = ds[test_key]
    if num_test_samples and len(test_data) > num_test_samples:
        test_data = test_data.select(range(num_test_samples))

    input_field = "article"
    target_field = "abstract"

    texts = [sample[input_field] for sample in test_data]
    references = [sample[target_field] for sample in test_data]

    # Generate
    summaries = generate_summaries(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        max_input_length=max_input_length,
        max_target_length=model_config.max_target_length,
        beam_size=beam_size,
        length_penalty=length_penalty,
        batch_size=batch_size,
        device=device,
        is_bart_fact=is_bart_fact,
    )

    # Evaluate
    rouge_scores = compute_rouge(summaries, references)
    pred_length_stats = compute_length_stats(summaries)
    ref_length_stats = compute_length_stats(references)

    from benchmark import full_benchmark
    bench_results = full_benchmark(
        summaries, references, texts,
        compute_bert=True, compute_met=True,
    )

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "max_input_length": max_input_length,
        "num_test_samples": len(texts),
        "beam_size": beam_size,
        "length_penalty": length_penalty,
        "is_bart_fact": is_bart_fact,
        "rouge": rouge_scores,
        "benchmark": bench_results,
        "pred_length_stats": pred_length_stats,
        "ref_length_stats": ref_length_stats,
    }

    # Save
    result_dir = os.path.join(
        output_dir, f"{model_name}_{dataset_name}_ctx{max_input_length}"
    )
    os.makedirs(result_dir, exist_ok=True)

    with open(os.path.join(result_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(result_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "input": texts[i][:500] + "...",
                    "reference": references[i],
                    "prediction": summaries[i],
                }
                for i in range(min(50, len(texts)))
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        f"Results: ROUGE-1={rouge_scores['rouge1']['fmeasure']:.4f}, "
        f"ROUGE-2={rouge_scores['rouge2']['fmeasure']:.4f}, "
        f"ROUGE-L={rouge_scores['rougeL']['fmeasure']:.4f}"
    )

    return results, summaries, references


# ── Context length sweep ───────────────────────────────────────────────

def evaluate_context_length_impact(
    model_name: str,
    context_lengths: List[int],
    dataset_name: str = "arxiv",
    num_test_samples: int = 100,
    beam_size: int = 4,
    output_dir: str = "./results",
):
    model_config = get_model_config(model_name)
    valid_lengths = [cl for cl in context_lengths if cl <= model_config.max_input_length]

    all_results = {}
    for ctx_len in valid_lengths:
        logger.info(f"\nEvaluating {model_name} with context length {ctx_len}")
        try:
            results, _, _ = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                num_test_samples=num_test_samples,
                max_input_length=ctx_len,
                beam_size=beam_size,
                output_dir=output_dir,
            )
            all_results[ctx_len] = results
        except Exception as e:
            logger.error(f"Failed at context length {ctx_len}: {e}")
            all_results[ctx_len] = {"error": str(e)}

    summary_path = os.path.join(
        output_dir, f"context_length_impact_{model_name}_{dataset_name}.json"
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    return all_results


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate summarization models")
    parser.add_argument("--model", type=str, default="bart-large-cnn")
    parser.add_argument("--dataset", type=str, default="arxiv", choices=["arxiv", "pubmed"])
    parser.add_argument("--max_input_length", type=int, default=None)
    parser.add_argument("--num_test_samples", type=int, default=100)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--context_lengths", type=str, default=None)

    args = parser.parse_args()

    if args.context_lengths:
        ctx_lengths = [int(x) for x in args.context_lengths.split(",")]
        results = evaluate_context_length_impact(
            model_name=args.model,
            context_lengths=ctx_lengths,
            dataset_name=args.dataset,
            num_test_samples=args.num_test_samples,
            beam_size=args.beam_size,
            output_dir=args.output_dir,
        )
    else:
        results, _, _ = evaluate_model(
            model_name=args.model,
            dataset_name=args.dataset,
            num_test_samples=args.num_test_samples,
            max_input_length=args.max_input_length,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
