import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import logging
from typing import Dict, List

import torch
from transformers import Seq2SeqTrainingArguments

from config import (
    ModelConfig, TrainingConfig, MODEL_CONFIGS, get_model_config, get_device,
    get_bart_fact_config,
)
from data_utils import (
    load_arxiv_dataset, load_pubmed_dataset,
    prepare_dataset_for_model, prepare_dataset_for_bart_fact, set_seed,
)
from models.bart_fact import BARTFaCTForConditionalGeneration, BARTFaCTConfig, ABLATION_CONFIGS
from train import BARTFaCTTrainer, BARTFaCTDataCollator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Ablation model definitions ─────────────────────────────────────────

ABLATION_MODELS = {
    "bart_baseline": {
        "config": ABLATION_CONFIGS["bart_baseline"],
        "model_name": "bart-baseline",
        "description": "BART baseline (no novel modules) — pretrained bart-large-cnn",
    },
    "bart_fact_no_hse": {
        "config": ABLATION_CONFIGS["bart_fact_no_hse"],
        "model_name": "bart-fact-no-hse",
        "description": "BART-FaCT w/o HSE: BART + CFA + CPO (no hierarchical structure encoding)",
    },
    "bart_fact_no_cfa": {
        "config": ABLATION_CONFIGS["bart_fact_no_cfa"],
        "model_name": "bart-fact-no-cfa",
        "description": "BART-FaCT w/o CFA: BART + HSE + CPO (no calibrated faithfulness attention)",
    },
    "bart_fact_no_cpo": {
        "config": ABLATION_CONFIGS["bart_fact_no_cpo"],
        "model_name": "bart-fact-no-cpo",
        "description": "BART-FaCT w/o CPO: BART + HSE + CFA (no contrastive preference optimization)",
    },
    "bart_fact_full": {
        "config": ABLATION_CONFIGS["bart_fact_full"],
        "model_name": "bart-fact-full",
        "description": "BART-FaCT (Full): BART + HSE + CFA + CPO",
    },
}


# ── Single ablation run ────────────────────────────────────────────────

def run_single_ablation(
    ablation_name: str,
    dataset_name: str = "arxiv",
    max_samples: int = 1000,
    num_test: int = 100,
    output_dir: str = "./results/ablation",
    epochs: int = 3,
    learning_rate: float = 3e-5,
    batch_size: int = 4,
):
    if ablation_name not in ABLATION_MODELS:
        raise ValueError(
            f"Unknown ablation: {ablation_name}. "
            f"Available: {list(ABLATION_MODELS.keys())}"
        )

    ablation_info = ABLATION_MODELS[ablation_name]
    bart_fact_config = ablation_info["config"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Ablation Study: {ablation_name}")
    logger.info(f"  Description: {ablation_info['description']}")
    logger.info(
        f"  HSE: {bart_fact_config.use_hse}, "
        f"CFA: {bart_fact_config.use_cfa}, "
        f"CPO: {bart_fact_config.use_cpo}"
    )
    logger.info(f"{'='*60}")

    set_seed(42)
    device = get_device()

    ablation_dir = os.path.join(output_dir, ablation_name)
    os.makedirs(ablation_dir, exist_ok=True)

    config_copy = BARTFaCTConfig(
        use_hse=bart_fact_config.use_hse,
        use_cfa=bart_fact_config.use_cfa,
        use_cpo=bart_fact_config.use_cpo,
        hse_num_heads=bart_fact_config.hse_num_heads,
        hse_ffn_dim=bart_fact_config.hse_ffn_dim,
        hse_dropout=bart_fact_config.hse_dropout,
        cfa_bottleneck_dim=bart_fact_config.cfa_bottleneck_dim,
        cfa_dropout=bart_fact_config.cfa_dropout,
        cpo_projection_dim=bart_fact_config.cpo_projection_dim,
        cpo_temperature=bart_fact_config.cpo_temperature,
        cpo_beta=bart_fact_config.cpo_beta,
        cpo_alpha=bart_fact_config.cpo_alpha,
        dropout=bart_fact_config.dropout,
        base_model_name=bart_fact_config.base_model_name,
        max_input_length=bart_fact_config.max_input_length,
        max_target_length=bart_fact_config.max_target_length,
    )

    model = BARTFaCTForConditionalGeneration(config_copy)
    model = model.to(device)

    param_info = model.get_trainable_params_summary()
    logger.info(f"Model parameters: {param_info}")

    with open(os.path.join(ablation_dir, "model_params.json"), "w") as f:
        json.dump(param_info, f, indent=2)

    tokenizer = model.tokenizer

    dataset = prepare_dataset_for_bart_fact(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_input_length=config_copy.max_input_length,
        max_target_length=config_copy.max_target_length,
        max_samples=max_samples,
    )

    data_collator = BARTFaCTDataCollator(
        tokenizer=tokenizer,
        model=model.bart,
        padding=True,
    )

    use_cpo = config_copy.use_cpo
    cpo_alpha = config_copy.cpo_alpha

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(ablation_dir, "checkpoints"),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=epochs,
        warmup_steps=200,
        weight_decay=0.01,
        fp16=device.type == "cuda",
        gradient_checkpointing=False,
        logging_steps=50,
        evaluation_strategy="no",
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=config_copy.max_target_length,
        report_to=[],
        seed=42,
        remove_unused_columns=False,
        dataloader_num_workers=0 if device.type == "cpu" else 2,
        dataloader_pin_memory=device.type == "cuda",
    )

    trainer = BARTFaCTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", None),
        tokenizer=tokenizer,
        data_collator=data_collator,
        bart_fact_model=model,
        use_cpo=use_cpo,
        cpo_alpha=cpo_alpha,
    )

    logger.info(f"Starting training for ablation: {ablation_name}")
    train_result = trainer.train()

    model.save_pretrained(os.path.join(ablation_dir, "model"))

    logger.info(f"Training complete for {ablation_name}. Evaluating...")

    from evaluate import evaluate_model
    eval_results, summaries, references = evaluate_model(
        model_name=ablation_info["model_name"],
        dataset_name=dataset_name,
        num_test_samples=num_test,
        output_dir=ablation_dir,
        trained_model=model,
        trained_tokenizer=tokenizer,
    )

    from hallucination import evaluate_hallucination_for_model
    if dataset_name == "arxiv":
        ds = load_arxiv_dataset(max_samples=None)
    else:
        ds = load_pubmed_dataset(max_samples=None)
    test_key = "test" if "test" in ds else "validation"
    test_data = ds[test_key]
    if num_test and len(test_data) > num_test:
        test_data = test_data.select(range(num_test))
    source_texts = [sample["article"] for sample in test_data]

    hallucination_results = evaluate_hallucination_for_model(
        model_name=ablation_name,
        source_texts=source_texts[: len(summaries)],
        generated_summaries=summaries,
        references=references[: len(summaries)],
        use_nli=True,
        output_dir=os.path.join(ablation_dir, "hallucination"),
    )

    ablation_result = {
        "ablation_name": ablation_name,
        "description": ablation_info["description"],
        "use_hse": config_copy.use_hse,
        "use_cfa": config_copy.use_cfa,
        "use_cpo": config_copy.use_cpo,
        "model_params": param_info,
        "eval_results": eval_results,
        "hallucination_results": hallucination_results,
    }

    result_path = os.path.join(ablation_dir, "ablation_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(ablation_result, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Ablation {ablation_name} complete. Results saved to {result_path}")
    return ablation_result


# ── Run all ablations ──────────────────────────────────────────────────

def run_all_ablations(
    dataset_name: str = "arxiv",
    max_samples: int = 1000,
    num_test: int = 100,
    output_dir: str = "./results/ablation",
    epochs: int = 3,
    learning_rate: float = 3e-5,
    batch_size: int = 4,
    ablation_list: List[str] = None,
):
    if ablation_list is None:
        ablation_list = list(ABLATION_MODELS.keys())

    logger.info(f"\n{'='*60}")
    logger.info("Running Full Ablation Study (Module Ablation)")
    logger.info(f"Ablations to run: {ablation_list}")
    logger.info(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for ablation_name in ablation_list:
        logger.info(f"\n--- Running ablation: {ablation_name} ---")
        try:
            result = run_single_ablation(
                ablation_name=ablation_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                num_test=num_test,
                output_dir=output_dir,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )
            all_results[ablation_name] = result
        except Exception as e:
            logger.error(f"Failed ablation {ablation_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ablation_name] = {"error": str(e)}

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Build comparison summary
    compare_results = {}
    for name, result in all_results.items():
        if isinstance(result, dict) and "error" not in result:
            entry = {
                "use_hse": result.get("use_hse", False),
                "use_cfa": result.get("use_cfa", False),
                "use_cpo": result.get("use_cpo", False),
            }
            if "eval_results" in result and isinstance(result["eval_results"], dict):
                if "rouge" in result["eval_results"]:
                    rouge = result["eval_results"]["rouge"]
                    entry["rouge1"] = rouge.get("rouge1", {}).get("fmeasure", 0)
                    entry["rouge2"] = rouge.get("rouge2", {}).get("fmeasure", 0)
                    entry["rougeL"] = rouge.get("rougeL", {}).get("fmeasure", 0)
            if "hallucination_results" in result and isinstance(
                result["hallucination_results"], dict
            ):
                nli = result["hallucination_results"].get("nli_metrics", {})
                entry["factuality_rate"] = nli.get("factuality_rate", 0)
                entry["hallucination_rate"] = nli.get("hallucination_rate", 0)
            compare_results[name] = entry

    summary_path = os.path.join(output_dir, "ablation_comparison.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(compare_results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nAblation comparison saved to {summary_path}")
    logger.info("\nAblation Results Summary:")
    logger.info(
        f"{'Model':<25} {'HSE':>5} {'CFA':>5} {'CPO':>5} "
        f"{'R1':>8} {'R2':>8} {'RL':>8} {'Fact':>8} {'Hall':>8}"
    )
    logger.info("-" * 90)
    for name, entry in compare_results.items():
        logger.info(
            f"{name:<25} {str(entry.get('use_hse', '-')):>5} "
            f"{str(entry.get('use_cfa', '-')):>5} "
            f"{str(entry.get('use_cpo', '-')):>5} "
            f"{entry.get('rouge1', 0):>8.4f} "
            f"{entry.get('rouge2', 0):>8.4f} "
            f"{entry.get('rougeL', 0):>8.4f} "
            f"{entry.get('factuality_rate', 0):>8.4f} "
            f"{entry.get('hallucination_rate', 0):>8.4f}"
        )

    return all_results


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation experiments (module ablation)")
    parser.add_argument(
        "--ablation", type=str, default="all",
        choices=list(ABLATION_MODELS.keys()) + ["all"],
        help="Which ablation to run",
    )
    parser.add_argument("--dataset", type=str, default="arxiv", choices=["arxiv", "pubmed"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./results/ablation")

    args = parser.parse_args()

    if args.ablation == "all":
        run_all_ablations(
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            num_test=args.num_test,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )
    else:
        run_single_ablation(
            ablation_name=args.ablation,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            num_test=args.num_test,
            output_dir=args.output_dir,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
        )
