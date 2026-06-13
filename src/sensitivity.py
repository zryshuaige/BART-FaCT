import os
import json
import logging
from typing import Dict, List

from config import ModelConfig, TrainingConfig, MODEL_CONFIGS, get_model_config, get_device
from data_utils import load_arxiv_dataset, load_pubmed_dataset, set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Beam size ──────────────────────────────────────────────────────────

def sensitivity_beam_size(
    model_name: str = "bart-fact-full",
    dataset_name: str = "arxiv",
    beam_sizes: List[int] = None,
    num_test: int = 100,
    output_dir: str = "./results/sensitivity",
):
    if beam_sizes is None:
        beam_sizes = [1, 2, 4, 6, 8]

    from evaluate import evaluate_model

    all_results = {}
    for beam in beam_sizes:
        logger.info(f"Sensitivity: beam_size={beam}")
        try:
            results, _, _ = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                num_test_samples=num_test,
                beam_size=beam,
                output_dir=os.path.join(output_dir, f"beam_{beam}"),
            )
            all_results[str(beam)] = {
                "beam_size": beam,
                "rouge": results.get("rouge", {}),
                "benchmark": results.get("benchmark", {}),
            }
        except Exception as e:
            logger.error(f"Failed at beam_size={beam}: {e}")
            all_results[str(beam)] = {"beam_size": beam, "error": str(e)}

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sensitivity_beam_size.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    return all_results


# ── Length penalty ─────────────────────────────────────────────────────

def sensitivity_length_penalty(
    model_name: str = "bart-fact-full",
    dataset_name: str = "arxiv",
    length_penalties: List[float] = None,
    num_test: int = 100,
    output_dir: str = "./results/sensitivity",
):
    if length_penalties is None:
        length_penalties = [0.6, 1.0, 1.5, 2.0, 2.5]

    from evaluate import evaluate_model

    all_results = {}
    for lp in length_penalties:
        logger.info(f"Sensitivity: length_penalty={lp}")
        try:
            results, _, _ = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                num_test_samples=num_test,
                length_penalty=lp,
                output_dir=os.path.join(output_dir, f"length_penalty_{lp}"),
            )
            all_results[str(lp)] = {
                "length_penalty": lp,
                "rouge": results.get("rouge", {}),
            }
        except Exception as e:
            logger.error(f"Failed at length_penalty={lp}: {e}")
            all_results[str(lp)] = {"length_penalty": lp, "error": str(e)}

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sensitivity_length_penalty.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    return all_results


# ── Learning rate ──────────────────────────────────────────────────────

def sensitivity_learning_rate(
    model_name: str = "bart-fact-full",
    dataset_name: str = "arxiv",
    learning_rates: List[float] = None,
    max_samples: int = 1000,
    num_test: int = 100,
    output_dir: str = "./results/sensitivity",
):
    if learning_rates is None:
        learning_rates = [1e-5, 3e-5, 5e-5, 1e-4]

    from train import train_model
    from evaluate import evaluate_model

    all_results = {}
    for lr in learning_rates:
        logger.info(f"Sensitivity: learning_rate={lr}")
        set_seed(42)

        config = TrainingConfig(
            dataset_name=dataset_name,
            model_name=model_name,
            max_samples=max_samples,
            num_train_epochs=3,
            learning_rate=lr,
            output_dir=os.path.join(output_dir, f"lr_{lr}"),
        )

        try:
            trainer, model, tokenizer = train_model(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                training_config=config,
            )
            results, _, _ = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                num_test_samples=num_test,
                output_dir=os.path.join(output_dir, f"lr_{lr}"),
                trained_model=model,
                trained_tokenizer=tokenizer,
            )
            all_results[str(lr)] = {
                "learning_rate": lr,
                "eval_results": results,
            }
        except Exception as e:
            logger.error(f"Failed at lr={lr}: {e}")
            all_results[str(lr)] = {"learning_rate": lr, "error": str(e)}

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sensitivity_learning_rate.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    return all_results


# ── CPO alpha ──────────────────────────────────────────────────────────

def sensitivity_cpo_alpha(
    model_name: str = "bart-fact-full",
    dataset_name: str = "arxiv",
    alphas: List[float] = None,
    max_samples: int = 1000,
    num_test: int = 100,
    output_dir: str = "./results/sensitivity",
):
    if alphas is None:
        alphas = [0.01, 0.05, 0.1, 0.2, 0.5]

    from train import train_model
    from evaluate import evaluate_model
    from config import get_bart_fact_config

    all_results = {}
    for alpha in alphas:
        logger.info(f"Sensitivity: CPO alpha={alpha}")

        bart_fact_config = get_bart_fact_config(model_name)
        bart_fact_config.cpo_alpha = alpha

        config = TrainingConfig(
            dataset_name=dataset_name,
            model_name=model_name,
            max_samples=max_samples,
            num_train_epochs=3,
            output_dir=os.path.join(output_dir, f"cpo_alpha_{alpha}"),
        )

        try:
            trainer, model, tokenizer = train_model(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                training_config=config,
                bart_fact_config_override=bart_fact_config,
            )
            results, _, _ = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                num_test_samples=num_test,
                output_dir=os.path.join(output_dir, f"cpo_alpha_{alpha}"),
                trained_model=model,
                trained_tokenizer=tokenizer,
            )
            all_results[str(alpha)] = {
                "cpo_alpha": alpha,
                "eval_results": results,
            }
        except Exception as e:
            logger.error(f"Failed at alpha={alpha}: {e}")
            all_results[str(alpha)] = {"cpo_alpha": alpha, "error": str(e)}

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sensitivity_cpo_alpha.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    return all_results


# ── CFA bottleneck dim ────────────────────────────────────────────────────

def sensitivity_cfa_dim(
    model_name: str = "bart-fact-full",
    dataset_name: str = "arxiv",
    dims: List[int] = None,
    max_samples: int = 1000,
    num_test: int = 100,
    output_dir: str = "./results/sensitivity",
):
    if dims is None:
        dims = [32, 64, 128, 256]

    from train import train_model
    from evaluate import evaluate_model
    from config import get_bart_fact_config

    all_results = {}
    for dim in dims:
        logger.info(f"Sensitivity: CFA bottleneck dim={dim}")

        bart_fact_config = get_bart_fact_config(model_name)
        bart_fact_config.cfa_bottleneck_dim = dim

        config = TrainingConfig(
            dataset_name=dataset_name,
            model_name=model_name,
            max_samples=max_samples,
            num_train_epochs=3,
            output_dir=os.path.join(output_dir, f"cfa_dim_{dim}"),
        )

        try:
            trainer, model, tokenizer = train_model(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                training_config=config,
                bart_fact_config_override=bart_fact_config,
            )
            results, _, _ = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                num_test_samples=num_test,
                output_dir=os.path.join(output_dir, f"cfa_dim_{dim}"),
                trained_model=model,
                trained_tokenizer=tokenizer,
            )
            all_results[str(dim)] = {
                "cfa_bottleneck_dim": dim,
                "eval_results": results,
            }
        except Exception as e:
            logger.error(f"Failed at dim={dim}: {e}")
            all_results[str(dim)] = {"cfa_bottleneck_dim": dim, "error": str(e)}

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sensitivity_cfa_dim.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    return all_results


# ── Epochs ─────────────────────────────────────────────────────────────

def sensitivity_epochs(
    model_name: str = "bart-fact-full",
    dataset_name: str = "arxiv",
    epochs: List[int] = None,
    max_samples: int = 1000,
    num_test: int = 100,
    output_dir: str = "./results/sensitivity",
):
    if epochs is None:
        epochs = [1, 2, 3, 5]

    from train import train_model
    from evaluate import evaluate_model

    all_results = {}
    for num_epochs in epochs:
        logger.info(f"Sensitivity: num_epochs={num_epochs}")
        set_seed(42)

        config = TrainingConfig(
            dataset_name=dataset_name,
            model_name=model_name,
            max_samples=max_samples,
            num_train_epochs=num_epochs,
            output_dir=os.path.join(output_dir, f"epochs_{num_epochs}"),
        )

        try:
            trainer, model, tokenizer = train_model(
                model_name=model_name,
                dataset_name=dataset_name,
                max_samples=max_samples,
                training_config=config,
            )
            results, _, _ = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                num_test_samples=num_test,
                output_dir=os.path.join(output_dir, f"epochs_{num_epochs}"),
                trained_model=model,
                trained_tokenizer=tokenizer,
            )
            train_metrics = trainer.state.log_history if hasattr(trainer, 'state') else []
            all_results[str(num_epochs)] = {
                "num_epochs": num_epochs,
                "eval_results": results,
                "train_loss_history": [
                    {"step": entry.get("step"), "loss": entry.get("loss"), "epoch": entry.get("epoch")}
                    for entry in train_metrics if "loss" in entry
                ],
            }
        except Exception as e:
            logger.error(f"Failed at epochs={num_epochs}: {e}")
            all_results[str(num_epochs)] = {"num_epochs": num_epochs, "error": str(e)}

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sensitivity_epochs.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    return all_results


# ── Truncation strategy ────────────────────────────────────────────────

def sensitivity_truncation_strategy(
    model_name: str = "bart-fact-full",
    dataset_name: str = "arxiv",
    strategies: List[str] = None,
    max_input_length: int = 1024,
    num_test: int = 100,
    output_dir: str = "./results/sensitivity",
):
    if strategies is None:
        strategies = ["head_only", "tail_only", "head_tail_mixed"]

    from transformers import AutoTokenizer
    from models.bart_fact import BARTFaCTForConditionalGeneration, BARTFaCTConfig
    from config import get_bart_fact_config
    from evaluate import compute_rouge, compute_length_stats

    model_config_info = get_model_config(model_name)
    bart_fact_config = get_bart_fact_config(model_name)

    checkpoint_dir = os.path.join(
        output_dir, f"{model_name}_{dataset_name}_ctx{max_input_length}"
    )
    if os.path.exists(os.path.join(checkpoint_dir, "bart_fact_config.json")):
        logger.info(f"Loading trained BART-FaCT model from {checkpoint_dir}")
        model = BARTFaCTForConditionalGeneration.from_pretrained(checkpoint_dir)
    else:
        logger.warning(f"No checkpoint found at {checkpoint_dir}, using untrained model")
        model = BARTFaCTForConditionalGeneration(bart_fact_config)
    model = model.to(get_device())
    tokenizer = model.tokenizer

    if dataset_name == "arxiv":
        ds = load_arxiv_dataset()
    else:
        ds = load_pubmed_dataset()
    test_key = "test" if "test" in ds else "validation"
    test_data = ds[test_key]
    if len(test_data) > num_test:
        test_data = test_data.select(range(num_test))

    all_results = {}
    for strategy in strategies:
        logger.info(
            f"Sensitivity: truncation_strategy={strategy}, ctx_len={max_input_length}"
        )

        truncated_texts = []
        for sample in test_data:
            text = sample["article"]
            tokens = tokenizer.encode(text, truncation=False)
            token_len = len(tokens)

            if token_len <= max_input_length:
                truncated_texts.append(text)
            elif strategy == "head_only":
                truncated_texts.append(
                    tokenizer.decode(tokens[:max_input_length], skip_special_tokens=True)
                )
            elif strategy == "tail_only":
                truncated_texts.append(
                    tokenizer.decode(tokens[-max_input_length:], skip_special_tokens=True)
                )
            elif strategy == "head_tail_mixed":
                half = max_input_length // 2
                truncated_texts.append(
                    tokenizer.decode(
                        tokens[:half] + tokens[-(max_input_length - half):],
                        skip_special_tokens=True,
                    )
                )

        from evaluate import generate_summaries
        references = [sample["abstract"] for sample in test_data]

        summaries = generate_summaries(
            model=model,
            tokenizer=tokenizer,
            texts=truncated_texts,
            max_input_length=max_input_length,
            max_target_length=model_config_info.max_target_length,
            batch_size=8,
            device=get_device(),
            is_bart_fact=True,
        )

        rouge_scores = compute_rouge(summaries, references)

        from benchmark import full_benchmark
        benchmark_results = full_benchmark(
            summaries, references, truncated_texts, compute_bert=False
        )

        result = {
            "strategy": strategy,
            "max_input_length": max_input_length,
            "rouge": rouge_scores,
            "benchmark": benchmark_results,
        }
        all_results[strategy] = result

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sensitivity_truncation.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    return all_results


# ── Run all ────────────────────────────────────────────────────────────

def run_all_sensitivity(
    model_name: str = "bart-fact-full",
    dataset_name: str = "arxiv",
    max_samples: int = 1000,
    num_test: int = 100,
    output_dir: str = "./results/sensitivity",
):
    logger.info("=" * 60)
    logger.info("Running full parameter sensitivity analysis")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    logger.info("\n--- Sensitivity 1: Beam Size ---")
    results["beam_size"] = sensitivity_beam_size(
        model_name=model_name, dataset_name=dataset_name,
        num_test=num_test, output_dir=output_dir,
    )

    logger.info("\n--- Sensitivity 2: Length Penalty ---")
    results["length_penalty"] = sensitivity_length_penalty(
        model_name=model_name, dataset_name=dataset_name,
        num_test=num_test, output_dir=output_dir,
    )

    logger.info("\n--- Sensitivity 3: CPO Alpha ---")
    results["cpo_alpha"] = sensitivity_cpo_alpha(
        model_name=model_name, dataset_name=dataset_name,
        max_samples=max_samples, num_test=num_test, output_dir=output_dir,
    )

    logger.info("\n--- Sensitivity 4: CFA Bottleneck Dim ---")
    results["cfa_dim"] = sensitivity_cfa_dim(
        model_name=model_name, dataset_name=dataset_name,
        max_samples=max_samples, num_test=num_test, output_dir=output_dir,
    )

    logger.info("\n--- Sensitivity 5: Learning Rate ---")
    results["learning_rate"] = sensitivity_learning_rate(
        model_name=model_name, dataset_name=dataset_name,
        max_samples=max_samples, num_test=num_test, output_dir=output_dir,
    )

    logger.info("\n--- Sensitivity 6: Epochs ---")
    results["epochs"] = sensitivity_epochs(
        model_name=model_name, dataset_name=dataset_name,
        max_samples=max_samples, num_test=num_test, output_dir=output_dir,
    )

    logger.info("\n--- Sensitivity 7: Truncation Strategy ---")
    results["truncation"] = sensitivity_truncation_strategy(
        model_name=model_name, dataset_name=dataset_name,
        num_test=num_test, output_dir=output_dir,
    )

    summary_path = os.path.join(output_dir, "sensitivity_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    return results


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run parameter sensitivity analysis")
    parser.add_argument(
        "--analysis", type=str, required=True,
        choices=["beam_size", "length_penalty", "learning_rate",
                  "cpo_alpha", "cfa_dim", "epochs", "truncation", "all"],
        help="Which sensitivity analysis to run",
    )
    parser.add_argument("--model", type=str, default="bart-fact-full")
    parser.add_argument("--dataset", type=str, default="arxiv", choices=["arxiv", "pubmed"])
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--num_test", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./results/sensitivity")

    args = parser.parse_args()

    func_map = {
        "beam_size": sensitivity_beam_size,
        "length_penalty": sensitivity_length_penalty,
        "learning_rate": sensitivity_learning_rate,
        "cpo_alpha": sensitivity_cpo_alpha,
        "cfa_dim": sensitivity_cfa_dim,
        "epochs": sensitivity_epochs,
        "truncation": sensitivity_truncation_strategy,
    }

    if args.analysis == "all":
        run_all_sensitivity(
            model_name=args.model, dataset_name=args.dataset,
            max_samples=args.max_samples, num_test=args.num_test,
            output_dir=args.output_dir,
        )
    else:
        func = func_map[args.analysis]
        func(
            model_name=args.model, dataset_name=args.dataset,
            max_samples=args.max_samples, num_test=args.num_test,
            output_dir=args.output_dir,
        )
