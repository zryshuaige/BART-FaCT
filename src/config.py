from dataclasses import dataclass, field, asdict
from typing import Optional, List
import copy

import torch

from models.bart_fact import BARTFaCTConfig, ABLATION_CONFIGS


@dataclass
class ModelConfig:
    """Lightweight model descriptor."""

    name: str
    hf_path: str
    max_input_length: int
    max_target_length: int = 256
    is_bart_fact: bool = False
    is_encoder_decoder: bool = True
    description: str = ""


# ── Model registry ──────────────────────────────────────────────────────

MODEL_CONFIGS = {
    # ── Baselines ──────────────────────────────────────────────────
    "bart-large-cnn": ModelConfig(
        name="bart-large-cnn",
        hf_path="facebook/bart-large-cnn",
        max_input_length=1024,
        max_target_length=256,
        is_encoder_decoder=True,
        description="BART-large fine-tuned on CNN/DailyMail — primary baseline (400M)",
    ),
    "pegasus-cnn_dailymail": ModelConfig(
        name="pegasus-cnn_dailymail",
        hf_path="google/pegasus-cnn_dailymail",
        max_input_length=1024,
        max_target_length=256,
        is_encoder_decoder=True,
        description="PEGASUS fine-tuned on CNN/DailyMail (568M) — alternative summarization pre-training",
    ),
    "pegasus-arxiv": ModelConfig(
        name="pegasus-arxiv",
        hf_path="google/pegasus-arxiv",
        max_input_length=1024,
        max_target_length=256,
        is_encoder_decoder=True,
        description="PEGASUS fine-tuned on arXiv summarization (568M)",
    ),
    "pegasus-xsum": ModelConfig(
        name="pegasus-xsum",
        hf_path="google/pegasus-xsum",
        max_input_length=512,
        max_target_length=256,
        is_encoder_decoder=True,
        description="PEGASUS fine-tuned on XSum — extreme summarization (568M)",
    ),
    "distilbart-cnn-12-6": ModelConfig(
        name="distilbart-cnn-12-6",
        hf_path="sshleifer/distilbart-cnn-12-6",
        max_input_length=1024,
        max_target_length=256,
        is_encoder_decoder=True,
        description="DistilBART-CNN (306M) — distilled BART, fast inference baseline",
    ),

    # ── BART-FaCT variants (HSE / CFA / CPO) ───────────────────────
    "bart-fact-full": ModelConfig(
        name="bart-fact-full",
        hf_path="facebook/bart-large-cnn",
        max_input_length=1024,
        max_target_length=256,
        is_bart_fact=True,
        is_encoder_decoder=True,
        description="BART-FaCT (Full): BART + HSE + CFA + CPO",
    ),
    "bart-fact-no-hse": ModelConfig(
        name="bart-fact-no-hse",
        hf_path="facebook/bart-large-cnn",
        max_input_length=1024,
        max_target_length=256,
        is_bart_fact=True,
        is_encoder_decoder=True,
        description="BART-FaCT w/o HSE: BART + CFA + CPO (no hierarchical structure)",
    ),
    "bart-fact-no-cfa": ModelConfig(
        name="bart-fact-no-cfa",
        hf_path="facebook/bart-large-cnn",
        max_input_length=1024,
        max_target_length=256,
        is_bart_fact=True,
        is_encoder_decoder=True,
        description="BART-FaCT w/o CFA: BART + HSE + CPO (no calibrated faithfulness attention)",
    ),
    "bart-fact-no-cpo": ModelConfig(
        name="bart-fact-no-cpo",
        hf_path="facebook/bart-large-cnn",
        max_input_length=1024,
        max_target_length=256,
        is_bart_fact=True,
        is_encoder_decoder=True,
        description="BART-FaCT w/o CPO: BART + HSE + CFA (no preference optimization)",
    ),
    "bart-baseline": ModelConfig(
        name="bart-baseline",
        hf_path="facebook/bart-large-cnn",
        max_input_length=1024,
        max_target_length=256,
        is_bart_fact=False,
        is_encoder_decoder=True,
        description="BART baseline (no novel modules), same as bart-large-cnn",
    ),
}


def get_bart_fact_config(model_name: str) -> BARTFaCTConfig:
    """Map a model name to its BARTFaCTConfig (for ablation variants)."""
    config_map = {
        "bart-fact-full": ABLATION_CONFIGS["bart_fact_full"],
        "bart-fact-no-hse": ABLATION_CONFIGS["bart_fact_no_hse"],
        "bart-fact-no-cfa": ABLATION_CONFIGS["bart_fact_no_cfa"],
        "bart-fact-no-cpo": ABLATION_CONFIGS["bart_fact_no_cpo"],
        "bart-baseline": ABLATION_CONFIGS["bart_baseline"],
    }
    if model_name in config_map:
        return copy.deepcopy(config_map[model_name])
    raise ValueError(
        f"Unknown BART-FaCT config: {model_name}. "
        f"Available: {list(config_map.keys())}"
    )


# ── Training configuration ─────────────────────────────────────────────

@dataclass
class TrainingConfig:
    learning_rate: float = 3e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 200
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = False
    logging_steps: int = 50
    eval_steps: int = 250
    save_steps: int = 500
    save_total_limit: int = 3
    beam_size: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    output_dir: str = "./results"
    seed: int = 42
    dataset_name: str = "arxiv"
    model_name: str = "bart-large-cnn"
    max_samples: Optional[int] = None


# ── Experiment descriptors ─────────────────────────────────────────────

@dataclass
class ContextLengthExperiment:
    model_name: str
    context_lengths: List[int] = field(default_factory=lambda: [256, 512, 768, 1024])
    dataset_name: str = "arxiv"
    max_samples: Optional[int] = 5000


@dataclass
class HallucinationConfig:
    detector_model: str = "textattack/roberta-base-STS-B"
    similarity_threshold: float = 0.7
    nli_model: str = "roberta-large-mnli"


# ── Device helpers ─────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_available_models():
    return list(MODEL_CONFIGS.keys())


def get_model_config(model_name: str) -> ModelConfig:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MODEL_CONFIGS.keys())}"
        )
    return copy.deepcopy(MODEL_CONFIGS[model_name])
