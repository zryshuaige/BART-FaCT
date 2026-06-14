"""Model registry and runtime configuration for SUMM-Lens.

All entries are inference-only. The registry covers a 2019→2024 baseline ladder
plus four ablation configurations for the proposed CoD + NLR enhancements.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import copy

import torch


@dataclass
class ModelConfig:
    """Lightweight model descriptor."""

    name: str
    hf_path: str
    max_input_length: int
    max_target_length: int = 256
    is_causal_lm: bool = False  # True for chat LLMs (Qwen2.5-Instruct, Llama, ...)
    is_encoder_decoder: bool = True
    description: str = ""
    # SUMM-Lens inference-time augmentation flags (only for causal-LM rows).
    use_cod: bool = False
    use_nlr: bool = False
    cod_iters: int = 3
    nlr_candidates: int = 4
    nlr_temperature: float = 0.7
    nlr_top_p: float = 0.95


# ── Baseline ladder: 2019 → 2024 ───────────────────────────────────────

MODEL_CONFIGS = {
    # 2019 — original BART
    "bart-large-cnn": ModelConfig(
        name="bart-large-cnn",
        hf_path="facebook/bart-large-cnn",
        max_input_length=1024,
        max_target_length=256,
        description="BART-large fine-tuned on CNN/DailyMail (400M, 2019).",
    ),
    # 2020 — PEGASUS family
    "pegasus-arxiv": ModelConfig(
        name="pegasus-arxiv",
        hf_path="google/pegasus-arxiv",
        max_input_length=1024,
        max_target_length=256,
        description="PEGASUS fine-tuned on arXiv (568M, 2020).",
    ),
    "pegasus-cnn_dailymail": ModelConfig(
        name="pegasus-cnn_dailymail",
        hf_path="google/pegasus-cnn_dailymail",
        max_input_length=1024,
        max_target_length=256,
        description="PEGASUS fine-tuned on CNN/DailyMail (568M, 2020).",
    ),
    "distilbart-cnn-12-6": ModelConfig(
        name="distilbart-cnn-12-6",
        hf_path="sshleifer/distilbart-cnn-12-6",
        max_input_length=1024,
        max_target_length=256,
        description="Distilled BART-CNN, fast inference baseline (306M, 2020).",
    ),
    # 2020 — long-document specialist
    "led-large-arxiv": ModelConfig(
        name="led-large-arxiv",
        hf_path="allenai/led-large-16384-arxiv",
        max_input_length=4096,  # truncate from 16384 for free-tier RAM
        max_target_length=512,
        description="Longformer Encoder-Decoder fine-tuned on arXiv (460M, 2020).",
    ),
    # 2024 — instruction-tuned LLM (vanilla)
    "qwen2.5-1.5b": ModelConfig(
        name="qwen2.5-1.5b",
        hf_path="Qwen/Qwen2.5-1.5B-Instruct",
        max_input_length=8192,
        max_target_length=384,
        is_causal_lm=True,
        is_encoder_decoder=False,
        description="Qwen2.5-1.5B-Instruct — 2024 zero-shot baseline (Apache 2.0).",
    ),
    # SUMM-Lens variants (same backbone, different inference-time setup)
    "summlens-cod": ModelConfig(
        name="summlens-cod",
        hf_path="Qwen/Qwen2.5-1.5B-Instruct",
        max_input_length=8192,
        max_target_length=384,
        is_causal_lm=True,
        is_encoder_decoder=False,
        use_cod=True,
        use_nlr=False,
        description="Qwen2.5-1.5B + Chain-of-Density prompting.",
    ),
    "summlens-nlr": ModelConfig(
        name="summlens-nlr",
        hf_path="Qwen/Qwen2.5-1.5B-Instruct",
        max_input_length=8192,
        max_target_length=384,
        is_causal_lm=True,
        is_encoder_decoder=False,
        use_cod=False,
        use_nlr=True,
        description="Qwen2.5-1.5B + NLI-Rerank over 4 sampled candidates.",
    ),
    "summlens-full": ModelConfig(
        name="summlens-full",
        hf_path="Qwen/Qwen2.5-1.5B-Instruct",
        max_input_length=8192,
        max_target_length=384,
        is_causal_lm=True,
        is_encoder_decoder=False,
        use_cod=True,
        use_nlr=True,
        description="Qwen2.5-1.5B + CoD + NLR — proposed method.",
    ),
}


# Default rosters used by the experiment driver.
BASELINE_MODELS: List[str] = [
    "bart-large-cnn",
    "distilbart-cnn-12-6",
    "pegasus-arxiv",
    "led-large-arxiv",
    "qwen2.5-1.5b",
]

ABLATION_MODELS: List[str] = [
    "qwen2.5-1.5b",     # vanilla (no CoD, no NLR)
    "summlens-cod",      # +CoD
    "summlens-nlr",      # +NLR
    "summlens-full",     # +CoD +NLR
]


# ── Runtime config ─────────────────────────────────────────────────────


@dataclass
class RuntimeConfig:
    """Decoding parameters shared across all models."""

    beam_size: int = 4
    length_penalty: float = 2.0
    no_repeat_ngram_size: int = 3
    batch_size: int = 4
    seed: int = 42
    output_dir: str = "./results"


# ── Helpers ────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_available_models() -> List[str]:
    return list(MODEL_CONFIGS.keys())


def get_model_config(model_name: str) -> ModelConfig:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}"
        )
    return copy.deepcopy(MODEL_CONFIGS[model_name])
