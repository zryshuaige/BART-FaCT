"""Dataset loaders for arXiv and PubMed long-document summarization.

Uses HuggingFace datasets with a Chinese mirror by default. Local on-disk caches
under ``data/<name>/`` are reused if present.
"""

import os
import random

import numpy as np
import torch

# Force HuggingFace mirror for users in mainland China — set BEFORE the datasets
# import so the mirror takes effect at module load time.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_TIMEOUT", "120")

import datasets

datasets.config.HF_ENDPOINT = os.environ["HF_ENDPOINT"]

import huggingface_hub

if hasattr(huggingface_hub, "constants"):
    huggingface_hub.constants.HF_ENDPOINT = os.environ["HF_ENDPOINT"]
if hasattr(huggingface_hub, "HF_ENDPOINT"):
    huggingface_hub.HF_ENDPOINT = os.environ["HF_ENDPOINT"]

from datasets import DatasetDict, load_dataset

SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_mirror() -> None:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_TIMEOUT"] = "120"
    datasets.config.HF_ENDPOINT = "https://hf-mirror.com"
    if hasattr(huggingface_hub, "constants"):
        huggingface_hub.constants.HF_ENDPOINT = "https://hf-mirror.com"
    if hasattr(huggingface_hub, "HF_ENDPOINT"):
        huggingface_hub.HF_ENDPOINT = "https://hf-mirror.com"


def _local_dir(dataset_name: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        dataset_name,
    )


def _load_or_download(name: str, hf_path: str, max_samples=None, val_ratio=0.1):
    local = _local_dir(name)
    if os.path.exists(os.path.join(local, "dataset_dict.json")):
        ds = DatasetDict.load_from_disk(local)
    else:
        _ensure_mirror()
        ds = load_dataset(hf_path)
        os.makedirs(local, exist_ok=True)
        ds.save_to_disk(local)

    if max_samples and len(ds["train"]) > max_samples:
        ds["train"] = ds["train"].shuffle(seed=SEED).select(range(max_samples))
        val_size = max(1, int(len(ds["train"]) * val_ratio))
        if val_size < len(ds["train"]):
            split = ds["train"].train_test_split(test_size=val_size, seed=SEED)
            test_split = ds.get(
                "test", ds.get("validation", split["test"])
            )
            ds = DatasetDict(
                {
                    "train": split["train"],
                    "validation": split["test"],
                    "test": test_split,
                }
            )
    return ds


def load_arxiv_dataset(max_samples=None, val_ratio=0.1):
    """Load the arXiv long-document summarization dataset."""
    return _load_or_download(
        "arxiv", "ccdv/arxiv-summarization", max_samples, val_ratio
    )


def load_pubmed_dataset(max_samples=None, val_ratio=0.1):
    """Load the PubMed long-document summarization dataset."""
    return _load_or_download(
        "pubmed", "ccdv/pubmed-summarization", max_samples, val_ratio
    )


# ── Field names are uniform across both datasets ───────────────────────

INPUT_FIELD = "article"
TARGET_FIELD = "abstract"


# ── Statistics helpers ─────────────────────────────────────────────────


def token_length_statistics(dataset, text_field: str = INPUT_FIELD,
                             tokenizer_name: str = "facebook/bart-large"):
    """Compute token-length statistics for a dataset."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = []
    for sample in dataset.select(range(min(5000, len(dataset)))):
        text = sample.get(text_field, "")
        if text:
            lengths.append(len(tokenizer.encode(text, truncation=False)))
    lengths = np.array(lengths)
    return {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(np.max(lengths)),
        "under_512": float(np.mean(lengths <= 512)),
        "under_1024": float(np.mean(lengths <= 1024)),
        "under_2048": float(np.mean(lengths <= 2048)),
    }


DATA_LOADERS = {
    "arxiv": load_arxiv_dataset,
    "pubmed": load_pubmed_dataset,
}
