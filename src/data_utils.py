import os
import re
import random
import numpy as np
import torch

# Force HuggingFace mirror for users in China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_TIMEOUT", "120")

import datasets

datasets.config.HF_ENDPOINT = os.environ["HF_ENDPOINT"]

import huggingface_hub

if hasattr(huggingface_hub, "constants"):
    huggingface_hub.constants.HF_ENDPOINT = os.environ["HF_ENDPOINT"]
if hasattr(huggingface_hub, "HF_ENDPOINT"):
    huggingface_hub.HF_ENDPOINT = os.environ["HF_ENDPOINT"]

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

from models.section_embedding import SectionDetector

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_mirror():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HUGGINGFACE_HUB_TIMEOUT"] = "120"
    datasets.config.HF_ENDPOINT = "https://hf-mirror.com"
    if hasattr(huggingface_hub, "constants"):
        huggingface_hub.constants.HF_ENDPOINT = "https://hf-mirror.com"
    if hasattr(huggingface_hub, "HF_ENDPOINT"):
        huggingface_hub.HF_ENDPOINT = "https://hf-mirror.com"


def _get_local_data_dir(dataset_name):
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        dataset_name,
    )


# ── Dataset loaders ────────────────────────────────────────────────────


def load_arxiv_dataset(max_samples=None, val_ratio=0.1):
    """Load arXiv summarization dataset with local caching."""
    local_dir = _get_local_data_dir("arxiv")
    if os.path.exists(os.path.join(local_dir, "dataset_dict.json")):
        ds = DatasetDict.load_from_disk(local_dir)
    else:
        _ensure_mirror()
        ds = load_dataset("ccdv/arxiv-summarization")
        os.makedirs(local_dir, exist_ok=True)
        ds.save_to_disk(local_dir)

    if max_samples:
        if len(ds["train"]) > max_samples:
            ds["train"] = ds["train"].shuffle(seed=SEED).select(range(max_samples))
        val_size = max(1, int(len(ds["train"]) * val_ratio))
        if val_size < len(ds["train"]):
            split = ds["train"].train_test_split(test_size=val_size, seed=SEED)
            ds = DatasetDict(
                {
                    "train": split["train"],
                    "validation": split["test"],
                    "test": ds.get(
                        "validation", ds["test"] if "test" in ds else split["test"]
                    ),
                }
            )
        else:
            ds["validation"] = ds["train"]
    return ds


def load_pubmed_dataset(max_samples=None, val_ratio=0.1):
    """Load PubMed summarization dataset with local caching."""
    local_dir = _get_local_data_dir("pubmed")
    if os.path.exists(os.path.join(local_dir, "dataset_dict.json")):
        ds = DatasetDict.load_from_disk(local_dir)
    else:
        _ensure_mirror()
        ds = load_dataset("ccdv/pubmed-summarization")
        os.makedirs(local_dir, exist_ok=True)
        ds.save_to_disk(local_dir)

    if max_samples:
        if len(ds["train"]) > max_samples:
            ds["train"] = ds["train"].shuffle(seed=SEED).select(range(max_samples))
        val_size = max(1, int(len(ds["train"]) * val_ratio))
        if val_size < len(ds["train"]):
            split = ds["train"].train_test_split(test_size=val_size, seed=SEED)
            ds = DatasetDict(
                {
                    "train": split["train"],
                    "validation": split["test"],
                    "test": ds.get(
                        "validation", ds["test"] if "test" in ds else split["test"]
                    ),
                }
            )
        else:
            ds["validation"] = ds["train"]
    return ds


# ── Preprocessing ──────────────────────────────────────────────────────


def prepare_dataset_for_model(
    dataset_name="arxiv",
    tokenizer=None,
    max_input_length=1024,
    max_target_length=256,
    max_samples=None,
    is_bart_fact=False,
):
    """Tokenize and prepare a dataset for a given model."""
    if dataset_name == "arxiv":
        ds = load_arxiv_dataset(max_samples=max_samples)
        input_field = "article"
        target_field = "abstract"
    elif dataset_name == "pubmed":
        ds = load_pubmed_dataset(max_samples=max_samples)
        input_field = "article"
        target_field = "abstract"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def _preprocess(examples):
        inputs = (
            examples[input_field]
            if isinstance(examples[input_field], list)
            else [examples[input_field]]
        )
        targets = (
            examples[target_field]
            if isinstance(examples[target_field], list)
            else [examples[target_field]]
        )

        model_inputs = tokenizer(
            inputs,
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )

        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = labels["input_ids"]

        # For BART-FaCT: compute boundary_mask (HSE) per sample
        if is_bart_fact:
            from models.hierarchical_structure import detect_sentence_boundaries
            boundary_masks = []
            for text in inputs:
                bm = detect_sentence_boundaries(text, tokenizer, max_input_length)
                boundary_masks.append(bm.squeeze(0).tolist())
            model_inputs["boundary_mask"] = boundary_masks
            model_inputs["input_texts"] = inputs

        return model_inputs

    remove_cols = ds["train"].column_names
    ds = ds.map(
        _preprocess,
        batched=True,
        remove_columns=remove_cols,
        desc=f"Tokenizing {dataset_name}",
    )
    return ds


def prepare_dataset_for_bart_fact(
    dataset_name="arxiv",
    tokenizer=None,
    max_input_length=1024,
    max_target_length=256,
    max_samples=None,
):
    """Shortcut for preparing data for BART-FaCT models."""
    return prepare_dataset_for_model(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
        max_samples=max_samples,
        is_bart_fact=True,
    )


# ── Statistics helpers ─────────────────────────────────────────────────


def token_length_statistics(
    dataset, text_field="article", tokenizer_name="facebook/bart-large"
):
    """Compute token-length statistics for a dataset."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = []
    for sample in dataset.select(range(min(5000, len(dataset)))):
        text = (
            sample[text_field]
            if text_field in sample
            else sample.get("document", "")
        )
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


def detect_sections_in_text(text: str) -> list:
    """Detect section boundaries in a scientific paper."""
    section_detector = SectionDetector()
    spans = section_detector.detect_sections(text)
    return [
        {
            "label": s.label,
            "label_id": s.label_id,
            "start": s.start_char,
            "end": s.end_char,
        }
        for s in spans
    ]


# ── Data loader registry ───────────────────────────────────────────────

DATA_LOADERS = {
    "arxiv": load_arxiv_dataset,
    "pubmed": load_pubmed_dataset,
}
