"""Unified summarizer interface for both seq2seq and causal-LM backbones.

The two model families are wrapped behind a common ``.summarize(text, n=k, ...)``
method that always returns a list of strings — single-candidate (n=1) or
multi-candidate (n>1, used by NLI-Rerank).

Backend selection is automatic from the model config's ``is_causal_lm`` flag.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def _pick_dtype(device: torch.device) -> torch.dtype:
    """Pick a sensible default dtype for the device."""
    if device.type == "cuda":
        # bf16 if Ampere+, else fp16
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32  # CPU and MPS run fp32


# ── Abstract base ──────────────────────────────────────────────────────


class Summarizer(ABC):
    """Common interface for any summarization backbone."""

    name: str
    is_causal_lm: bool = False
    max_input_length: int = 1024
    max_target_length: int = 256

    @abstractmethod
    def summarize(
        self,
        text: str,
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **gen_kwargs,
    ) -> List[str]:
        """Generate `n` candidate summaries for `text`."""

    def summarize_batch(
        self,
        texts: List[str],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **gen_kwargs,
    ) -> List[List[str]]:
        """Default: loop. Subclasses may override with true batching."""
        return [
            self.summarize(t, n=n, temperature=temperature, top_p=top_p, **gen_kwargs)
            for t in texts
        ]


# ── Seq2seq backbone (BART / PEGASUS / LED / DistilBART) ───────────────


class Seq2SeqSummarizer(Summarizer):
    """Wraps `AutoModelForSeq2SeqLM` for summarization."""

    is_causal_lm = False

    def __init__(
        self,
        hf_path: str,
        max_input_length: int = 1024,
        max_target_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.name = hf_path
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dtype = dtype or _pick_dtype(self.device)

        logger.info(f"Loading seq2seq model {hf_path} (dtype={dtype}, device={self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path)
        # transformers >= 4.55 renamed `torch_dtype` to `dtype`; fall back for older.
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_path, dtype=dtype)
        except TypeError:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_path, torch_dtype=dtype)
        self.model = self.model.to(self.device).eval()

    @torch.no_grad()
    def summarize(
        self,
        text: str,
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        beam_size: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        **_,
    ) -> List[str]:
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        if n == 1 and temperature == 0.0:
            # Deterministic beam search.
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_target_length,
                num_beams=beam_size,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )
        else:
            # Multi-candidate sampling.
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_target_length,
                do_sample=True,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                num_return_sequences=n,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


# ── Causal LM backbone (Qwen2.5-Instruct, etc.) ────────────────────────


class CausalLMSummarizer(Summarizer):
    """Wraps an instruction-tuned causal LM (Qwen2.5, Llama-3, etc.)."""

    is_causal_lm = True

    def __init__(
        self,
        hf_path: str,
        max_input_tokens: int = 8192,
        max_new_tokens: int = 384,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = hf_path
        self.max_input_length = max_input_tokens
        self.max_target_length = max_new_tokens
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dtype = dtype or _pick_dtype(self.device)

        logger.info(f"Loading causal LM {hf_path} (dtype={dtype}, device={self.device})")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=False)
        # Some chat tokenizers default to no pad token — use eos for left-padding generation.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_path, dtype=dtype, trust_remote_code=False
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_path, torch_dtype=dtype, trust_remote_code=False
            )
        self.model = self.model.to(self.device).eval()

    def _build_chat_input(self, prompt: str) -> torch.Tensor:
        """Apply chat template; return token ids on device."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        ).input_ids.to(self.device)
        return ids

    @torch.no_grad()
    def summarize(
        self,
        text: str,
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_new_tokens: Optional[int] = None,
        **_,
    ) -> List[str]:
        """Generate `n` candidate summaries.

        ``text`` is treated as a fully-formed prompt (use ``methods.prompts.get_prompt``
        to wrap raw articles). When ``temperature == 0`` and ``n == 1`` we use greedy
        decoding for reproducibility.
        """
        input_ids = self._build_chat_input(text)
        max_new = max_new_tokens or self.max_target_length

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if n == 1 and temperature == 0.0:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(
                do_sample=True,
                temperature=max(temperature, 1e-3),
                top_p=top_p,
                num_return_sequences=n,
            )

        outputs = self.model.generate(**gen_kwargs)
        # Strip the prompt: each row's prompt prefix is identical when num_return_sequences>1.
        prompt_len = input_ids.shape[1]
        gen_only = outputs[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        return [d.strip() for d in decoded]


# ── Factory ────────────────────────────────────────────────────────────


def build_summarizer(model_config, device=None) -> Summarizer:
    """Build the right summarizer subclass from a `ModelConfig`."""
    if getattr(model_config, "is_causal_lm", False):
        return CausalLMSummarizer(
            hf_path=model_config.hf_path,
            max_input_tokens=model_config.max_input_length,
            max_new_tokens=model_config.max_target_length,
            device=device,
        )
    return Seq2SeqSummarizer(
        hf_path=model_config.hf_path,
        max_input_length=model_config.max_input_length,
        max_target_length=model_config.max_target_length,
        device=device,
    )
