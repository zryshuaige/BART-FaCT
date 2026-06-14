"""SUMM-Lens: zero-training inference-time enhancements for long-document summarization.

Public API:
    Seq2SeqSummarizer  — wraps BART/PEGASUS/LED/DistilBART for inference
    CausalLMSummarizer — wraps Qwen2.5-Instruct (and similar chat LLMs)
    chain_of_density   — CoD prompting (operates on a CausalLMSummarizer)
    nli_rerank         — multi-candidate NLI rerank (operates on either summarizer)
"""

from methods.llm_summarizer import (
    Summarizer,
    Seq2SeqSummarizer,
    CausalLMSummarizer,
    build_summarizer,
)
from methods.cod import chain_of_density
from methods.nli_rerank import NLIReranker, nli_rerank
from methods.prompts import get_prompt

__all__ = [
    "Summarizer",
    "Seq2SeqSummarizer",
    "CausalLMSummarizer",
    "build_summarizer",
    "chain_of_density",
    "NLIReranker",
    "nli_rerank",
    "get_prompt",
]
