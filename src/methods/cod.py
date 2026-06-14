"""Chain-of-Density (CoD) prompting — zero-training, inference-time only.

Adams et al., "From Sparse to Dense: GPT-4 Summarization with the Chain of Density
Prompt", EMNLP 2023.

Idea: starting from an initial summary, iteratively rewrite it to incorporate 1-3
previously-missed entities while keeping the length essentially constant. Each
rewrite increases information density without bloating the output.

Usage:
    s = CausalLMSummarizer("Qwen/Qwen2.5-1.5B-Instruct")
    summary = chain_of_density(s, article, dataset="arxiv", num_iters=3)
"""

from __future__ import annotations

import logging
from typing import Optional

from methods.llm_summarizer import CausalLMSummarizer, Summarizer
from methods.prompts import get_prompt, get_cod_prompt

logger = logging.getLogger(__name__)


def chain_of_density(
    summarizer: Summarizer,
    article: str,
    dataset: str = "arxiv",
    num_iters: int = 3,
    target_words: int = 200,
    initial_summary: Optional[str] = None,
    max_chars: int = 12000,
) -> str:
    """Run CoD on `article`, returning the densest (final) summary.

    Falls back to a single-pass generation when the underlying model is not a
    causal LM (e.g. BART) — chat-style iterative rewrites do not apply there.

    Parameters
    ----------
    summarizer : Summarizer
        Backbone. Must be a `CausalLMSummarizer` for true CoD.
    article : str
        Source document.
    dataset : str
        Used to pick the initial prompt template.
    num_iters : int
        Number of densification rounds. 0 = vanilla single-pass.
    target_words : int
        Approximate word count to maintain across rewrites.
    initial_summary : str, optional
        Skip the seed pass by providing an existing summary.
    max_chars : int
        Article truncation cap (head + tail) to fit a small LLM's context.
    """
    # Seed: either reuse a given summary or generate one.
    if initial_summary is not None:
        summary = initial_summary
    else:
        seed_prompt = get_prompt(dataset, article, max_chars=max_chars)
        summary = summarizer.summarize(seed_prompt, n=1, temperature=0.0)[0]

    if not summarizer.is_causal_lm:
        if num_iters > 0:
            logger.warning(
                "Chain-of-Density requires a causal LM; got seq2seq backbone "
                f"({summarizer.name}). Falling back to single-pass output."
            )
        return summary.strip()

    # Iterative densification.
    for it in range(num_iters):
        prompt = get_cod_prompt(
            article=article,
            prev_summary=summary,
            target_words=target_words,
            max_chars=max_chars,
        )
        new_summary = summarizer.summarize(prompt, n=1, temperature=0.0)[0].strip()
        # Guard against empty/degenerate rewrites: keep the previous if the new
        # one collapsed (occasional small-LLM failure mode).
        if len(new_summary.split()) < max(20, target_words // 4):
            logger.debug(
                f"CoD iter {it + 1}: rewrite too short ({len(new_summary.split())} words), "
                "keeping previous summary."
            )
            continue
        summary = new_summary

    return summary.strip()
