"""NLI-Rerank — zero-training inference-time faithfulness selector.

Pipeline:
    1. Generate K diverse candidate summaries (different temperature / top_p).
    2. For each candidate, split into sentences and run NLI (premise = source,
       hypothesis = sentence) using a pretrained MNLI checkpoint.
    3. Score each candidate by the mean entailment probability over its sentences.
    4. Return the highest-scoring candidate.

This wraps the existing `hallucination.HallucinationDetector` so we do not load
two NLI models. Inspired by SummaC (Laban et al., TACL 2022) and faithfulness-
aware reranking work (Chen et al., ACL 2024).
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

from hallucination import HallucinationDetector

logger = logging.getLogger(__name__)

# Conservative source-chunk size (chars) for NLI — RoBERTa max is 512 tokens.
_PREMISE_MAX_CHARS = 1500


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter (avoids requiring NLTK at import time).

    Falls back to NLTK punkt if available; otherwise uses a regex on
    .!? boundaries. Both produce reasonable chunks for NLI scoring.
    """
    text = text.strip()
    if not text:
        return []
    try:
        import nltk  # type: ignore

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        # Fallback regex.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z一-鿿])", text)
        return [p.strip() for p in parts if p.strip()]


def _truncate_premise(source: str, max_chars: int = _PREMISE_MAX_CHARS) -> str:
    """Keep head+tail of the source so the NLI premise fits in 512 tokens."""
    s = source.strip()
    if len(s) <= max_chars:
        return s
    head = s[: max_chars * 3 // 4]
    tail = s[-max_chars // 4 :]
    return head + " ... " + tail


class NLIReranker:
    """Score and rerank multiple candidate summaries by NLI entailment."""

    def __init__(
        self,
        nli_model_name: str = "roberta-large-mnli",
        device=None,
        detector: Optional[HallucinationDetector] = None,
    ):
        # Reuse a passed-in detector to share weights across runs.
        self.detector = detector or HallucinationDetector(
            nli_model_name=nli_model_name, device=device
        )

    def score(self, source: str, candidate: str) -> float:
        """Mean per-sentence entailment probability for a single candidate."""
        sentences = _split_sentences(candidate)
        if not sentences:
            return 0.0
        premise = _truncate_premise(source)
        premises = [premise] * len(sentences)
        probs = self.detector.check_entailment_batch(
            premises, sentences, batch_size=8
        )
        ent = [p.get("entailment", 0.0) for p in probs]
        return float(np.mean(ent)) if ent else 0.0

    def rerank(
        self, source: str, candidates: List[str]
    ) -> Tuple[str, List[float]]:
        """Return ``(best_candidate, scores)``. ``scores[i]`` aligns with ``candidates[i]``."""
        if not candidates:
            return "", []
        scores = [self.score(source, c) for c in candidates]
        best_idx = int(np.argmax(scores))
        return candidates[best_idx], scores


# ── Convenience wrapper used by the experiment driver ──────────────────


def nli_rerank(
    summarizer,
    article: str,
    dataset: str = "arxiv",
    n_candidates: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.95,
    reranker: Optional[NLIReranker] = None,
    use_cod: bool = False,
    cod_iters: int = 3,
    target_words: int = 200,
) -> Tuple[str, dict]:
    """Generate `n_candidates` diverse summaries and pick the most faithful one.

    When ``use_cod=True`` each candidate is produced via Chain-of-Density on the
    underlying causal LM (``methods.cod.chain_of_density``); otherwise candidates
    are direct samples from the model.

    Returns ``(best_summary, info)`` where ``info`` includes per-candidate scores.
    """
    from methods.prompts import get_prompt
    from methods.cod import chain_of_density

    candidates: List[str] = []

    if use_cod and summarizer.is_causal_lm:
        # CoD on each sampled seed -> denser & diverse.
        seed_prompt = get_prompt(dataset, article)
        seeds = summarizer.summarize(
            seed_prompt, n=n_candidates, temperature=temperature, top_p=top_p
        )
        for seed in seeds:
            dense = chain_of_density(
                summarizer,
                article,
                dataset=dataset,
                num_iters=cod_iters,
                target_words=target_words,
                initial_summary=seed,
            )
            candidates.append(dense)
    elif summarizer.is_causal_lm:
        prompt = get_prompt(dataset, article)
        candidates = summarizer.summarize(
            prompt, n=n_candidates, temperature=temperature, top_p=top_p
        )
    else:
        # Seq2seq: just sample N times directly on the article.
        candidates = summarizer.summarize(
            article, n=n_candidates, temperature=temperature, top_p=top_p
        )

    reranker = reranker or NLIReranker()
    best, scores = reranker.rerank(article, candidates)
    info = {
        "n_candidates": len(candidates),
        "scores": scores,
        "best_idx": int(np.argmax(scores)) if scores else -1,
        "candidates": candidates,
    }
    return best, info
