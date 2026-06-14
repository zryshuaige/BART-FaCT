"""Prompt templates for zero-shot summarization with instruction-tuned LLMs.

Each dataset has its own template because the conventions differ:
- arXiv / PubMed: long scientific articles, abstract-style summary, ~200 words
- CNN/DailyMail: news article, bullet-style summary, ~3 sentences
"""

# Initial summarization prompts (single-pass, used by Vanilla and as CoD seed).
_INITIAL = {
    "arxiv": (
        "You are an expert scientific writer. Read the following research article and "
        "write a faithful, self-contained abstract of about 200 words that covers the "
        "motivation, method, and main findings. Use only information present in the "
        "article — do not invent numbers or claims.\n\n"
        "Article:\n{article}\n\n"
        "Abstract:"
    ),
    "pubmed": (
        "You are an expert biomedical writer. Read the following biomedical article and "
        "write a faithful structured abstract of about 200 words that covers the "
        "objective, methods, results, and conclusion. Use only information present in "
        "the article.\n\n"
        "Article:\n{article}\n\n"
        "Abstract:"
    ),
    "cnn_dailymail": (
        "Write a concise summary of the following news article in about 3 sentences. "
        "Include only facts that appear in the article.\n\n"
        "Article:\n{article}\n\n"
        "Summary:"
    ),
}

# Chain-of-Density iteration prompt — applied repeatedly with the previous
# summary in `prev_summary`. Adapted from Adams et al., EMNLP 2023.
_COD_STEP = (
    "You will rewrite a summary to make it denser without making it longer.\n\n"
    "Article:\n{article}\n\n"
    "Current summary (about {target_words} words):\n{prev_summary}\n\n"
    "Step 1. Identify 1-3 informative entities, findings, or numbers from the article "
    "that are MISSING from the current summary.\n"
    "Step 2. Rewrite the summary to incorporate these missing items while keeping the "
    "overall length essentially unchanged (still about {target_words} words). Compress "
    "less informative phrasing to make room.\n\n"
    "Constraints: every fact in the new summary must be supported by the article. Do "
    "not introduce any information not present in the article. Output only the rewritten "
    "summary, with no preamble.\n\n"
    "Denser summary:"
)


def get_prompt(dataset: str, article: str, max_chars: int = 12000) -> str:
    """Return the initial zero-shot summarization prompt for `dataset`.

    `max_chars` truncates the article to keep the LLM's input reasonable on CPU.
    """
    key = dataset.lower().replace("-", "_")
    if key not in _INITIAL:
        key = "arxiv"  # default for unknown datasets
    article = (article or "").strip()
    if len(article) > max_chars:
        # head + tail truncation: keep beginning and end, drop middle
        head = article[: max_chars * 3 // 4]
        tail = article[-max_chars // 4 :]
        article = head + "\n[... truncated ...]\n" + tail
    return _INITIAL[key].format(article=article)


def get_cod_prompt(
    article: str,
    prev_summary: str,
    target_words: int = 200,
    max_chars: int = 12000,
) -> str:
    """CoD iteration prompt: rewrite `prev_summary` to be denser, same length."""
    article = (article or "").strip()
    if len(article) > max_chars:
        head = article[: max_chars * 3 // 4]
        tail = article[-max_chars // 4 :]
        article = head + "\n[... truncated ...]\n" + tail
    return _COD_STEP.format(
        article=article, prev_summary=prev_summary.strip(), target_words=target_words
    )
