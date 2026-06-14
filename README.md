<div align="center">

# SUMM-Lens

### Zero-Training Inference-Time Enhancements for<br>Long-Document Summarization

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/YOUR_REPO/blob/main/notebooks/run.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-FFD21E)](https://huggingface.co/docs/transformers)
[![HF Models](https://img.shields.io/badge/🤗%20Models-Qwen2.5%20%7C%20BART%20%7C%20PEGASUS%20%7C%20LED-blue)](https://huggingface.co/models?pipeline_tag=summarization)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Abstract

Long-document summarization on arXiv and PubMed faces three intertwined difficulties. **(P1) Training cost.** Mainstream encoder-decoder baselines (BART, PEGASUS, LED) require per-dataset fine-tuning at the 300–600M parameter scale, which is impractical on free-tier compute. **(P2) Sparsity of single-pass generation.** Zero-shot summaries from instruction-tuned LLMs are fluent but *sparse* — they tend to miss key entities, numbers, and cross-section findings, because a single forward pass commits to one trajectory through the document. **(P3) Unreliability of single-sample decoding.** A fluent summary may quietly diverge from the source; standard greedy or beam decoding has no internal mechanism to prefer the most source-faithful candidate among several plausible ones.

We propose **SUMM-Lens**, a framework that targets all three problems **without training a single parameter**. With Qwen2.5-1.5B-Instruct (Apache 2.0, 2024) as a fixed modern backbone, we add two lightweight inference-time modules:

- **CoD — Chain-of-Density prompting (≈ 80 lines)** addresses **P2**: it iteratively rewrites the summary to incorporate previously-missed entities while preserving length, turning a sparse first draft into a denser one without any gradient update.
- **NLR — NLI-Rerank (≈ 120 lines)** addresses **P3**: it samples K diverse candidates and selects the one whose every sentence is most entailed by the source under a pretrained MNLI model, replacing a single point estimate with a faithfulness-aware selection step.

The zero-training property itself addresses **P1**: every component is a plug-in, dataset-agnostic, and runnable on free Colab or CPU. We benchmark a 2019→2024 baseline ladder (BART-Large-CNN / DistilBART / PEGASUS-arXiv / LED-arXiv / Qwen2.5-Vanilla) on arXiv and PubMed, and run a four-configuration ablation (Vanilla, +CoD, +NLR, +Both) to isolate each module's contribution. The result is a fully reproducible faithful-summarization pipeline with no training cost — useful both as a strong zero-shot baseline for compute-constrained research and as a drop-in faithfulness booster for any causal-LM summarizer.

---

## Two Modules

### CoD · Chain-of-Density Prompting

Vanilla summarization with a small LLM tends to produce a sparse first draft — a few high-level sentences that miss entities, numbers, and cross-section findings. CoD treats the summary as something to *densify*, not just to *generate*. Starting from a single seed summary, the model is asked to rewrite it 3 times; each rewrite must (a) identify 1–3 informative entities or numbers from the article that are *missing* from the current summary, and (b) integrate them while keeping the overall length essentially constant. The rewrite-budget forces compression of less informative phrasing as denser content is added.

The module is a 3-pass loop over a fixed prompt template; no model weights are updated. On causal-LM backbones it runs end-to-end in chat format. On seq2seq backbones (BART/PEGASUS) it gracefully degrades to a single-pass generation.

> **Inspiration.** Adams et al. *From Sparse to Dense: GPT-4 Summarization with the Chain of Density Prompt.* EMNLP 2023.

### NLR · NLI-Rerank

A single-sample summary is a single point estimate. NLR turns summarization into a generate-then-select pipeline: 4 candidates are sampled at temperature 0.7 / top-p 0.95, then each candidate is split into sentences and scored against the source via a pretrained MNLI model (`roberta-large-mnli`). Each sentence becomes a hypothesis whose premise is a head-tail truncation of the article; we average per-sentence entailment probability to score the candidate, then return the maximum-scoring summary.

NLR is purely zero-shot and reuses the same NLI checkpoint already loaded by the hallucination evaluation module — no additional weights, no fine-tuning, ~120 lines.

> **Inspiration.** Laban et al. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL 2022. Plus 2024-era generate-then-rerank work for faithfulness.

When CoD and NLR compose, NLR samples K seeds, runs CoD on each, and then NLI-reranks the densified set.

---

## Baseline Ladder (2019 → 2024)

All entries are **inference-only**. Every model loads via `AutoModel*.from_pretrained`.

| Year | Model | HF Path | Params | Notes |
|:---:|:---|:---|:---:|:---|
| 2019 | BART-Large-CNN | `facebook/bart-large-cnn` | 400M | Seq2seq baseline |
| 2020 | DistilBART-CNN | `sshleifer/distilbart-cnn-12-6` | 306M | Distilled, fast |
| 2020 | PEGASUS-arXiv | `google/pegasus-arxiv` | 568M | arXiv-specific |
| 2020 | LED-arXiv | `allenai/led-large-16384-arxiv` | 460M | Long-document encoder |
| **2024** | **Qwen2.5-1.5B-Instruct** | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Modern zero-shot baseline |
| **Ours** | Qwen2.5 + CoD + NLR | — | 1.5B | Inference-time enhanced |

---

## Ablation Design

The four ablation configurations all share the same 2024 backbone (`Qwen/Qwen2.5-1.5B-Instruct`); only the inference-time pipeline changes.

| Configuration | CoD | NLR | Question |
|:---|:---:|:---:|:---|
| Qwen2.5-Vanilla | ✗ | ✗ | What does a 2024 LLM achieve zero-shot? |
| + CoD | ✓ | ✗ | Does iterative densification help alone? |
| + NLR | ✗ | ✓ | Does NLI-based candidate selection help alone? |
| **+ CoD + NLR** | **✓** | **✓** | Are the two modules complementary? |

---

## Quick Start

```bash
git clone <repo-url> && cd end
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Smoke test (5 samples, 3 models, no BERTScore/METEOR — runs on CPU in minutes)
python src/run_experiments.py --mode quick_test --dataset arxiv

# Full baseline ladder
python src/run_experiments.py --mode baseline --dataset arxiv --num_test 100

# Module ablation only
python src/run_experiments.py --mode ablation --dataset arxiv --num_test 100

# Everything + figures
python src/run_experiments.py --mode all --dataset arxiv --num_test 100
```

The `notebooks/run.ipynb` notebook is Colab-compatible and walks through dataset preview, baseline runs, ablation, and figure generation.

---

## Architecture

```
                    ┌──────────────────────────────┐
   long article ──► │  Qwen2.5-1.5B-Instruct       │ ── seed summary ──┐
                    │  (zero-shot, chat template)  │                   │
                    └──────────────────────────────┘                   │
                                                                       ▼
                                       ┌──────── Chain-of-Density ────┐
                                       │  iter 1: add 1-3 missed       │
                                       │           entities, keep len  │
                                       │  iter 2: ditto                │
                                       │  iter 3: ditto                │
                                       └─────────────┬─────────────────┘
                                                     │ K densified candidates
                                                     ▼
                                       ┌──────── NLI-Rerank ──────────┐
                                       │  for each candidate:          │
                                       │    split → sentences          │
                                       │    NLI(article, sentence)     │
                                       │    score = mean P(entail)     │
                                       │  return argmax                │
                                       └─────────────┬─────────────────┘
                                                     │
                                                     ▼
                                                final summary
```

---

## Evaluation Metrics

**Quality:** ROUGE-1/2/L/Lsum, BERTScore F1, METEOR

**Faithfulness:** NLI Entailment Ratio (RoBERTa-large-MNLI), per-candidate entailment score

**Auxiliary:** JS Divergence (n-gram), 4-gram repetition ratio, compression ratio, novelty ratio

All metrics are computed by `src/benchmark.py`; faithfulness via `src/hallucination.py` (shared with NLR).

---

## Project Structure

```
end/
├── src/
│   ├── methods/                # ★ inference-time modules (zero training)
│   │   ├── llm_summarizer.py   #   unified seq2seq / causal-LM wrapper
│   │   ├── cod.py              #   Chain-of-Density (~80 lines)
│   │   ├── nli_rerank.py       #   NLI rerank (~120 lines)
│   │   └── prompts.py          #   per-dataset templates
│   ├── config.py               # model registry + runtime config
│   ├── data_utils.py           # arXiv / PubMed loaders, HF mirror
│   ├── evaluate.py             # per-model inference + metrics
│   ├── benchmark.py            # ROUGE / BERTScore / METEOR / JS / ...
│   ├── hallucination.py        # NLI-based hallucination detector (shared)
│   ├── analyze.py              # figures + LaTeX tables (CJK-aware fonts)
│   ├── visualization.py        # notebook helpers
│   └── run_experiments.py      # top-level CLI
├── notebooks/
│   └── run.ipynb               # Colab / VSCode walkthrough
├── data/
│   ├── arxiv/                  # cached HF dataset
│   └── pubmed/
├── results/                    # eval outputs (per-model JSON + figures)
├── requirements.txt
├── README.md / README_zh.md / 论文.md
└── EXPERIMENT_PLAN.md
```

---

## References

1. Adams G, et al. *From Sparse to Dense: GPT-4 Summarization with the Chain of Density Prompt.* EMNLP 2023.
2. Laban P, et al. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL 2022.
3. Qwen Team. *Qwen2.5 Technical Report.* 2024.
4. Lewis M, et al. *BART: Denoising Sequence-to-Sequence Pre-training.* ACL 2020.
5. Zhang J, et al. *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.* ICML 2020.
6. Beltagy I, et al. *Longformer: The Long-Document Transformer.* arXiv:2004.05150, 2020.

---

## License

MIT for SUMM-Lens code. Pre-trained models follow their respective HuggingFace licenses (Apache 2.0 for Qwen2.5; permissive research licenses for BART/PEGASUS/LED).
