<div align="center">

# BART-FaCT

### Faithfulness-Enhanced Long-Document Summarization via<br>Hierarchical Structure Encoding & Calibrated Faithfulness Attention

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_REPO/blob/main/notebooks/run.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-FFD21E)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Abstract

Generating faithful summaries of long scientific documents is difficult for two reasons that compound each other. First, standard encoder-decoder models like BART process papers as flat token sequences. A sentence in the Abstract and a sentence in the Methods look identical to the encoder — there is no signal about where a fact sits within the document's argument. Second, at each decoding step, cross-attention distributes weight across all encoder positions uniformly. The model cannot tell whether it is retrieving a specific result from the source or guessing a plausible-sounding number. When the context is long and attention becomes diffuse, the model defaults to its language-model prior and produces fluent but unsupported statements. Maximum-likelihood training with cross-entropy loss does not distinguish between these two behaviors.

We introduce **BART-FaCT**, which augments a summarization-pretrained BART checkpoint with three lightweight modules, each targeting one of these failure modes. **HSE** (Hierarchical Structure Encoding) learns a hierarchical representation of the document — sentence, paragraph, section — and injects it back into token embeddings so the encoder knows where every token belongs in the paper's rhetorical structure. **CFA** (Calibrated Faithfulness Attention) wraps each decoder cross-attention layer with a bottleneck network that estimates per-token faithfulness uncertainty and adjusts the source-attention contribution accordingly: uncertain tokens attend harder to the source. **CPO** (Contrastive Preference Optimization) replaces heuristic negative sampling with a DPO-style preference loss where the dispreferred response is the model's own generation without encoder context — the summary it would produce if it ignored the paper entirely.

We evaluate on the arXiv and PubMed long-document summarization benchmarks, comparing against four pre-trained summarization models and isolating each module's contribution through a five-configuration ablation.

---

## Three Modules

### HSE · Hierarchical Structure Encoding

Scientific papers are organized hierarchically — claims build on methods, results support conclusions — but BART reads them as one long string. Prior attempts to fix this use regular expressions to detect section headers like "Introduction", which works poorly across disciplines and captures nothing about sentence-to-sentence discourse flow.

HSE takes a different approach. It first detects sentence boundaries using NLTK's linguistically-motivated segmenter, then mean-pools each sentence's token representations. A compact 2-layer Transformer (4 heads, 256-dim FFN, Pre-LN, GELU) processes these sentence vectors, modeling how sentences relate to each other and where they sit in the document's argument. The resulting structure-enriched representation is broadcast back to every token through a learned gate:

```
enriched = token_emb + σ(W·[token_emb ⊕ structure_context]) ⊙ structure_context
```

The encoder then sees not just "this is token 547" but "this is the third sentence of the Methods section." The module adds ~2.7M parameters — roughly 0.7% of the BART-Large backbone.

> **Inspiration.** Hierarchical Transformers (Liu & Lapata, EMNLP 2019); Lost in the Middle (Liu et al., TACL 2023).

---

### CFA · Calibrated Faithfulness Attention

In a standard decoder, cross-attention computes a weighted average of encoder states and adds it to the self-attention output. The same operation runs whether the model is looking up a fact or smoothing a transition. The model has no way to say "I am uncertain here — I should check the source more carefully."

CFA gives each decoder layer this ability. A small bottleneck network takes the cross-attention output and the self-attention state, compresses them through a 128-dimensional hidden layer, and estimates a scalar uncertainty per token. This uncertainty acts as an additive bias on a faithfulness gate:

```
uncertainty = σ(MLP([cross_attn ⊕ self_attn]))
gate = σ(W·bottleneck + uncertainty)
output = gate ⊙ cross_attn + (1−gate) ⊙ self_attn
```

When the model is uncertain (diffuse attention, large cross-self discrepancy), the gate pushes toward the source — "go look at the paper." When the model is confident, the gate allows it to generate more freely. A 0.5–0.5 residual blend with the original decoder output keeps training stable. Per-layer overhead is ~320K, totaling ~3.8M across 12 layers.

> **Inspiration.** DoLa (Chuang et al., ICLR 2024); Context-Aware Decoding (Shi et al., ACL 2024).

---

### CPO · Contrastive Preference Optimization

Cross-entropy loss teaches the model to pick likely tokens. It does not teach the model that a summary should be grounded in the source. A hallucinated number and a faithfully copied one receive the same loss if both differ from the reference phrasing.

Prior work adds contrastive objectives with heuristically perturbed negatives — swap entities, change numbers, shuffle sentences. But these synthetic perturbations may not resemble how the model actually fails. CPO constructs negatives from the model itself. The *preferred* response is the human-written reference. The *dispreferred* response is what the model generates with zero-valued encoder states — decoder-only, starting from BOS. This is the model's ungrounded prior. Any factual content in it is, by construction, hallucinated.

A DPO-style preference loss then pulls the model toward grounded generation:

```
L_cpo = −log σ(β·[log π(y_pref|x) − log π(y_disf|x)])
L_total = L_ce + λ·L_cpo   (λ=0.15, β=0.5)
```

The projection head adds ~1.2M parameters.

> **Inspiration.** DPO (Rafailov et al., NeurIPS 2023); Model-based Preference Optimization (Gao et al., EMNLP 2024).

---

## Ablation Design

| Configuration | HSE | CFA | CPO | Question |
|:---|:---:|:---:|:---:|:---|
| BART-Large-CNN | ✗ | ✗ | ✗ | Baseline: pre-trained summarization model with no enhancements |
| w/o HSE | ✗ | ✓ | ✓ | Does structure encoding help beyond faithfulness modules? |
| w/o CFA | ✓ | ✗ | ✓ | Does calibrated attention reduce hallucination further? |
| w/o CPO | ✓ | ✓ | ✗ | Does preference optimization improve factuality further? |
| **Full** | **✓** | **✓** | **✓** | Are the three contributions complementary? |

---

## Models Compared

All comparison models are pre-trained for summarization and run inference directly.

| Model | Source | Params |
|:---|:---|:---:|
| BART-Large-CNN | `facebook/bart-large-cnn` | 400M |
| PEGASUS-arXiv | `google/pegasus-arxiv` | 568M |
| PEGASUS-CNN/DM | `google/pegasus-cnn_dailymail` | 568M |
| DistilBART-CNN-12-6 | `sshleifer/distilbart-cnn-12-6` | 306M |
| **BART-FaCT** | this work | **~403M** |

---

## Quick Start

```bash
git clone <repo-url> && cd end
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Smoke test
python src/run_experiments.py --mode quick_test --dataset arxiv

# Full comparison
python src/run_experiments.py --mode exp1 --dataset arxiv \
    --models "bart-large-cnn,pegasus-arxiv,pegasus-cnn_dailymail,distilbart-cnn-12-6" \
    --max_samples 1000 --num_test 100

# Ablation
python src/run_experiments.py --mode ablation --ablation_type all
```

---

## Architecture

```
Document
    │
    ▼
┌─ HSE ──────────────────────────────────────────┐
│  sentences → 2-layer Transformer → gate ⊙ broadcast  │
└────────────────┬────────────────────────────────┘
                 ▼
┌─ BART Encoder (12 layers) ─────────────────────┐
└────────────────┬────────────────────────────────┘
                 ▼
┌─ BART Decoder (12 layers, each with CFA) ──────┐
│  Self-Attn → Cross-Attn                        │
│  CFA: uncertainty → gate modulates cross-attn   │
│  → FFN + LayerNorm                              │
└────────────────┬────────────────────────────────┘
                 │
         ┌───────┴────────┐
         ▼                ▼
      L_ce (MLE)    L_cpo (preference)
         └───────┬────────┘
                 ▼
        L_total = L_ce + λ·L_cpo
```

---

## Evaluation Metrics

**Quality:** ROUGE-1/2/L/Lsum, BERTScore F1, METEOR

**Factuality:** NLI Entailment Ratio (RoBERTa-large-MNLI), Hallucination Rate (intrinsic / extrinsic / contradiction), n-gram Overlap

**Auxiliary:** Compression Ratio, JS Divergence, 4-gram Repetition Ratio

---

## Project Structure

```
end/
├── src/
│   ├── models/
│   │   ├── bart_fact.py              # Main model & config
│   │   ├── hierarchical_structure.py # HSE module
│   │   ├── calibrated_attention.py   # CFA module
│   │   └── preference_loss.py        # CPO module
│   ├── config.py / data_utils.py
│   ├── train.py / evaluate.py / benchmark.py
│   ├── hallucination.py / ablation.py / sensitivity.py
│   ├── analyze.py / visualization.py
│   └── run_experiments.py
├── notebooks/
│   ├── run.ipynb          # Experiment runner (Colab-compatible)
│   └── preview.ipynb      # Module visualization & demo
├── data/ / results/
└── README.md / README_zh.md / EXPERIMENT_PLAN.md
```

---

## References

1. Liu & Lapata. "Hierarchical Transformers for Long Document Summarization." *EMNLP*, 2019.
2. Chuang et al. "DoLa: Decoding by Contrasting Layers Improves Factuality." *ICLR*, 2024.
3. Rafailov et al. "Direct Preference Optimization." *NeurIPS*, 2023.
4. Shi et al. "Context-Aware Decoding for Faithful Summarization." *ACL*, 2024.
5. Gao et al. "Model-based Preference Optimization in Summarization without Human Feedback." *EMNLP*, 2024.
6. Liu et al. "Lost in the Middle: How Language Models Use Long Contexts." *TACL*, 2023.
7. Lewis et al. "BART: Denoising Sequence-to-Sequence Pre-training." *ACL*, 2020.
8. Zhang et al. "PEGASUS: Pre-training with Extracted Gap-sentences." *ICML*, 2020.

---

## License

MIT. Pre-trained models under their respective HuggingFace licenses.
