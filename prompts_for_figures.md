# SUMM-Lens Figure Prompts for AI Image Generation

## Style Guide

All figures should follow a consistent academic paper illustration style:
- Clean vector-like lines, no photorealism
- White background, blue/navy/teal color palette with occasional orange accent for highlights
- Sans-serif font labels (similar to LaTeX default or Arial)
- Minimal shading, flat design with slight gradients where depth is needed
- Arrow style: solid, with clean arrowheads (no decorative elements)
- Consistent rounded-rectangle boxes for modules/processing steps
- No 3D perspective, no drop shadows, no glass effect
- Aspect ratio: 16:9 for workflow diagrams, 4:3 for comparison visuals

---

## Figure 1: SUMM-Lens Overall Architecture

**Purpose:** Show the complete inference pipeline from source document to final summary output, emphasizing the modular and training-free nature of the framework.

**Main Prompt:**

> A clean academic architecture diagram for a natural language processing framework called SUMM-Lens. The diagram flows left-to-right in a pipeline layout on a white background.
>
> Left side: A tall document icon labeled "Source Document x" with decorative text lines, colored in light navy blue (#2C3E6B).
>
> Center-left: A large rounded rectangle box representing "Qwen2.5-1.5B-Instruct (frozen backbone)" with a "No gradient update" lock icon badge, colored in solid navy blue. From this box, four parallel arrows emerge downward, each labeled "s_k ~ f_theta(x; T)" with temperature T=0.7, representing seed summary sampling with diverse candidates.
>
> Center: A mid-level processing block labeled "CoD-Lite Module" in teal (#1A8A7D), showing three small iterative steps inside marked "Round 1", "Round 2", "Round 3" with a small circular arrow indicating iterative densification. Inside each round, a tiny label "Identify missing entities → Integrate into summary" appears. A dashed exit arrow labeled "Length collapse guard" branches off to a fallback path returning to the previous round's output.
>
> Center-right: A selection block labeled "NLR Module" in burnt orange (#D4762C). Inside, four candidate summaries arrive from left. Each candidate is split into sentences (shown as small horizontal bars), and each bar connects to a small "roberta-large-mnli" scoring node that outputs an entailment probability. The scores are averaged per candidate, and the highest-scoring candidate is highlighted with a checkmark.
>
> Right side: A summary output icon labeled "Optimized Summary y-hat" in dark teal.
>
> At the top, a horizontal legend bar shows four configurations: Vanilla (gray), +CoD-Lite (teal), +NLR (orange), +CoD-Lite+NLR (teal+orange gradient). At the bottom, a note reads "Training-free: theta and phi are never updated."
>
> Style: flat vector illustration, clean lines, no 3D effects, academic conference paper quality, white background.

**Negative Prompt:**

> photorealistic, 3D render, shadows, gradients on boxes, comic style, hand-drawn, sketched, watermark, text overflow, cluttered layout, dark background, neon colors, decorative borders.

---

## Figure 2: CoD-Lite Module Structure

**Purpose:** Illustrate the 3-round iterative densification process with the length collapse guard mechanism in detail.

**Main Prompt:**

> A detailed flow diagram of the CoD-Lite (Chain-of-Density Lite) module for text summarization, in academic paper illustration style on a white background.
>
> The diagram is arranged as a horizontal iterative loop with three rounds, flowing left to right, then curving back for the next iteration.
>
> Starting point (far left): A rounded rectangle in navy blue labeled "Seed Summary s_0" with a subtitle "f_theta(x)" indicating it comes from the frozen LLM backbone.
>
> Round 1 block: A teal (#1A8A7D) rounded rectangle containing two sub-steps arranged vertically:
> - Step A (top): "Identify 1-3 missing entities from x not covered by s_0" shown as a magnifying glass icon over the source document
> - Step B (bottom): "Integrate identified entities into s_0 while maintaining |s| ≈ 200 words" shown as a merging arrow
> Output arrow labeled "s_1" goes right.
>
> Round 2 block: Identical structure to Round 1 but with "s_1" as input and "s_2" as output, showing iterative densification. The entities identified become fewer (only 1-2), indicated by a diminishing arrow thickness.
>
> Round 3 block: Final iteration, producing "s_3" as the final densified summary output.
>
> Below the three round blocks, a dashed orange path represents the "Length Collapse Guard": a diamond decision node labeled "|s_t| < W/4?" with two exits: "No → Continue" (green, going right) and "Yes → Rollback: s_t = s_{t-1}" (red, looping back to previous round's output).
>
> A small inset box in the bottom-right corner shows a before/after example: "Before: 'The model performs well on benchmarks...' | After: 'The Qwen2.5-1.5B model achieves 0.41 ROUGE-1 on PubMed, surpassing fine-tuned LED...'" to illustrate entity infilling.
>
> Top-right corner: A note box reading "N = 3 iterations, W = 200 target words, No gradient updates required".
>
> Style: flat vector, clean geometric shapes, academic ACL/EMNLP paper quality, consistent teal and navy color scheme, white background.

**Negative Prompt:**

> photorealistic, 3D render, heavy shadows, hand-drawn sketch, cartoon style, dark background, neon colors, watermark, clutter, excessive decoration, gradient fills on boxes.

---

## Figure 3: NLR Module Structure

**Purpose:** Illustrate the NLI-based reranking process: sampling multiple candidates, splitting into sentences, entailment scoring, and selecting the best candidate.

**Main Prompt:**

> A detailed flow diagram of the NLR (NLI-Rerank) module for summary candidate selection, in academic paper illustration style on a white background.
>
> The diagram flows top-to-bottom in four vertically stacked stages.
>
> Stage 1 - Candidate Sampling (top): A large "f_theta (Qwen2.5-1.5B)" box in navy blue at the top, with four arrows diverging downward, each labeled with "T=0.7, top_p=0.95" and numbered "c_1, c_2, c_3, c_4". Each arrow leads to a small document icon representing a candidate summary. The four candidates are shown as short text blocks with slightly different content (some bolder, some shorter) to indicate diversity.
>
> Stage 2 - Source Document Truncation: On the left side, the full source document "x" is shown as a tall blue rectangle, with a highlighted "Head 75% + Tail 25%" region shown in lighter blue, producing a truncated version "x'" in a compact box. A dashed line connects x' to Stage 3.
>
> Stage 3 - Sentence-Level NLI Scoring (center, largest block): For each candidate c_k, the text is split into sentence bars "u_{k,1}, u_{k,2}, ..., u_{k,M_k}" shown as horizontal stripes. Each sentence bar connects to a small scoring node labeled "RoBERTa-large-MNLI" in burnt orange (#D4762C). The node takes two inputs: the truncated source x' (as premise) and the sentence u_{k,j} (as hypothesis), and outputs P(entail | x', u_{k,j}) as a probability value. Four parallel scoring tracks are shown, one per candidate.
>
> Stage 4 - Candidate Selection (bottom): A score aggregation box computes "score(c_k) = mean of sentence-level entailment probabilities" for each candidate. The four scores are shown as horizontal bar charts side by side. The highest bar (c_2 or similar) is highlighted in green with a star, and an arrow points down to the final output box "Selected Summary: c-hat = argmax score(c_k)".
>
> Bottom-right corner note: "K=4 candidates, shared roberta-large-mnli weights, zero additional storage".
>
> Style: flat vector illustration, clean lines, consistent navy/teal/orange palette, academic paper quality, white background, no photorealism.

**Negative Prompt:**

> photorealistic, 3D, shadows, gradient fills, cartoon, hand-drawn, dark background, neon, cluttered, watermark, excessive text, decorative borders.

---

## Figure 4: Generation Effect Visualization Comparison

**Purpose:** Visually compare the outputs of four configurations (Vanilla, +CoD-Lite, +NLR, +CoD-Lite+NLR) on the same input document, highlighting entity coverage differences and structural/faithfulness differences.

**Main Prompt:**

> A side-by-side comparison visualization of four summarization configurations on the same scientific paper, in academic poster illustration style on a clean white background.
>
> The layout is a 2x2 grid, each cell representing one configuration. Above the grid, a header reads "PubMed Sample idx=4: Pentoxifylline treatment for Type 1 Diabetes" as the shared input document context.
>
> Top-left cell - "Qwen2.5 Vanilla": Shows a summary text block where key sentences are highlighted in light gray. Entity names (drug names, dosages, cytokine markers like IFN-gamma, IL-17) are shown in plain text without special marking. The text flows as continuous paragraphs without section headers. A small bar chart in the corner shows: ROUGE-1=0.484, Nov2g=0.77.
>
> Top-right cell - "Qwen2.5 + CoD-Lite": Shows a summary text block where newly added entities from iterative densification are highlighted in teal (#1A8A7D) bold text. More specific numbers and entity names appear compared to Vanilla (e.g., "100 mg/kg/day", "C57BL/6 mice", "21 days"). The text is denser but still in paragraph form. Bar chart: ROUGE-1=0.472 (slightly lower on PubMed), but entity count is visibly higher.
>
> Bottom-left cell - "Qwen2.5 + NLR": Shows a summary text block with clear IMRaD section headers highlighted in burnt orange (#D4762C): "**Objective:**", "**Methods:**", "**Results:**". Each section contains specific numbers and entities from the source. The structure is visibly more organized than Vanilla. The NLI entailment scores for each sentence are shown as small green bars on the left margin of each sentence. Bar chart: ROUGE-1=0.520, Nov2g=0.80, NLI entailment=HIGH.
>
> Bottom-right cell - "Qwen2.5 + CoD-Lite + NLR": Shows a summary that attempts both densification and structural selection, but the text appears slightly redundant — entities from CoD compete with the clean structural preference of NLR. Highlighted in both teal and orange overlaps in a few places. Bar chart: ROUGE-1=0.44 (lower than either alone), indicating the diminishing return of stacking modules.
>
> At the bottom, a horizontal legend explains: teal highlighting = entities added by CoD-Lite, orange highlighting = IMRaD structure from NLR selection, green bar margin = NLI entailment score.
>
> A callout annotation between top-right and bottom-left cells reads "Module Independence: CoD-Lite and NLR optimize orthogonal objectives — use separately for best results."
>
> Style: clean data visualization, flat design, academic poster quality, consistent color coding, white background, sans-serif labels, no photorealism.

**Negative Prompt:**

> photorealistic, 3D render, hand-drawn, cartoon style, dark background, neon colors, watermark, messy layout, excessive gradients, decorative elements, illustration-style people or animals.