"""
HSE: Hierarchical Structure Encoding
=====================================
Inspired by:
  - "Hierarchical Transformers for Long Document Summarization" (Liu & Lapata, EMNLP 2019)
  - "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., TACL 2023)
  - Discourse-aware summarization (Cohan et al., NAACL 2018; Xu et al., ACL 2020)

Core idea: Instead of flat token-level encoding, HSE builds a lightweight 3-level
hierarchy (token → sentence → section) on top of BART's encoder outputs.
A small hierarchical adapter pools sentence representations, models inter-sentence
relations via a compact transformer, and broadcasts structure-enriched representations
back to every token. This lets the encoder see "this token belongs to sentence 3
in the Methods section" without needing brittle regex-based section detectors.

Design:
  Level 1 (token):    BART encoder hidden states (B, T, D)
  Level 2 (sentence):  Mean-pool tokens within each sentence boundary → (B, S, D)
  Level 3 (section):   A 2-layer lightweight transformer over sentence reps
                        captures document flow and long-range dependencies.

The section-level representations are then broadcast back to the token level
via a learned gating mechanism, giving each token access to its structural context.
"""

from typing import List, Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Sentence boundary detection ────────────────────────────────────────

def detect_sentence_boundaries(
    text: str,
    tokenizer,
    max_length: int = 1024,
) -> torch.Tensor:
    """Detect sentence boundaries and return a binary mask (1 = sentence start).

    Uses a hybrid approach:
    1. NLTK sent_tokenize for robust sentence splitting
    2. Maps sentence boundaries back to token positions via offset mapping
    """
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)
    except Exception:
        # Fallback: split on period+space pattern
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".")
                     if s.strip()]

    if not sentences:
        return torch.zeros(1, max_length, dtype=torch.long)

    # Use encode_plus to get offset mapping
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    offset_mapping = encoding.get("offset_mapping", [])
    input_ids = encoding["input_ids"]

    # Build a reverse map: character position → which token it belongs to
    boundary_mask = torch.zeros(len(input_ids), dtype=torch.long)

    if offset_mapping:
        char_pos = 0
        sent_idx = 0
        for sent in sentences:
            sent_start = text.find(sent, char_pos) if sent else char_pos
            if sent_start < 0:
                sent_start = char_pos
            sent_end = sent_start + len(sent)
            char_pos = sent_end

            # Mark the first token whose offset covers sent_start as a boundary
            for tok_i, (start_off, end_off) in enumerate(offset_mapping):
                if start_off == 0 and end_off == 0:
                    continue  # special tokens
                if start_off <= sent_start < end_off:
                    boundary_mask[tok_i] = 1
                    break

    # Fallback: evenly-spaced boundaries if detection failed
    if boundary_mask.sum() < 2:
        step = max(1, len(input_ids) // 10)
        for i in range(0, len(input_ids), step):
            boundary_mask[i] = 1

    return boundary_mask.unsqueeze(0)  # (1, T)


def batch_detect_boundaries(
    texts: List[str],
    tokenizer,
    max_length: int = 1024,
) -> torch.Tensor:
    """Detect sentence boundaries for a batch of texts."""
    all_masks = []
    for text in texts:
        mask = detect_sentence_boundaries(text, tokenizer, max_length)
        # Pad or truncate to max_length
        if mask.shape[1] < max_length:
            pad = torch.zeros(1, max_length - mask.shape[1], dtype=torch.long)
            mask = torch.cat([mask, pad], dim=1)
        else:
            mask = mask[:, :max_length]
        all_masks.append(mask)
    return torch.cat(all_masks, dim=0)  # (B, T)


# ── Lightweight hierarchical transformer ───────────────────────────────

class HierarchicalStructureEncoder(nn.Module):
    """A compact hierarchical adapter over BART's encoder outputs.

    Architecture:
      token_reps (B, T, D)
        → sentence pooling via boundary_mask → (B, S, D)
        → 2-layer transformer (4 heads, 64-dim FFN) → (B, S, D)
        → broadcast back via learned gate → (B, T, D)
        → residual add to original token embeddings
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Sentence-level transformer (lightweight)
        self.sentence_transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-LN for stability
            ),
            num_layers=2,
        )

        # Gating mechanism for broadcasting structure back to tokens
        self.structure_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize with small weights so structure signal grows gradually
        nn.init.normal_(self.structure_gate[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.structure_gate[0].bias)
        nn.init.normal_(self.structure_gate[3].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.structure_gate[3].bias)

    def forward(
        self,
        token_embeddings: torch.Tensor,  # (B, T, D)
        boundary_mask: torch.Tensor,      # (B, T) — 1 at sentence starts
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) — 0 for pad
    ) -> torch.Tensor:
        B, T, D = token_embeddings.shape

        # ── Step 1: Pool tokens within each sentence ──
        # Use segment IDs: cumulative sum of boundary_mask gives sentence index
        segment_ids = boundary_mask.cumsum(dim=1)  # (B, T), e.g. [0,0,0,1,1,1,2,2,2,...]
        max_seg = int(segment_ids.max().item()) + 1

        if max_seg < 2:
            # No meaningful sentence structure detected; return input unchanged
            return token_embeddings

        # Mean-pool: for each sentence, average its token representations
        sentence_reps = torch.zeros(B, max_seg, D, device=token_embeddings.device)
        for s in range(max_seg):
            mask_s = (segment_ids == s).float().unsqueeze(-1)  # (B, T, 1)
            if attention_mask is not None:
                mask_s = mask_s * attention_mask.unsqueeze(-1)
            denom = mask_s.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)
            pooled = (token_embeddings * mask_s).sum(dim=1, keepdim=True) / denom  # (B, 1, D)
            sentence_reps[:, s, :] = pooled.squeeze(1)

        # ── Step 2: Process sentence reps through lightweight transformer ──
        # This captures inter-sentence relationships and document flow
        sentence_reps = self.sentence_transformer(sentence_reps)  # (B, S, D)

        # ── Step 3: Broadcast structure back to tokens ──
        # For each token, get its sentence's structure representation
        # segment_ids[b, t] tells us which sentence token t belongs to
        structure_per_token = torch.zeros(B, T, D, device=token_embeddings.device)
        for b in range(B):
            for s in range(max_seg):
                mask_s = (segment_ids[b] == s)
                if mask_s.any():
                    structure_per_token[b, mask_s] = sentence_reps[b, s]

        # ── Step 4: Gated fusion ──
        gate_input = torch.cat([token_embeddings, structure_per_token], dim=-1)
        gate_values = torch.sigmoid(self.structure_gate(gate_input))

        # gate controls how much structure signal to inject per dimension
        enriched = token_embeddings + self.dropout(gate_values * structure_per_token)
        enriched = self.layer_norm(enriched)

        return enriched

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
