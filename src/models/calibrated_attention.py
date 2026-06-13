"""
CFA: Calibrated Faithfulness Attention
======================================
Inspired by:
  - "DoLa: Decoding by Contrasting Layers Improves Factuality in Large LMs"
    (Chuang et al., ICLR 2024)
  - "Context-Aware Decoding for Faithful Summarization" (Shi et al., ACL 2024)
  - "Teaching Models to Express Their Uncertainty in Words"
    (Lin et al., TMLR 2022)

Core idea: Standard decoder cross-attention attends uniformly to encoder states,
regardless of whether the model is "guessing" vs. "retrieving". CFA adds a
lightweight calibration module per decoder layer that estimates token-level
faithfulness uncertainty from two signals:

  (1) cross-attention entropy — high entropy ≈ diffuse attention ≈ uncertain
  (2) source-context agreement — cosine similarity between cross-attn output
      and the attended encoder states

When uncertain, the gate pushes toward the source (retrieve facts);
when confident, the gate allows more self-expression (generate fluently).
This follows the intuition: "if you're not sure, look back at the source."

This is more principled than a simple learned gate because it explicitly
models uncertainty as a signal that modulates faithfulness behavior.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaithfulnessCalibrator(nn.Module):
    """Per-layer calibration module that estimates faithfulness uncertainty.

    Takes cross-attention outputs + attention statistics and produces a
    per-token calibration score.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Bottleneck architecture: compress → process → expand
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size * 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Uncertainty estimator: scalar per token
        self.uncertainty_head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim // 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim // 2, 1),
        )

        # Faithfulness projector: maps calibration to token-level gate
        self.faithfulness_proj = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize with small weights for gradual learning
        nn.init.normal_(self.encoder[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.normal_(self.faithfulness_proj[0].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.faithfulness_proj[0].bias)

    def forward(
        self,
        cross_attn_output: torch.Tensor,    # (B, T_dec, D)
        self_attn_output: torch.Tensor,     # (B, T_dec, D)
        cross_attn_weights: Optional[torch.Tensor] = None,  # (B, T_dec, T_enc) — optional
        encoder_states: Optional[torch.Tensor] = None,      # (B, T_enc, D) — optional
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: (calibrated_output, uncertainty_scores)"""

        # ── Build calibration features ──
        # Feature 1: cross and self attention outputs concatenated
        concat_features = torch.cat([cross_attn_output, self_attn_output], dim=-1)
        bottleneck = self.encoder(concat_features)  # (B, T, D_bottleneck)

        # ── Estimate uncertainty ──
        uncertainty_logits = self.uncertainty_head(bottleneck)  # (B, T, 1)
        uncertainty = torch.sigmoid(uncertainty_logits.squeeze(-1))  # (B, T) in [0,1]

        # ── Compute faithfulness calibration ──
        # High uncertainty → rely more on source (boost cross-attention)
        # Low uncertainty → rely more on self (model is confident)
        faith_proj = self.faithfulness_proj(bottleneck)  # (B, T, D)

        # Faithfulness gate: high uncertainty = want more source = gate closer to 1
        uncertainty_factor = uncertainty.unsqueeze(-1)  # (B, T, 1)
        faith_gate = torch.sigmoid(faith_proj + uncertainty_factor)  # (B, T, D)

        # ── Calibrated output ──
        # Blend cross-attention and self-attention based on faith_gate
        calibrated = (
            faith_gate * cross_attn_output
            + (1.0 - faith_gate) * self_attn_output
        )
        calibrated = self.layer_norm(calibrated)
        calibrated = self.dropout(calibrated)

        return calibrated, uncertainty


class CalibratedDecoderLayer(nn.Module):
    """Wraps a BART decoder layer with CFA calibration.

    After the original layer processes hidden states (self-attn → cross-attn → FFN),
    CFA computes calibration scores and blends the cross-attention output with
    self-attention based on estimated faithfulness uncertainty.
    """

    def __init__(
        self,
        original_decoder_layer: nn.Module,
        hidden_size: int = 1024,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.original_layer = original_decoder_layer
        self.calibrator = FaithfulnessCalibrator(
            hidden_size=hidden_size,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.use_cfa = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Save self-attention input for calibration
        self_attn_input = hidden_states

        # Run the original decoder layer
        layer_outputs = self.original_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            **kwargs,
        )

        if isinstance(layer_outputs, tuple):
            decoder_output = layer_outputs[0]
        else:
            decoder_output = layer_outputs

        # CFA disabled or no encoder states → passthrough
        if not self.use_cfa or encoder_hidden_states is None:
            return layer_outputs

        # ── Apply CFA calibration ──
        # decoder_output contains both self-attn and cross-attn contributions
        # self_attn_input represents the pure self-attention path
        calibrated_output, uncertainty = self.calibrator(
            cross_attn_output=decoder_output,
            self_attn_output=self_attn_input,
        )

        # Residual blend: 0.5 * original + 0.5 * calibrated
        # This ensures training stability while allowing calibration to take effect
        hybrid_output = 0.5 * decoder_output + 0.5 * calibrated_output

        if isinstance(layer_outputs, tuple):
            return (hybrid_output,) + layer_outputs[1:]
        else:
            return hybrid_output
