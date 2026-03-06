"""
Eye-Tracking Encoder for ExplainMoE-ADHD v2.13.

This module implements the Eye-tracking encoder (Section 5.2.6):
- BiLSTM for temporal modeling
- Multi-head attention for sequence aggregation

Dataset: D8 (Wainstein)
Status: PERMANENTLY FROZEN after Phase 1 (never unfrozen in any phase)
Evaluation: Standalone linear probe ONLY, NOT through FuseMoE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EyeTrackingEncoder(nn.Module):
    """
    Eye-Tracking Encoder.
    
    Architecture:
    - BiLSTM for temporal processing of gaze data
    - Multi-head attention for sequence aggregation
    - Learned query vector for attention
    
    STATUS: Permanently frozen after Phase 1.
    NOT used in FuseMoE. Evaluated via standalone linear probe only.
    
    Input: (batch, time, 3) - x, y, pupil diameter
    Output: (batch, 256)
    """
    def __init__(
        self,
        # BiLSTM
        input_size: int = 3,  # x, y, pupil
        bilstm_hidden_size: int = 128,
        bilstm_num_layers: int = 2,
        
        # Attention
        attn_num_heads: int = 4,
        
        # Output
        latent_dim: int = 256,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            bidirectional=True,
            batch_first=True,
        )
        
        # Output dimension: bidirectional = 2 * hidden_size
        lstm_output_dim = bilstm_hidden_size * 2
        
        # Multi-head attention (expects embed_dim == lstm_output_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=attn_num_heads,
            batch_first=True,
        )
        
        # Learned query for attention
        self.query = nn.Parameter(torch.randn(1, 1, lstm_output_dim))
        
        # Output projection to latent_dim
        self.output_projection = nn.Linear(lstm_output_dim, latent_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(latent_dim)
    
    def forward(
        self,
        gaze_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            gaze_sequence: (batch, time, 3) - x, y, pupil diameter
        
        Returns:
            (batch, latent_dim)
        """
        batch_size = gaze_sequence.size(0)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(gaze_sequence)  # (B, T, 2*hidden)
        
        # Create query
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, 2*hidden)
        
        # Multi-head attention: query attends to BiLSTM output
        attn_out, _ = self.attention(
            query, lstm_out, lstm_out,
            need_weights=False,
        )  # (B, 1, 2*hidden)
        
        attn_out = attn_out.squeeze(1)  # (B, 2*hidden)
        
        # Output projection
        h = self.output_projection(attn_out)  # (B, latent_dim)
        
        # Layer norm
        h = self.layer_norm(h)
        
        return h


class EyeTrackingEncoderFactory:
    """Factory for creating Eye-Tracking encoders."""
    
    @classmethod
    def create(
        cls,
        latent_dim: int = 256,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        **kwargs
    ) -> EyeTrackingEncoder:
        encoder = EyeTrackingEncoder(
            latent_dim=latent_dim,
            **kwargs
        )
        
        # Load pretrained weights if specified (GazeBase pretraining)
        if pretrained and pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            encoder.load_state_dict(state_dict)
        
        return encoder


# =============================================================================
# STANDALONE LINEAR PROBE (Section 8.6)
# =============================================================================

class EyeTrackingLinearProbe(nn.Module):
    """
    Standalone linear probe for eye-tracking evaluation.
    
    Used for:
    - PCA dimensionality reduction (optional)
    - Logistic regression classification
    
    NOT part of FuseMoE. Evaluated separately.
    """
    def __init__(
        self,
        input_dim: int = 256,
        use_pca: bool = False,
        pca_components: int = 20,
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        if use_pca:
            self.pca = nn.Linear(input_dim, pca_components, bias=False)
            self.classifier = nn.Linear(pca_components, num_classes)
        else:
            self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - encoder output
        
        Returns:
            (batch, num_classes) - logits
        """
        if self.use_pca:
            x = self.pca(x)
        return self.classifier(x)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_eye_tracking_encoder():
    """Test the Eye-Tracking encoder."""
    batch_size = 4
    device = torch.device('cpu')
    
    encoder = EyeTrackingEncoder(latent_dim=256).to(device)
    
    # Test input: (batch, time, 3)
    gaze_sequence = torch.randn(batch_size, 100, 3).to(device)
    
    out = encoder(gaze_sequence=gaze_sequence)
    assert out.shape == (batch_size, 256), f"Got: {out.shape}"
    print(f"Γ£ô EyeTrackingEncoder: {out.shape}")
    
    # Test linear probe
    probe = EyeTrackingLinearProbe(input_dim=256, use_pca=False).to(device)
    probe_pca = EyeTrackingLinearProbe(input_dim=256, use_pca=True, pca_components=20).to(device)
    
    logits = probe(out)
    assert logits.shape == (batch_size, 2), f"Got: {logits.shape}"
    print(f"Γ£ô EyeTrackingLinearProbe: {logits.shape}")
    
    logits_pca = probe_pca(out)
    assert logits_pca.shape == (batch_size, 2), f"Got: {logits_pca.shape}"
    print(f"Γ£ô EyeTrackingLinearProbe (with PCA): {logits_pca.shape}")
    
    print("\nEye-Tracking encoder tests passed!")


if __name__ == '__main__':
    test_eye_tracking_encoder()
