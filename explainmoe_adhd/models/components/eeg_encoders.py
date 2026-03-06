"""
EEG Encoders for ExplainMoE-ADHD v2.13.

This module implements three EEG encoder variants:
- Child EEG 19-ch (clinical-grade): Section 5.2.1
- Child EEG 10-ch (EMOTIV EPOC): Section 5.2.2
- Adult EEG 5-ch: Section 5.2.3

All encoders use EEGNet + Transformer architecture with hardware domain tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EEGNet(nn.Module):
    """
    EEGNet architecture adapted from Lawhern et al. (2018).
    
    Input: (batch, channels, time_samples)
    Output: (batch, F2 * time_samples // 32)
    """
    def __init__(
        self,
        channels: int,
        temporal_filters: int = 8,
        depth_multiplier: int = 2,
        separable_filters: int = 16,
        temporal_kernel_size: int = 64,
        separable_kernel_size: int = 16,
        pool1_size: int = 4,
        pool2_size: int = 8,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        self.channels = channels
        self.F1 = temporal_filters
        self.D = depth_multiplier
        self.F2 = separable_filters
        
        # Temporal convolution
        self.temporal_conv = nn.Conv2d(
            1, 
            self.F1, 
            (1, temporal_kernel_size), 
            padding="same",
            bias=False
        )
        
        # Depthwise convolution (per-channel spatial)
        self.depthwise_conv = nn.Conv2d(
            self.F1,
            self.F1 * self.D,
            (channels, 1),
            groups=self.F1,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(self.F1 * self.D)
        
        # Separable convolution
        self.separable_conv = nn.Conv2d(
            self.F1 * self.D,
            self.F2,
            (1, separable_kernel_size),
            padding="same",
            bias=False
        )
        
        self.bn2 = nn.BatchNorm2d(self.F2)
        
        self.pool1_size = pool1_size
        self.pool2_size = pool2_size
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) - e.g., (B, 19, 256)
        Returns:
            (batch, F2 * time // 32)
        """
        # x: (B, 1, Ch, T)
        x = x.unsqueeze(1)
        
        # Temporal conv
        x = self.temporal_conv(x)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, self.pool1_size))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        
        # Separable conv
        x = self.separable_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, self.pool2_size))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        
        # Flatten: (B, F2, T')
        x = x.flatten(1)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


class EEGEncoder(nn.Module):
    """
    EEG Encoder: EEGNet + Transformer with hardware token.
    
    This is the base class for all three EEG encoder variants.
    
    Architecture:
        1. EEGNet feature extraction
        2. Linear projection to Transformer dim
        3. Prepend [CLS] token and hardware token
        4. Transformer encoder
        5. Return CLS token output
    
    Input: (batch, channels, time_samples) - e.g., (B, 19, 256)
    Output: (batch, latent_dim) - e.g., (B, 256)
    """
    def __init__(
        self,
        channels: int,
        hardware_token_id: int,
        eegnet_config: Optional[dict] = None,
        transformer_d_model: int = 256,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        transformer_dim_feedforward: int = 512,
        transformer_dropout: float = 0.1,
        latent_dim: int = 256,
    ):
        super().__init__()
        
        self.channels = channels
        self.hardware_token_id = hardware_token_id
        self.latent_dim = latent_dim
        
        # EEGNet configuration
        if eegnet_config is None:
            eegnet_config = {}
        
        # Calculate EEGNet output size
        # After EEGNet: F2 * (T // (pool1 * pool2)) features
        # We use a lazy linear layer to handle variable T dynamically
        
        self.eegnet = EEGNet(
            channels=channels,
            temporal_filters=eegnet_config.get('temporal_filters', 8),
            depth_multiplier=eegnet_config.get('depth_multiplier', 2),
            separable_filters=eegnet_config.get('separable_filters', 16),
            temporal_kernel_size=eegnet_config.get('temporal_kernel_size', 64),
            separable_kernel_size=eegnet_config.get('separable_kernel_size', 16),
            pool1_size=eegnet_config.get('pool1_size', 4),
            pool2_size=eegnet_config.get('pool2_size', 8),
            dropout_rate=eegnet_config.get('dropout_rate', 0.5),
        )
        
        # Use LazyLinear to handle variable T (T=256 for 128Hz, T=512 for 256Hz)
        # EEGNet output size depends on T: F2 * (T // (pool1 * pool2))
        self.eegnet_projection = nn.LazyLinear(latent_dim)
        
        # Hardware token embedding
        self.hw_embedding = nn.Embedding(4, latent_dim)  # 4 hardware types
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(latent_dim, max_len=100)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_num_layers,
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        hw_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) - EEG data
            hw_id: Hardware token ID (optional, uses default if None)
        Returns:
            (batch, latent_dim) - Encoded representation
        """
        batch_size = x.size(0)
        
        # EEGNet features
        eeg_features = self.eegnet(x)  # (B, eegnet_out_dim)
        
        # Project to Transformer dim
        h = self.eegnet_projection(eeg_features)  # (B, latent_dim)
        
        # Add sequence dimension for Transformer
        h = h.unsqueeze(1)  # (B, 1, latent_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, latent_dim)
        h = torch.cat([cls_tokens, h], dim=1)  # (B, 2, latent_dim)
        
        # Add positional encoding
        h = self.pos_encoding(h)
        
        # Add hardware token (at position 1, after CLS)
        if hw_id is None:
            hw_id = self.hardware_token_id
        hw_token = self.hw_embedding(
            torch.tensor(hw_id, device=x.device).expand(batch_size)
        ).unsqueeze(1)  # (B, 1, latent_dim)
        
        # Insert hardware token at position 1
        h = torch.cat([h[:, :1, :], hw_token, h[:, 1:, :]], dim=1)  # (B, 3, latent_dim)
        
        # Transformer
        h = self.transformer(h)  # (B, 3, latent_dim)
        
        # Extract CLS token output (position 0)
        h = h[:, 0, :]  # (B, latent_dim)
        
        return h


class ChildEEG19chEncoder(EEGEncoder):
    """
    Child EEG 19-channel encoder (clinical-grade).
    Hardware token ID: 0 (clinical_19ch_wet)
    Dataset: D1 (IEEE DataPort)
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault('channels', 19)
        kwargs.setdefault('hardware_token_id', 0)
        super().__init__(**kwargs)


class ChildEEG10chEncoder(EEGEncoder):
    """
    Child EEG 10-channel encoder (EMOTIV EPOC, 10 retained channels).
    Hardware token ID: 1 (emotiv_10ch_saline)
    Dataset: D2 (FOCUS)
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault('channels', 10)
        kwargs.setdefault('hardware_token_id', 1)
        super().__init__(**kwargs)


class AdultEEG5chEncoder(EEGEncoder):
    """
    Adult EEG 5-channel encoder.
    Hardware token ID: 2 (mendeley_5ch)
    Dataset: D3 (Mendeley)
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault('channels', 5)
        kwargs.setdefault('hardware_token_id', 2)
        super().__init__(**kwargs)


class EEGEncoderFactory:
    """
    Factory for creating EEG encoders.
    
    Usage:
        factory = EEGEncoderFactory()
        encoder = factory.create('child_19ch')
        encoder = factory.create('child_10ch', pretrained_path='...')
    """
    
    ENCODER_TYPES = {
        'child_19ch': ChildEEG19chEncoder,
        'child_10ch': ChildEEG10chEncoder,
        'adult_5ch': AdultEEG5chEncoder,
    }
    
    CHANNELS = {
        'child_19ch': 19,
        'child_10ch': 10,
        'adult_5ch': 5,
    }
    
    HARDWARE_IDS = {
        'child_19ch': 0,
        'child_10ch': 1,
        'adult_5ch': 2,
    }
    
    @classmethod
    def create(
        cls,
        encoder_type: str,
        latent_dim: int = 256,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        **kwargs
    ) -> EEGEncoder:
        """
        Create an EEG encoder.
        
        Args:
            encoder_type: 'child_19ch', 'child_10ch', or 'adult_5ch'
            latent_dim: Output dimension
            pretrained: Whether to load pretrained weights
            pretrained_path: Path to pretrained weights
            **kwargs: Additional arguments to encoder
        
        Returns:
            EEGEncoder instance
        """
        if encoder_type not in cls.ENCODER_TYPES:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. "
                f"Available: {list(cls.ENCODER_TYPES.keys())}"
            )
        
        encoder_cls = cls.ENCODER_TYPES[encoder_type]
        
        # Default config
        config = {
            'latent_dim': latent_dim,
            'channels': cls.CHANNELS[encoder_type],
            'hardware_token_id': cls.HARDWARE_IDS[encoder_type],
            **kwargs
        }
        
        encoder = encoder_cls(**config)
        
        # Load pretrained weights if specified
        if pretrained and pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            encoder.load_state_dict(state_dict)
        
        return encoder
    
    @classmethod
    def list_encoders(cls) -> list:
        """List available encoder types."""
        return list(cls.ENCODER_TYPES.keys())


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_eeg_encoders():
    """Test all three EEG encoder variants."""
    batch_size = 4
    device = torch.device('cpu')
    
    # Test 19-channel encoder
    encoder_19ch = ChildEEG19chEncoder(latent_dim=256).to(device)
    x_19ch = torch.randn(batch_size, 19, 256).to(device)
    out_19ch = encoder_19ch(x_19ch)
    assert out_19ch.shape == (batch_size, 256), f"19ch: {out_19ch.shape}"
    print(f"Γ£ô ChildEEG19chEncoder: {out_19ch.shape}")
    
    # Test 10-channel encoder
    encoder_10ch = ChildEEG10chEncoder(latent_dim=256).to(device)
    x_10ch = torch.randn(batch_size, 10, 256).to(device)
    out_10ch = encoder_10ch(x_10ch)
    assert out_10ch.shape == (batch_size, 256), f"10ch: {out_10ch.shape}"
    print(f"Γ£ô ChildEEG10chEncoder: {out_10ch.shape}")
    
    # Test 5-channel encoder
    encoder_5ch = AdultEEG5chEncoder(latent_dim=256).to(device)
    x_5ch = torch.randn(batch_size, 5, 512).to(device)  # 512 samples for 256Hz
    out_5ch = encoder_5ch(x_5ch)
    assert out_5ch.shape == (batch_size, 256), f"5ch: {out_5ch.shape}"
    print(f"Γ£ô AdultEEG5chEncoder: {out_5ch.shape}")
    
    # Test factory
    encoder = EEGEncoderFactory.create('child_19ch')
    assert isinstance(encoder, ChildEEG19chEncoder)
    print("Γ£ô EEGEncoderFactory works")
    
    print("\nAll EEG encoder tests passed!")


if __name__ == '__main__':
    test_eeg_encoders()
