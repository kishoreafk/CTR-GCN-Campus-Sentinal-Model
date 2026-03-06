"""
Actigraphy Encoder for ExplainMoE-ADHD v2.13.

This module implements the Actigraphy encoder (Section 5.2.5):
- ResNet1D for time-series feature extraction
- BiLSTM for temporal modeling
- Auxiliary MLP for age/sex features
- Merge MLP to combine both branches

Dataset: D7 (Hyperaktiv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResNet1DBlock(nn.Module):
    """Basic ResNet-1D block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """1D ResNet for time-series feature extraction."""
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        num_blocks: int = 3,
        output_features: int = 224,
    ):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet blocks
        self.layer1 = self._make_layer(base_channels, base_channels, num_blocks)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, num_blocks, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, num_blocks, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.fc = nn.Linear(base_channels * 4, output_features)
    
    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        
        layers = []
        layers.append(ResNet1DBlock(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, num_blocks):
            layers.append(ResNet1DBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time) - e.g., (B, 4, 1000)
        Returns:
            (batch, output_features)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        
        return x


class ActigraphyEncoder(nn.Module):
    """
    Actigraphy Encoder.
    
    Two-branch architecture:
    - Branch A: ResNet1D + BiLSTM for time-series
    - Branch B: MLP for auxiliary features (age, sex)
    
    Input:
        - timeseries: (batch, 4, time) - 3-axis accel + heart rate
        - age: (batch,) - age
        - sex: (batch,) - sex (0/1)
    
    Output: (batch, 256)
    """
    def __init__(
        self,
        # Time-series branch
        ts_in_channels: int = 4,  # 3 accel + 1 HR
        ts_resnet_channels: int = 32,
        ts_resnet_blocks: int = 3,
        ts_output_features: int = 224,
        
        # BiLSTM
        bilstm_hidden_size: int = 112,
        bilstm_num_layers: int = 2,
        
        # Auxiliary branch
        aux_input_dim: int = 2,  # age, sex
        aux_output_dim: int = 32,
        
        # Merge MLP
        merge_input_dim: int = 256,  # 224 + 32
        merge_hidden_dim: int = 256,
        
        # Output
        latent_dim: int = 256,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Branch A: ResNet1D
        self.resnet = ResNet1D(
            in_channels=ts_in_channels,
            base_channels=ts_resnet_channels,
            num_blocks=ts_resnet_blocks,
            output_features=ts_output_features,
        )
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=ts_output_features,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            bidirectional=True,
            batch_first=True,
        )
        
        # Branch B: Auxiliary MLP
        self.aux_mlp = nn.Sequential(
            nn.Linear(aux_input_dim, aux_output_dim),
            nn.GELU(),
        )
        
        # Merge MLP (256ΓåÆ256 by default, no bottleneck per spec)
        self.merge_mlp = nn.Sequential(
            nn.Linear(merge_input_dim, merge_hidden_dim),
            nn.GELU(),
            nn.Linear(merge_hidden_dim, latent_dim),
        )
    
    def forward(
        self,
        timeseries: torch.Tensor,
        age: torch.Tensor,
        sex: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            timeseries: (batch, 4, time) - 3-axis accel + heart rate
            age: (batch,) - age
            sex: (batch,) - sex (0/1)
        
        Returns:
            (batch, latent_dim)
        """
        batch_size = timeseries.size(0)
        
        # Branch A: Time-series
        h_ts = self.resnet(timeseries)  # (B, 224)
        
        # BiLSTM: take last hidden state
        h_ts = h_ts.unsqueeze(1)  # (B, 1, 224)
        lstm_out, (h_n, c_n) = self.bilstm(h_ts)
        
        # Concatenate forward and backward final hidden states
        h_forward = h_n[-2, :, :]  # (B, 112)
        h_backward = h_n[-1, :, :]  # (B, 112)
        h_ts = torch.cat([h_forward, h_backward], dim=-1)  # (B, 224)
        
        # Branch B: Auxiliary
        aux_input = torch.stack([age, sex], dim=-1).float()  # (B, 2)
        h_aux = self.aux_mlp(aux_input)  # (B, 32)
        
        # Merge branches
        h_combined = torch.cat([h_ts, h_aux], dim=-1)  # (B, 256)
        h = self.merge_mlp(h_combined)  # (B, latent_dim)
        
        return h


class ActigraphyEncoderFactory:
    """Factory for creating Actigraphy encoders."""
    
    @classmethod
    def create(
        cls,
        latent_dim: int = 256,
        **kwargs
    ) -> ActigraphyEncoder:
        return ActigraphyEncoder(
            latent_dim=latent_dim,
            **kwargs
        )


# =============================================================================
# ABLATION VARIANTS (Section 10, A13)
# =============================================================================

class ActigraphyEncoderWithBottleneck(ActigraphyEncoder):
    """
    Ablation A13: Actigraphy encoder with bottleneck in merge MLP.
    Uses 256ΓåÆ128ΓåÆ256 instead of 256ΓåÆ256.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Override merge MLP with bottleneck
        self.merge_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_actigraphy_encoder():
    """Test the Actigraphy encoder."""
    batch_size = 4
    device = torch.device('cpu')
    
    encoder = ActigraphyEncoder(latent_dim=256).to(device)
    
    # Test input
    timeseries = torch.randn(batch_size, 4, 1000).to(device)  # 4 channels, 1000 samples
    age = torch.randint(18, 65, (batch_size,)).float().to(device)
    sex = torch.randint(0, 2, (batch_size,)).to(device)
    
    out = encoder(timeseries=timeseries, age=age, sex=sex)
    assert out.shape == (batch_size, 256), f"Got: {out.shape}"
    print(f"Γ£ô ActigraphyEncoder: {out.shape}")
    
    # Test bottleneck variant
    encoder_bottleneck = ActigraphyEncoderWithBottleneck(latent_dim=256).to(device)
    out_bottleneck = encoder_bottleneck(timeseries=timeseries, age=age, sex=sex)
    assert out_bottleneck.shape == (batch_size, 256), f"Got: {out_bottleneck.shape}"
    print(f"Γ£ô ActigraphyEncoderWithBottleneck: {out_bottleneck.shape}")
    
    print("\nActigraphy encoder tests passed!")


if __name__ == '__main__':
    test_actigraphy_encoder()
