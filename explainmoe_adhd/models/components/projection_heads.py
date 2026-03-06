"""
Projection Heads for ExplainMoE-ADHD v2.13.

This module implements modality-specific projection heads (Section 5.4):
- One per modality pathway (5 total)
- Depth is configurable (1, 2, or 3 layers)
- Used for MMD alignment in Phase 3

Training behavior:
- Phase 3: Sole trainable component (MMD alignment)
- Phase 4: Trainable at 0.1├ù main LR (preserve alignment)
- Phase 5: Trainable at 1e-5 (same as all components)
"""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionHead(nn.Module):
    """
    Projection Head for MMD alignment.
    
    Architecture:
        Linear ΓåÆ LayerNorm ΓåÆ GELU ΓåÆ (repeat depth-1 times)
    
    Default depth: 2 layers
    
    Input: (batch, dim) - Encoder output h_m
    Output: (batch, dim) - Projected representation z_m
    """
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 256,
        depth: int = 2,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        
        layers = []
        in_dim = input_dim
        
        for i in range(depth):
            layers.append(nn.Linear(in_dim, output_dim))
            
            if i < depth - 1:  # Intermediate layers get LayerNorm + GELU
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.GELU())
            
            in_dim = output_dim
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights (Xavier uniform)
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, input_dim) - Encoder output
        
        Returns:
            (batch, output_dim) - Projected representation
        """
        return self.net(h)


class ProjectionHeadFactory:
    """
    Factory for creating projection heads.
    
    Manages all 5 modality-specific projection heads.
    """
    
    MODALITIES = [
        'child_eeg_19ch',
        'child_eeg_10ch', 
        'adult_eeg_5ch',
        'clinical',
        'actigraphy',
    ]
    
    @classmethod
    def create(
        cls,
        modality: str,
        input_dim: int = 256,
        output_dim: int = 256,
        depth: int = 2,
    ) -> ProjectionHead:
        """Create a projection head for a specific modality."""
        
        if modality not in cls.MODALITIES:
            raise ValueError(
                f"Unknown modality: {modality}. "
                f"Available: {cls.MODALITIES}"
            )
        
        return ProjectionHead(
            input_dim=input_dim,
            output_dim=output_dim,
            depth=depth,
        )
    
    @classmethod
    def create_all(
        cls,
        input_dim: int = 256,
        output_dim: int = 256,
        depth: int = 2,
    ) -> nn.ModuleDict:
        """Create all projection heads as a ModuleDict."""
        
        heads = nn.ModuleDict()
        for modality in cls.MODALITIES:
            heads[modality] = ProjectionHead(
                input_dim=input_dim,
                output_dim=output_dim,
                depth=depth,
            )
        
        return heads
    
    @classmethod
    def get_modality_index(cls, modality: str) -> int:
        """Get router index for a modality."""
        return cls.MODALITIES.index(modality)


# =============================================================================
# ABLATION VARIANTS (Section 10, A2)
# =============================================================================

class ProjectionHeadDepth1(ProjectionHead):
    """Ablation A2: Single layer projection head."""
    def __init__(self, **kwargs):
        kwargs['depth'] = 1
        super().__init__(**kwargs)


class ProjectionHeadDepth3(ProjectionHead):
    """Ablation A2: Three layer projection head."""
    def __init__(self, **kwargs):
        kwargs['depth'] = 3
        super().__init__(**kwargs)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_projection_heads():
    """Test projection heads."""
    batch_size = 4
    device = torch.device('cpu')
    
    # Test single head
    head = ProjectionHead(input_dim=256, output_dim=256, depth=2).to(device)
    h = torch.randn(batch_size, 256).to(device)
    z = head(h)
    assert z.shape == (batch_size, 256), f"Got: {z.shape}"
    print(f"Γ£ô ProjectionHead (depth=2): {z.shape}")
    
    # Test depth variants
    head1 = ProjectionHeadDepth1(input_dim=256, output_dim=256).to(device)
    head3 = ProjectionHeadDepth3(input_dim=256, output_dim=256).to(device)
    
    z1 = head1(h)
    z3 = head3(h)
    
    assert z1.shape == (batch_size, 256), f"Got: {z1.shape}"
    assert z3.shape == (batch_size, 256), f"Got: {z3.shape}"
    print(f"Γ£ô ProjectionHeadDepth1: {z1.shape}")
    print(f"Γ£ô ProjectionHeadDepth3: {z3.shape}")
    
    # Test factory
    heads = ProjectionHeadFactory.create_all()
    assert isinstance(heads, nn.ModuleDict)
    assert len(heads) == 5
    print(f"Γ£ô ProjectionHeadFactory.create_all: {len(heads)} heads")
    
    # Test get_modality_index
    assert ProjectionHeadFactory.get_modality_index('child_eeg_19ch') == 0
    assert ProjectionHeadFactory.get_modality_index('clinical') == 3
    print("Γ£ô ProjectionHeadFactory.get_modality_index")
    
    print("\nProjection head tests passed!")


if __name__ == '__main__':
    test_projection_heads()
