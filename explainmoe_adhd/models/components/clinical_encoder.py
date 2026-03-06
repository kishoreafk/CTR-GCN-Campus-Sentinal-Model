"""
Clinical/fMRI Encoder for ExplainMoE-ADHD v2.13.

This module implements the Clinical/fMRI encoder (Section 5.2.4):
- FT-Transformer for tabular features (Tier 1)
- MLP for fMRI connectivity features
- Merge MLP to combine both branches

Datasets: D5 (ADHD-200), D6 (ds002424)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class FTTransformerTabular(nn.Module):
    """
    FT-Transformer (Tabular Feature Transformer) for tabular features.
    
    This is a simplified version for the clinical encoder.
    Processes: age, sex, handedness, IQ, site_id, dataset_source
    
    Input: (batch, num_features) with categorical and continuous features
    Output: (batch, dim) - CLS token representation
    """
    def __init__(
        self,
        num_categories: list,
        num_continuous: int,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        dim_head: int = 32,
        ff_dim: int = 256,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        # Missing value handling
        miss_token_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_categories = num_categories
        self.num_continuous = num_continuous
        self.dim = dim
        self.depth = depth
        
        # Categorical embeddings (one per feature)
        self.category_embeddings = nn.ModuleList([
            nn.Embedding(num_cats, dim)
            for num_cats in num_categories
        ])
        
        # Continuous feature processing
        if num_continuous > 0:
            self.cont_embedding = nn.Linear(num_continuous, dim)
        
        # Missing value tokens (learned)
        if miss_token_dim is not None:
            self.miss_token_IQ = nn.Parameter(torch.randn(miss_token_dim))
            self.miss_token_age = nn.Parameter(torch.randn(miss_token_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(
                    dim, heads, dropout=attn_dropout, batch_first=True
                ),
                'norm1': nn.LayerNorm(dim),
                'ff': nn.Sequential(
                    nn.Linear(dim, ff_dim),
                    nn.GELU(),
                    nn.Dropout(ff_dropout),
                    nn.Linear(ff_dim, dim),
                    nn.Dropout(ff_dropout),
                ),
                'norm2': nn.LayerNorm(dim),
            }))
        
        self.pool = 'cls'
    
    def forward(
        self,
        categorical_features: Optional[torch.Tensor] = None,
        continuous_features: Optional[torch.Tensor] = None,
        missing_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            categorical_features: (batch, num_categories) - int tensor
            continuous_features: (batch, num_continuous) - float tensor
            missing_mask: dict mapping feature names to boolean masks
        Returns:
            (batch, dim)
        """
        batch_size = categorical_features.size(0) if categorical_features is not None else continuous_features.size(0)
        
        tokens = []
        
        # Process categorical features
        if categorical_features is not None:
            for i, emb_layer in enumerate(self.category_embeddings):
                cat_feat = categorical_features[:, i]
                tokens.append(emb_layer(cat_feat))
        
        # Process continuous features
        if continuous_features is not None:
            cont_emb = self.cont_embedding(continuous_features)
            tokens.append(cont_emb)
        
        # Stack tokens: (batch, num_features, dim)
        x = torch.stack(tokens, dim=1)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Apply transformer layers
        for layer in self.layers:
            # Self-attention
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # Feed-forward
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)
        
        # Return CLS token
        return x[:, 0, :]


class ClinicalEncoder(nn.Module):
    """
    Clinical/fMRI Encoder.
    
    Two-branch architecture:
    - Branch A: FT-Transformer for tabular features (Tier 1)
    - Branch B: MLP for fMRI connectivity features
    
    Input:
        - tabular: (batch, 6) - age, sex, handedness, IQ, site_id, dataset_source
        - fmri: (batch, 4006) - connectivity features + mean_FD
        - mean_fd: (batch,) - framewise displacement
    
    Output: (batch, 256)
    """
    def __init__(
        self,
        # Tabular branch (FT-Transformer)
        num_categories: list = None,
        num_continuous: int = 2,
        ft_dim: int = 64,
        ft_depth: int = 2,
        ft_heads: int = 4,
        ft_dim_head: int = 32,
        ft_ff_dim: int = 256,
        ft_attn_dropout: float = 0.1,
        ft_ff_dropout: float = 0.1,
        
        # fMRI branch
        fmri_input_dim: int = 4006,
        fmri_hidden_dim: int = 512,
        fmri_output_dim: int = 128,
        fmri_dropout: float = 0.3,
        
        # Tabular projection
        tabular_output_dim: int = 128,
        
        # Merge MLP
        merge_hidden_dim: int = 256,
        
        # Output
        latent_dim: int = 256,
        
        # Missing value handling
        use_missing_tokens: bool = True,
    ):
        super().__init__()
        
        # Default values
        if num_categories is None:
            # sex (2), handedness (4), site_id (20), dataset_source (10)
            num_categories = [2, 4, 20, 10]
        
        self.num_categories = num_categories
        self.num_continuous = num_continuous
        self.latent_dim = latent_dim
        
        # Branch A: FT-Transformer for tabular
        self.ft_transformer = FTTransformerTabular(
            num_categories=num_categories,
            num_continuous=num_continuous,
            dim=ft_dim,
            depth=ft_depth,
            heads=ft_heads,
            dim_head=ft_dim_head,
            ff_dim=ft_ff_dim,
            attn_dropout=ft_attn_dropout,
            ff_dropout=ft_ff_dropout,
            miss_token_dim=ft_dim if use_missing_tokens else None,
        )
        self.tabular_proj = nn.Linear(ft_dim, tabular_output_dim)
        
        # Branch B: fMRI MLP
        self.fmri_mlp = nn.Sequential(
            nn.Linear(fmri_input_dim, fmri_hidden_dim),
            nn.GELU(),
            nn.Dropout(fmri_dropout),
            nn.Linear(fmri_hidden_dim, fmri_output_dim),
            nn.GELU(),
            nn.Dropout(fmri_dropout),
        )
        
        # Merge MLP
        self.merge_mlp = nn.Sequential(
            nn.Linear(tabular_output_dim + fmri_output_dim, merge_hidden_dim),
            nn.GELU(),
            nn.Linear(merge_hidden_dim, latent_dim),
        )
    
    def forward(
        self,
        tabular_features: torch.Tensor,
        fmri_features: Optional[torch.Tensor] = None,
        mean_fd: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tabular_features: (batch, 6) - [age, sex, handedness, IQ, site_id, dataset_source]
            fmri_features: (batch, 4005) - connectivity features (optional)
            mean_fd: (batch,) - mean framewise displacement (optional)
        
        Returns:
            (batch, latent_dim)
        """
        batch_size = tabular_features.size(0)
        
        # Parse tabular features
        # Feature order: [age, sex, handedness, IQ, site_id, dataset_source]
        # Continuous: age (idx 0), IQ (idx 3)
        continuous = torch.stack([tabular_features[:, 0], tabular_features[:, 3]], dim=-1)  # (B, 2)
        # Categorical: sex (idx 1), handedness (idx 2), site_id (idx 4), dataset_source (idx 5)
        categorical = torch.stack([
            tabular_features[:, 1],
            tabular_features[:, 2],
            tabular_features[:, 4],
            tabular_features[:, 5],
        ], dim=-1).long()  # (B, 4)
        
        # Branch A: FT-Transformer
        h_tabular = self.ft_transformer(
            categorical_features=categorical,
            continuous_features=continuous,
        )
        h_tabular = self.tabular_proj(h_tabular)  # (B, 128)
        
        # Branch B: fMRI
        if fmri_features is not None:
            # Concatenate connectivity with mean_FD
            if mean_fd is not None:
                mean_fd = mean_fd.unsqueeze(-1)  # (B, 1)
                fmri_input = torch.cat([fmri_features, mean_fd], dim=-1)  # (B, 4006)
            else:
                fmri_input = fmri_features
            
            h_fmri = self.fmri_mlp(fmri_input)  # (B, 128)
        else:
            # Zero padding if no fMRI
            h_fmri = torch.zeros(batch_size, 128, device=tabular_features.device)
        
        # Merge branches
        h_combined = torch.cat([h_tabular, h_fmri], dim=-1)  # (B, 256)
        h = self.merge_mlp(h_combined)  # (B, latent_dim)
        
        return h


class ClinicalEncoderFactory:
    """Factory for creating Clinical/fMRI encoders."""
    
    @classmethod
    def create(
        cls,
        latent_dim: int = 256,
        **kwargs
    ) -> ClinicalEncoder:
        return ClinicalEncoder(
            latent_dim=latent_dim,
            **kwargs
        )


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_clinical_encoder():
    """Test the Clinical/fMRI encoder."""
    batch_size = 4
    device = torch.device('cpu')
    
    encoder = ClinicalEncoder(latent_dim=256).to(device)
    
    # Test with both tabular and fMRI
    # Feature order: [age, sex, handedness, IQ, site_id, dataset_source]
    # Continuous: age (float), IQ (float)
    # Categorical: sex (0-1), handedness (0-3), site_id (0-19), dataset_source (0-9)
    tabular = torch.zeros(batch_size, 6).to(device)
    tabular[:, 0] = torch.randn(batch_size)  # age (continuous)
    tabular[:, 1] = torch.randint(0, 2, (batch_size,)).float()  # sex
    tabular[:, 2] = torch.randint(0, 4, (batch_size,)).float()  # handedness
    tabular[:, 3] = torch.randn(batch_size)  # IQ (continuous)
    tabular[:, 4] = torch.randint(0, 20, (batch_size,)).float()  # site_id
    tabular[:, 5] = torch.randint(0, 10, (batch_size,)).float()  # dataset_source
    fmri = torch.randn(batch_size, 4005).to(device)
    mean_fd = torch.randn(batch_size).to(device)
    
    out = encoder(tabular_features=tabular, fmri_features=fmri, mean_fd=mean_fd)
    assert out.shape == (batch_size, 256), f"Got: {out.shape}"
    print(f"Γ£ô ClinicalEncoder (with fMRI): {out.shape}")
    
    # Test with tabular only
    out_tab = encoder(tabular_features=tabular)
    assert out_tab.shape == (batch_size, 256), f"Got: {out_tab.shape}"
    print(f"Γ£ô ClinicalEncoder (tabular only): {out_tab.shape}")
    
    print("\nClinical encoder tests passed!")


if __name__ == '__main__':
    test_clinical_encoder()
