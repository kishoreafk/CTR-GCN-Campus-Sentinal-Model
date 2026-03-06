"""
FuseMoE Module for ExplainMoE-ADHD v2.13.

This module implements the FuseMoE (Fused Mixture of Experts) architecture (Section 5.5):
- 4 shared experts
- Per-modality routers with Laplace kernel gating
- Per-router learnable temperature
- K-means centroid initialization
- Residual connection with LayerNorm
- Load balancing loss

Key features:
- Top-K=2 expert selection (50% of expert pool)
- Laplace kernel gating with adaptive temperature
- K-means centroid initialization from Phase 3 outputs
- Residual connection: y_moe = LayerNorm(z + expert_output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import numpy as np
from sklearn.cluster import KMeans


class Expert(nn.Module):
    """
    Single expert network.
    
    Architecture: FFN with GELU activation
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 256,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FuseMoE(nn.Module):
    """
    FuseMoE: Fused Mixture of Experts.
    
    Key parameters:
    - num_experts: 4 (shared across all modalities)
    - top_k: 2 (select top 2 experts per input)
    - num_routers: 5 (one per modality pathway)
    - gating: Laplace kernel with per-router temperature
    - residual: Yes (z + expert_output ΓåÆ LayerNorm)
    
    Input: (batch, dim) - projection head output z_m
    Output: (batch, dim) - FuseMoE output y_moe
    """
    def __init__(
        self,
        input_dim: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        num_routers: int = 5,
        expert_hidden_dim: int = 256,
        init_temperature: float = 1.0,
        use_residual: bool = True,
        balance_coeff: float = 0.01,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_routers = num_routers
        self.use_residual = use_residual
        self.balance_coeff = balance_coeff
        
        # Shared experts
        self.experts = nn.ModuleList([
            Expert(
                input_dim=input_dim,
                hidden_dim=expert_hidden_dim,
                output_dim=input_dim,
            )
            for _ in range(num_experts)
        ])
        
        # Per-router centroids (initialized randomly, will be overwritten by K-means)
        self.centroids = nn.ParameterList([
            nn.Parameter(torch.randn(num_experts, input_dim) * 0.5)
            for _ in range(num_routers)
        ])
        
        # Per-router learnable temperature (constrained positive via softplus)
        self.tau_raw = nn.ParameterList([
            nn.Parameter(torch.ones(1) * np.log(np.exp(init_temperature) - 1))
            for _ in range(num_routers)
        ])
        
        # Post-residual LayerNorm
        self.layer_norm = nn.LayerNorm(input_dim)
    
    def get_temperature(self, router_idx: int) -> torch.Tensor:
        """
        Get constrained positive temperature for a router.
        
        Uses softplus to ensure tau > 0.
        """
        return F.softplus(self.tau_raw[router_idx]) + 1e-6
    
    def route(
        self,
        z: torch.Tensor,
        router_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing weights using Laplace kernel gating.
        
        Args:
            z: (batch, dim) - projection head output
            router_idx: int - which modality router to use
        
        Returns:
            weights: (batch, top_k) - softmax weights for selected experts
            indices: (batch, top_k) - indices of selected experts
            all_scores: (batch, num_experts) - raw scores for all experts
        """
        batch_size = z.size(0)
        device = z.device
        
        # Get centroids and temperature for this router
        centroids = self.centroids[router_idx]  # (num_experts, dim)
        tau = self.get_temperature(router_idx)  # scalar
        
        # Compute distances: (batch, num_experts)
        # Using cdist for efficient computation
        z_expanded = z.unsqueeze(1)  # (batch, 1, dim)
        centroids_expanded = centroids.unsqueeze(0)  # (1, num_experts, dim)
        distances = torch.cdist(z_expanded, centroids_expanded).squeeze(1)  # (batch, num_experts)
        
        # Laplace kernel scores: exp(-distance / tau)
        scores = torch.exp(-distances / tau)  # (batch, num_experts)
        
        # Top-K selection
        top_k_scores, top_k_indices = torch.topk(scores, self.top_k, dim=-1)  # (batch, top_k)
        
        # Softmax over selected experts
        weights = F.softmax(top_k_scores, dim=-1)  # (batch, top_k)
        
        return weights, top_k_indices, scores
    
    def forward(
        self,
        z: torch.Tensor,
        router_idx: int,
        return_expert_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through FuseMoE.
        
        Args:
            z: (batch, dim) - projection head output
            router_idx: int - which modality router to use
            return_expert_weights: whether to return expert utilization weights
        
        Returns:
            y_moe: (batch, dim) - FuseMoE output
            expert_weights: (num_experts,) - expert utilization (for load balancing)
        """
        batch_size = z.size(0)
        
        # Route to experts
        weights, indices, all_scores = self.route(z, router_idx)
        
        # Compute expert outputs
        expert_outputs = torch.zeros_like(z)  # (batch, dim)
        
        # Process each expert
        for k in range(self.top_k):
            expert_idx = indices[:, k]  # (batch,)
            w = weights[:, k].unsqueeze(-1)  # (batch, 1)
            
            # Gather expert outputs per sample
            for e_idx in range(self.num_experts):
                mask = (expert_idx == e_idx)
                if mask.any():
                    expert_outputs[mask] += w[mask] * self.experts[e_idx](z[mask])
        
        # Residual connection
        if self.use_residual:
            y_moe = self.layer_norm(z + expert_outputs)
        else:
            y_moe = expert_outputs
        
        # Compute expert utilization for load balancing
        if return_expert_weights or self.training:
            expert_counts = torch.zeros(self.num_experts, device=z.device)
            for k in range(self.top_k):
                for e_idx in range(self.num_experts):
                    expert_counts[e_idx] += (indices[:, k] == e_idx).float().sum()
            
            # Normalize by batch size * top_k
            expert_weights = expert_counts / (batch_size * self.top_k)
        else:
            expert_weights = None
        
        return y_moe, expert_weights
    
    def load_balance_loss(
        self,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute load balancing loss.
        
        Uses coefficient of variation (CV) of expert utilization.
        """
        mean = expert_weights.mean()
        std = expert_weights.std()
        cv = std / (mean + 1e-8)
        return self.balance_coeff * cv
    
    def initialize_centroids_kmeans(
        self,
        z_all: torch.Tensor,
        random_state: int = 42,
    ):
        """
        Initialize expert centroids via K-means on Phase 3 outputs.
        
        Called once after Phase 3 completes, before Phase 4 starts.
        
        Args:
            z_all: (N, dim) - all training subjects' projection outputs, all modalities
        """
        # Convert to numpy for sklearn
        z_np = z_all.detach().cpu().numpy()
        
        # K-means clustering
        kmeans = KMeans(
            n_clusters=self.num_experts,
            n_init=10,
            random_state=random_state,
        )
        kmeans.fit(z_np)
        centers = torch.tensor(
            kmeans.cluster_centers_,
            dtype=torch.float32,
            device=z_all.device,
        )
        
        # Copy to all routers
        for router_centroids in self.centroids:
            router_centroids.data.copy_(centers)
    
    def get_router_info(self, router_idx: int) -> dict:
        """Get information about a router."""
        return {
            'centroids': self.centroids[router_idx].data,
            'temperature': self.get_temperature(router_idx),
            'num_experts': self.num_experts,
            'top_k': self.top_k,
        }


class FuseMoEFactory:
    """Factory for creating FuseMoE modules."""
    
    @classmethod
    def create(
        cls,
        input_dim: int = 256,
        num_experts: int = 4,
        top_k: int = 2,
        num_routers: int = 5,
        **kwargs
    ) -> FuseMoE:
        return FuseMoE(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            num_routers=num_routers,
            **kwargs
        )


# =============================================================================
# ABLATION VARIANTS
# =============================================================================

class FuseMoEWithTopK1(FuseMoE):
    """Ablation A4: Top-K=1 (transfer-negative control)."""
    def __init__(self, **kwargs):
        kwargs['top_k'] = 1
        super().__init__(**kwargs)


class FuseMoEWithTopK3(FuseMoE):
    """Ablation A4: Top-K=3 (dense routing)."""
    def __init__(self, **kwargs):
        kwargs['top_k'] = 3
        super().__init__(**kwargs)


class FuseMoEWithoutResidual(FuseMoE):
    """Ablation A12: Without residual connection."""
    def __init__(self, **kwargs):
        kwargs['use_residual'] = False
        super().__init__(**kwargs)


class FuseMoEWithSharedTemperature(FuseMoE):
    """Ablation A14: Shared temperature across routers."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override with single temperature
        self.tau_raw = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.0)
        ])
    
    def get_temperature(self, router_idx: int) -> torch.Tensor:
        return F.softplus(self.tau_raw[0]) + 1e-6


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_fusemoe():
    """Test FuseMoE module."""
    batch_size = 4
    device = torch.device('cpu')
    
    # Test FuseMoE
    fusemoe = FuseMoE(
        input_dim=256,
        num_experts=4,
        top_k=2,
        num_routers=5,
    ).to(device)
    
    z = torch.randn(batch_size, 256).to(device)
    
    # Test each router
    for router_idx in range(5):
        y_moe, expert_weights = fusemoe(z, router_idx, return_expert_weights=True)
        assert y_moe.shape == (batch_size, 256), f"Router {router_idx}: {y_moe.shape}"
        
        # Test load balancing loss
        lb_loss = fusemoe.load_balance_loss(expert_weights)
        assert lb_loss.item() >= 0, f"Router {router_idx}: lb_loss = {lb_loss.item()}"
    
    print(f"Γ£ô FuseMoE (top_k=2): output {y_moe.shape}, lb_loss = {lb_loss.item():.4f}")
    
    # Test K-means initialization
    z_all = torch.randn(100, 256)
    fusemoe.initialize_centroids_kmeans(z_all)
    print("Γ£ô FuseMoE.initialize_centroids_kmeans")
    
    # Test ablation variants
    fusemoe_k1 = FuseMoEWithTopK1(input_dim=256).to(device)
    fusemoe_k3 = FuseMoEWithTopK3(input_dim=256).to(device)
    fusemoe_no_res = FuseMoEWithoutResidual(input_dim=256).to(device)
    
    y_k1, _ = fusemoe_k1(z, 0)
    y_k3, _ = fusemoe_k3(z, 0)
    y_no_res, _ = fusemoe_no_res(z, 0)
    
    assert y_k1.shape == (batch_size, 256)
    assert y_k3.shape == (batch_size, 256)
    assert y_no_res.shape == (batch_size, 256)
    
    print(f"Γ£ô FuseMoEWithTopK1: {y_k1.shape}")
    print(f"Γ£ô FuseMoEWithTopK3: {y_k3.shape}")
    print(f"Γ£ô FuseMoEWithoutResidual: {y_no_res.shape}")
    
    print("\nFuseMoE tests passed!")


if __name__ == '__main__':
    test_fusemoe()
