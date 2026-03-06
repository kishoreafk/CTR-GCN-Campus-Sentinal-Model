"""
Task Heads for ExplainMoE-ADHD v2.13.

This module implements the task heads (Section 5.6):
- DiagnosisHead: Primary output (binary ADHD vs Control)
- SubtypeHead: Secondary output (3-class: Combined, HI, Inattentive)
- SeverityHead: Tertiary output (2 regression: inattentive, hyperactive scores)

Loss functions:
- Diagnosis: BCEWithLogitsLoss, weight=1.0
- Subtype: CrossEntropyLoss, weight=0.25, applies to ADHD-200 DXΓêê{1,2,3} only
- Severity: MSELoss, weight=0.1, applies to ADHD-200 scored subjects only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DiagnosisHead(nn.Module):
    """
    Primary task head: Binary classification (ADHD vs Control).
    
    Active in Phases 2, 4, 5.
    Phase 2 version is discarded before Phase 4.
    Phase 4 initializes a FRESH diagnosis head.
    
    Input: (batch, dim) - FuseMoE output y_moe
    Output: (batch, 1) - logit (apply sigmoid for probability)
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, y_moe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_moe: (batch, input_dim) - FuseMoE output
        
        Returns:
            (batch, 1) - logit
        """
        return self.net(y_moe)
    
    def predict_proba(self, y_moe: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(y_moe)
        return torch.sigmoid(logits)


class SubtypeHead(nn.Module):
    """
    Secondary task head: 3-class classification (ADHD subtypes).
    
    Active in Phases 4, 5.
    Applies ONLY to ADHD-200 subjects with DX Γêê {1, 2, 3}.
    Masked (loss=0) for non-ADHD-200 subjects AND ADHD-200 controls.
    
    Classes: 0=Combined, 1=Hyperactive/Impulsive, 2=Inattentive
    
    Input: (batch, dim) - FuseMoE output y_moe
    Output: (batch, 3) - logits (apply softmax for probabilities)
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(
        self,
        y_moe: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_moe: (batch, input_dim) - FuseMoE output
            mask: (batch,) - boolean mask for valid samples (optional)
        
        Returns:
            (batch, num_classes) - logits
        """
        logits = self.net(y_moe)
        
        # Apply mask if provided
        if mask is not None:
            logits = logits * mask.unsqueeze(-1)
        
        return logits
    
    def predict_proba(self, y_moe: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(y_moe)
        return F.softmax(logits, dim=-1)


class SeverityHead(nn.Module):
    """
    Tertiary task head: Regression (ADHD severity scores).
    
    Active in Phase 5 ONLY.
    INACTIVE in Phase 4 (weights exist but receive zero gradient).
    Predicts inattentive_score and hyperactive_score.
    
    Input: (batch, dim) - FuseMoE output y_moe
    Output: (batch, 2) - [inattentive, hyperactive]
    """
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.5,  # Higher dropout for regression
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [inattentive, hyperactive]
        )
    
    def forward(
        self,
        y_moe: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_moe: (batch, input_dim) - FuseMoE output
            mask: (batch,) - boolean mask for valid samples (optional)
        
        Returns:
            (batch, 2) - [inattentive, hyperactive] scores
        """
        scores = self.net(y_moe)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores * mask.unsqueeze(-1)
        
        return scores


class TaskHeads(nn.Module):
    """
    Combined task heads container.
    
    Contains all three task heads for convenience.
    """
    def __init__(
        self,
        input_dim: int = 256,
        diagnosis_hidden: int = 128,
        subtype_hidden: int = 128,
        severity_hidden: int = 128,
        diagnosis_dropout: float = 0.3,
        subtype_dropout: float = 0.3,
        severity_dropout: float = 0.5,
    ):
        super().__init__()
        
        self.diagnosis = DiagnosisHead(
            input_dim=input_dim,
            hidden_dim=diagnosis_hidden,
            dropout=diagnosis_dropout,
        )
        
        self.subtype = SubtypeHead(
            input_dim=input_dim,
            hidden_dim=subtype_hidden,
            dropout=subtype_dropout,
        )
        
        self.severity = SeverityHead(
            input_dim=input_dim,
            hidden_dim=severity_hidden,
            dropout=severity_dropout,
        )
    
    def forward(
        self,
        y_moe: torch.Tensor,
        subtype_mask: Optional[torch.Tensor] = None,
        severity_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all task heads.
        
        Args:
            y_moe: (batch, input_dim) - FuseMoE output
            subtype_mask: (batch,) - boolean mask for subtype valid samples
            severity_mask: (batch,) - boolean mask for severity valid samples
        
        Returns:
            diagnosis_logits: (batch, 1)
            subtype_logits: (batch, 3)
            severity_scores: (batch, 2)
        """
        diagnosis_logits = self.diagnosis(y_moe)
        subtype_logits = self.subtype(y_moe, subtype_mask)
        severity_scores = self.severity(y_moe, severity_mask)
        
        return diagnosis_logits, subtype_logits, severity_scores


class TaskHeadFactory:
    """Factory for creating task heads."""
    
    @classmethod
    def create_diagnosis(
        cls,
        input_dim: int = 256,
        **kwargs
    ) -> DiagnosisHead:
        return DiagnosisHead(input_dim=input_dim, **kwargs)
    
    @classmethod
    def create_subtype(
        cls,
        input_dim: int = 256,
        **kwargs
    ) -> SubtypeHead:
        return SubtypeHead(input_dim=input_dim, **kwargs)
    
    @classmethod
    def create_severity(
        cls,
        input_dim: int = 256,
        **kwargs
    ) -> SeverityHead:
        return SeverityHead(input_dim=input_dim, **kwargs)
    
    @classmethod
    def create_all(
        cls,
        input_dim: int = 256,
        **kwargs
    ) -> TaskHeads:
        return TaskHeads(input_dim=input_dim, **kwargs)


# =============================================================================
# ABLATION VARIANTS (Section 10, A16)
# =============================================================================

# Subtype loss weight variants are handled in the training loop, not here


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_task_heads():
    """Test task heads."""
    batch_size = 4
    device = torch.device('cpu')
    
    y_moe = torch.randn(batch_size, 256).to(device)
    
    # Test DiagnosisHead
    diagnosis = DiagnosisHead(input_dim=256).to(device)
    diag_logits = diagnosis(y_moe)
    assert diag_logits.shape == (batch_size, 1), f"Diagnosis: {diag_logits.shape}"
    print(f"Γ£ô DiagnosisHead: {diag_logits.shape}")
    
    # Test SubtypeHead
    subtype = SubtypeHead(input_dim=256).to(device)
    sub_logits = subtype(y_moe)
    assert sub_logits.shape == (batch_size, 3), f"Subtype: {sub_logits.shape}"
    print(f"Γ£ô SubtypeHead: {sub_logits.shape}")
    
    # Test with mask
    mask = torch.tensor([True, True, False, True]).to(device)
    sub_masked = subtype(y_moe, mask=mask)
    assert sub_masked.shape == (batch_size, 3)
    print(f"Γ£ô SubtypeHead (masked): {sub_masked.shape}")
    
    # Test SeverityHead
    severity = SeverityHead(input_dim=256).to(device)
    sev_scores = severity(y_moe)
    assert sev_scores.shape == (batch_size, 2), f"Severity: {sev_scores.shape}"
    print(f"Γ£ô SeverityHead: {sev_scores.shape}")
    
    # Test combined TaskHeads
    heads = TaskHeads(input_dim=256).to(device)
    diag, sub, sev = heads(y_moe, subtype_mask=mask, severity_mask=mask)
    assert diag.shape == (batch_size, 1)
    assert sub.shape == (batch_size, 3)
    assert sev.shape == (batch_size, 2)
    print(f"Γ£ô TaskHeads: diag {diag.shape}, sub {sub.shape}, sev {sev.shape}")
    
    print("\nTask head tests passed!")


if __name__ == '__main__':
    test_task_heads()
