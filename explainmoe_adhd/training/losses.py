"""
Loss Functions for ExplainMoE-ADHD v2.13.

This module implements all loss functions:
- Diagnosis loss: BCEWithLogitsLoss
- Subtype loss: CrossEntropyLoss (masked)
- Severity loss: MSELoss (masked)
- MMD loss: Class-conditional MMD with multibandwidth RBF kernel
- Load balancing loss: CV of expert utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


# =============================================================================
# DIAGNOSIS LOSS
# =============================================================================

class DiagnosisLoss(nn.Module):
    """Binary cross-entropy loss for ADHD diagnosis."""
    def __init__(self, pos_weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits.squeeze(-1), targets.squeeze(-1))


# =============================================================================
# SUBTYPE LOSS
# =============================================================================

class SubtypeLoss(nn.Module):
    """Cross-entropy loss for ADHD subtype classification."""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction=reduction)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return self.loss_fn(logits, targets)
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return logits.sum() * 0.0
        return self.loss_fn(logits[valid_indices], targets[valid_indices])


# =============================================================================
# SEVERITY LOSS
# =============================================================================

class SeverityLoss(nn.Module):
    """MSE loss for ADHD severity regression."""
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            return self.loss_fn(predictions, targets)
        valid_indices = mask.nonzero(as_tuple=True)[0]
        if valid_indices.numel() == 0:
            return predictions.sum() * 0.0
        return self.loss_fn(predictions[valid_indices], targets[valid_indices])


# =============================================================================
# MMD LOSS
# =============================================================================

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Compute Gaussian (RBF) kernel."""
    x_exp = x.unsqueeze(1)
    y_exp = y.unsqueeze(0)
    dist_sq = ((x_exp - y_exp) ** 2).sum(-1)
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def compute_mmd(x: torch.Tensor, y: torch.Tensor, kernel: str = 'gaussian',
                 sigma: Optional[float] = None) -> torch.Tensor:
    """Compute unbiased squared MMD."""
    n, m = x.size(0), y.size(0)
    
    if sigma is None:
        combined = torch.cat([x, y], dim=0)
        dist_matrix = torch.cdist(combined, combined)
        triu_indices = torch.triu_indices(n + m, n + m, offset=1)
        sigma = dist_matrix[triu_indices[0], triu_indices[1]].median().item()
    
    k_xx = gaussian_kernel(x, x, sigma)
    k_yy = gaussian_kernel(y, y, sigma)
    k_xy = gaussian_kernel(x, y, sigma)
    
    e_xx = (k_xx.sum() - k_xx.diagonal().sum()) / (n * (n - 1))
    e_yy = (k_yy.sum() - k_yy.diagonal().sum()) / (m * (m - 1))
    e_xy = k_xy.sum() / (n * m)
    
    return e_xx + e_yy - 2 * e_xy


def compute_multibandwidth_mmd(x: torch.Tensor, y: torch.Tensor,
                                 bandwidths: Optional[List[float]] = None) -> torch.Tensor:
    """Compute MMD with multiple bandwidths."""
    if bandwidths is None:
        combined = torch.cat([x, y], dim=0)
        dist_matrix = torch.cdist(combined, combined)
        triu_indices = torch.triu_indices(combined.size(0), combined.size(0), offset=1)
        sigma = dist_matrix[triu_indices[0], triu_indices[1]].median().item()
        bandwidths = [0.5 * sigma, sigma, 2.0 * sigma]
    
    return sum(compute_mmd(x, y, kernel='gaussian', sigma=bw) for bw in bandwidths)


class MMDLoss(nn.Module):
    """Class-conditional MMD Loss for cross-modal latent space alignment.
    
    Computes MMD between two MODALITIES, separately per class:
        total = MMD(modA_adhd, modB_adhd) + MMD(modA_control, modB_control)
    
    This aligns ADHD representations across modalities with each other,
    and control representations across modalities with each other.
    """
    def __init__(self, kernel: str = 'multibandwidth_rbf', min_batch_per_class: int = 16):
        super().__init__()
        self.kernel = kernel
        self.min_batch_per_class = min_batch_per_class
    
    def _compute_pairwise_mmd(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute MMD between two sets of representations."""
        if self.kernel == 'multibandwidth_rbf':
            return compute_multibandwidth_mmd(z_a, z_b)
        else:
            return compute_mmd(z_a, z_b, kernel=self.kernel)
    
    def forward(
        self,
        z_modA_adhd: torch.Tensor,
        z_modB_adhd: torch.Tensor,
        z_modA_control: torch.Tensor,
        z_modB_control: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class-conditional MMD between two modalities.
        
        Args:
            z_modA_adhd: (n, dim) - ADHD representations from modality A
            z_modB_adhd: (m, dim) - ADHD representations from modality B
            z_modA_control: (n', dim) - Control representations from modality A
            z_modB_control: (m', dim) - Control representations from modality B
        
        Returns:
            Scalar MMD loss (sum of per-class MMD)
        """
        loss = torch.tensor(0.0, device=z_modA_adhd.device)
        
        # ADHD class alignment
        if z_modA_adhd.size(0) >= self.min_batch_per_class and z_modB_adhd.size(0) >= self.min_batch_per_class:
            loss = loss + self._compute_pairwise_mmd(z_modA_adhd, z_modB_adhd)
        
        # Control class alignment
        if z_modA_control.size(0) >= self.min_batch_per_class and z_modB_control.size(0) >= self.min_batch_per_class:
            loss = loss + self._compute_pairwise_mmd(z_modA_control, z_modB_control)
        
        return loss


# =============================================================================
# LOAD BALANCING LOSS
# =============================================================================

def compute_load_balance_loss(expert_weights: torch.Tensor, balance_coeff: float = 0.01) -> torch.Tensor:
    """Compute load balancing loss using coefficient of variation."""
    mean = expert_weights.mean()
    std = expert_weights.std()
    cv = std / (mean + 1e-8)
    return balance_coeff * cv


# =============================================================================
# COMBINED LOSS
# =============================================================================

class CombinedLoss(nn.Module):
    """Combined loss for Phases 4 and 5."""
    def __init__(self, diagnosis_weight: float = 1.0, subtype_weight: float = 0.0,
                 severity_weight: float = 0.0, mmd_weight: float = 0.0,
                 load_balance_weight: float = 0.01, use_subtype: bool = False,
                 use_severity: bool = False, use_mmd: bool = False):
        super().__init__()
        self.diagnosis_weight = diagnosis_weight
        self.subtype_weight = subtype_weight
        self.severity_weight = severity_weight
        self.mmd_weight = mmd_weight
        self.load_balance_weight = load_balance_weight
        self.use_subtype = use_subtype
        self.use_severity = use_severity
        self.use_mmd = use_mmd
        
        self.diagnosis_loss = DiagnosisLoss()
        if use_subtype:
            self.subtype_loss = SubtypeLoss()
        if use_severity:
            self.severity_loss = SeverityLoss()
        if use_mmd:
            self.mmd_loss = MMDLoss()
    
    def forward(self, diagnosis_logits: torch.Tensor, diagnosis_targets: torch.Tensor,
                subtype_logits: Optional[torch.Tensor] = None,
                subtype_targets: Optional[torch.Tensor] = None,
                subtype_mask: Optional[torch.Tensor] = None,
                severity_preds: Optional[torch.Tensor] = None,
                severity_targets: Optional[torch.Tensor] = None,
                severity_mask: Optional[torch.Tensor] = None,
                mmd_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
                expert_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        
        total_loss = 0.0
        loss_dict = {}
        
        # Diagnosis loss
        diag_loss = self.diagnosis_loss(diagnosis_logits, diagnosis_targets)
        total_loss += self.diagnosis_weight * diag_loss
        loss_dict['diagnosis'] = diag_loss.item()
        
        # Subtype loss
        if self.use_subtype and subtype_logits is not None:
            sub_loss = self.subtype_loss(subtype_logits, subtype_targets, subtype_mask)
            total_loss += self.subtype_weight * sub_loss
            loss_dict['subtype'] = sub_loss.item()
        
        # Severity loss
        if self.use_severity and severity_preds is not None:
            sev_loss = self.severity_loss(severity_preds, severity_targets, severity_mask)
            total_loss += self.severity_weight * sev_loss
            loss_dict['severity'] = sev_loss.item()
        
        # MMD loss (class-conditional: each pair is 4 tensors)
        if self.use_mmd and mmd_pairs is not None:
            mmd_total = sum(
                self.mmd_loss(z_modA_adhd, z_modB_adhd, z_modA_ctrl, z_modB_ctrl)
                for z_modA_adhd, z_modB_adhd, z_modA_ctrl, z_modB_ctrl in mmd_pairs
            ) / len(mmd_pairs)
            total_loss += self.mmd_weight * mmd_total
            loss_dict['mmd'] = mmd_total.item()
        
        # Load balancing loss
        if expert_weights is not None:
            lb_loss = compute_load_balance_loss(expert_weights)
            total_loss += self.load_balance_weight * lb_loss
            loss_dict['load_balance'] = lb_loss.item()
        
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# =============================================================================
# TEST
# =============================================================================

def test_losses():
    device = torch.device('cpu')
    batch_size = 32
    
    # DiagnosisLoss
    diag_loss_fn = DiagnosisLoss()
    logits = torch.randn(batch_size, 1).to(device)
    targets = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    print(f"Γ£ô DiagnosisLoss: {diag_loss_fn(logits, targets).item():.4f}")
    
    # SubtypeLoss
    sub_loss_fn = SubtypeLoss()
    sub_logits = torch.randn(batch_size, 3).to(device)
    sub_targets = torch.randint(0, 3, (batch_size,)).to(device)
    mask = torch.rand(batch_size) > 0.5
    print(f"Γ£ô SubtypeLoss: {sub_loss_fn(sub_logits, sub_targets, mask).item():.4f}")
    
    # SeverityLoss
    sev_loss_fn = SeverityLoss()
    sev_preds = torch.randn(batch_size, 2).to(device)
    sev_targets = torch.randn(batch_size, 2).to(device)
    print(f"Γ£ô SeverityLoss: {sev_loss_fn(sev_preds, sev_targets, mask).item():.4f}")
    
    # MMDLoss (class-conditional: modA_adhd, modB_adhd, modA_control, modB_control)
    mmd_loss_fn = MMDLoss()
    z_modA_adhd = torch.randn(16, 256).to(device)
    z_modB_adhd = torch.randn(16, 256).to(device)
    z_modA_ctrl = torch.randn(16, 256).to(device)
    z_modB_ctrl = torch.randn(16, 256).to(device)
    print(f"Γ£ô MMDLoss: {mmd_loss_fn(z_modA_adhd, z_modB_adhd, z_modA_ctrl, z_modB_ctrl).item():.4f}")
    
    # CombinedLoss (Phase 4)
    combined_ph4 = CombinedLoss(diagnosis_weight=1.0, subtype_weight=0.25, 
                                load_balance_weight=0.01, use_subtype=True)
    total, ld = combined_ph4(logits, targets, sub_logits, sub_targets, mask, 
                            expert_weights=torch.ones(4) / 4)
    print(f"Γ£ô CombinedLoss Phase4: {total.item():.4f}")
    
    # CombinedLoss (Phase 5)
    combined_ph5 = CombinedLoss(diagnosis_weight=1.0, subtype_weight=0.25, 
                                severity_weight=0.1, mmd_weight=0.3,
                                load_balance_weight=0.01, use_subtype=True,
                                use_severity=True, use_mmd=True)
    mmd_pair = (z_modA_adhd, z_modB_adhd, z_modA_ctrl, z_modB_ctrl)
    total, ld = combined_ph5(logits, targets, sub_logits, sub_targets, mask,
                            sev_preds, sev_targets, mask, [mmd_pair],
                            torch.ones(4) / 4)
    print(f"Γ£ô CombinedLoss Phase5: {total.item():.4f}")
    
    print("\nAll loss tests passed!")


if __name__ == '__main__':
    test_losses()
