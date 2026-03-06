"""
Confound Probes for ExplainMoE-ADHD v2.13.

This module implements the four mandatory confound probes (Section 9.1):
1. Hardware probe: z_eeg ΓåÆ hw_type (classification)
2. Missingness probe: modality_availability_mask ΓåÆ ADHD_label (classification)
3. Motion probe: z_clin ΓåÆ mean_FD (regression)
4. Site probe: z_clin ΓåÆ site_id (classification)

Each probe uses sklearn classifiers/regressors with proper CV.
"""

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Result from a confound probe."""
    metric: str
    value: float
    ci_lower: float
    ci_upper: float
    is_confound: bool  # True if metric > threshold


class ConfoundProbe:
    """
    Base class for confound probes.
    """
    
    def __init__(
        self,
        name: str,
        threshold: float = 0.05,
        random_state: int = 42,
    ):
        self.name = name
        self.threshold = threshold
        self.random_state = random_state
    
    def fit_predict(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_outer_folds: int = 5,
    ) -> Tuple[float, float, float]:
        """
        Fit probe with inner CV and evaluate on outer folds.
        
        Returns:
            (mean_metric, ci_lower, ci_upper)
        """
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
        
        # Use logistic regression with inner CV
        model = LogisticRegressionCV(
            Cs=[0.001, 0.01, 0.1, 1.0, 10.0],
            cv=3,
            solver='lbfgs',
            max_iter=1000,
            scoring='roc_auc',
        )
        
        # Outer CV for evaluation
        cv = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Bootstrap CI
        ci_lower = max(0.5, mean_score - 1.96 * std_score)
        ci_upper = min(1.0, mean_score + 1.96 * std_score)
        
        return mean_score, ci_lower, ci_upper
    
    def is_confound(self, metric: float) -> bool:
        """Check if metric indicates a confound."""
        return metric > (0.5 + self.threshold)


class HardwareProbe(ConfoundProbe):
    """
    Probe 1: Hardware type detection.
    
    Tests if latent space encodes hardware identity rather than brain signal.
    
    Input: z_eeg ΓåÆ hw_type
    Red Flag: Near-perfect accuracy (AUROC > 0.95)
    """
    
    def __init__(self):
        super().__init__(name="hardware_probe", threshold=0.05)
    
    def evaluate(
        self,
        z_eeg: np.ndarray,
        hw_type: np.ndarray,
    ) -> ProbeResult:
        """
        Evaluate hardware confound.
        
        Args:
            z_eeg: (N, 256) - EEG latent representations
            hw_type: (N,) - hardware type labels
        
        Returns:
            ProbeResult
        """
        # Standardize
        scaler = StandardScaler()
        z_scaled = scaler.fit_transform(z_eeg)
        
        # Fit probe
        metric, ci_lower, ci_upper = self.fit_predict(z_scaled, hw_type)
        
        # Check confound
        is_confound = self.is_confound(metric)
        
        return ProbeResult(
            metric="hardware_auroc",
            value=metric,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_confound=is_confound,
        )


class MissingnessProbe(ConfoundProbe):
    """
    Probe 2: Missing modality detection.
    
    Tests if which modalities a subject has predicts ADHD.
    Could indicate ascertainment bias.
    
    Input: modality_availability_mask ΓåÆ ADHD_label
    Red Flag: AUROC > 0.55 (chance + 5%)
    """
    
    def __init__(self):
        super().__init__(name="missingness_probe", threshold=0.05)
    
    def evaluate(
        self,
        modality_mask: np.ndarray,
        adhd_label: np.ndarray,
    ) -> ProbeResult:
        """
        Evaluate missingness confound.
        
        Args:
            modality_mask: (N, 5) - binary mask for 5 modalities
            adhd_label: (N,) - ADHD labels
        
        Returns:
            ProbeResult
        """
        # Fit probe
        metric, ci_lower, ci_upper = self.fit_predict(modality_mask, adhd_label)
        
        # Check confound
        is_confound = self.is_confound(metric)
        
        return ProbeResult(
            metric="missingness_auroc",
            value=metric,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_confound=is_confound,
        )


class MotionProbe:
    """
    Probe 3: Head motion detection.
    
    Tests if clinical representations encode head motion.
    
    Input: z_clin ΓåÆ mean_FD
    Red Flag: R┬▓ > 0.3
    """
    
    def __init__(self):
        self.name = "motion_probe"
        self.threshold = 0.3
        self.random_state = 42
    
    def evaluate(
        self,
        z_clin: np.ndarray,
        mean_fd: np.ndarray,
    ) -> ProbeResult:
        """
        Evaluate motion confound.
        
        Args:
            z_clin: (N, 256) - clinical latent representations
            mean_fd: (N,) - mean framewise displacement
        
        Returns:
            ProbeResult
        """
        from sklearn.model_selection import KFold, cross_val_score
        
        # Standardize
        scaler = StandardScaler()
        z_scaled = scaler.fit_transform(z_clin)
        
        # Ridge regression
        model = RidgeCV(
            alphas=[0.01, 0.1, 1.0, 10.0, 100.0],
            cv=3,
        )
        
        # Outer CV (use KFold for regression, no stratification needed)
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        r2_scores = cross_val_score(model, z_scaled, mean_fd, cv=cv, scoring='r2')
        
        mean_r2 = r2_scores.mean()
        std_r2 = r2_scores.std()
        
        ci_lower = max(0, mean_r2 - 1.96 * std_r2)
        ci_upper = min(1, mean_r2 + 1.96 * std_r2)
        
        is_confound = mean_r2 > self.threshold
        
        return ProbeResult(
            metric="motion_r2",
            value=mean_r2,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_confound=is_confound,
        )


class SiteProbe(ConfoundProbe):
    """
    Probe 4: Site detection.
    
    Tests if representations are dominated by site rather than ADHD.
    
    Input: z_clin ΓåÆ site_id
    Red Flag: Near-perfect accuracy (AUROC > 0.95)
    """
    
    def __init__(self):
        super().__init__(name="site_probe", threshold=0.05)
    
    def evaluate(
        self,
        z_clin: np.ndarray,
        site_id: np.ndarray,
    ) -> ProbeResult:
        """
        Evaluate site confound.
        
        Args:
            z_clin: (N, 256) - clinical latent representations
            site_id: (N,) - site labels
        
        Returns:
            ProbeResult
        """
        # Standardize
        scaler = StandardScaler()
        z_scaled = scaler.fit_transform(z_clin)
        
        # Reduce dimensions if needed (for many sites)
        if z_scaled.shape[1] > 100:
            pca = PCA(n_components=min(100, z_scaled.shape[0] - 1))
            z_scaled = pca.fit_transform(z_scaled)
        
        # Fit probe
        metric, ci_lower, ci_upper = self.fit_predict(z_scaled, site_id)
        
        # Check confound
        is_confound = self.is_confound(metric)
        
        return ProbeResult(
            metric="site_auroc",
            value=metric,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            is_confound=is_confound,
        )


class ConfoundProbeSuite:
    """
    Suite of all four confound probes.
    """
    
    def __init__(self):
        self.probes = {
            'hardware': HardwareProbe(),
            'missingness': MissingnessProbe(),
            'motion': MotionProbe(),
            'site': SiteProbe(),
        }
    
    def evaluate_all(
        self,
        z_eeg: Optional[np.ndarray] = None,
        hw_type: Optional[np.ndarray] = None,
        modality_mask: Optional[np.ndarray] = None,
        adhd_label: Optional[np.ndarray] = None,
        z_clin: Optional[np.ndarray] = None,
        mean_fd: Optional[np.ndarray] = None,
        site_id: Optional[np.ndarray] = None,
    ) -> Dict[str, ProbeResult]:
        """
        Run all applicable probes.
        
        Args:
            z_eeg: EEG latent representations
            hw_type: Hardware type labels
            modality_mask: Modality availability mask
            adhd_label: ADHD labels
            z_clin: Clinical latent representations
            mean_fd: Mean framewise displacement
            site_id: Site labels
        
        Returns:
            Dict of probe results
        """
        results = {}
        
        # Hardware probe
        if z_eeg is not None and hw_type is not None:
            results['hardware'] = self.probes['hardware'].evaluate(z_eeg, hw_type)
        
        # Missingness probe
        if modality_mask is not None and adhd_label is not None:
            results['missingness'] = self.probes['missingness'].evaluate(
                modality_mask, adhd_label
            )
        
        # Motion probe
        if z_clin is not None and mean_fd is not None:
            results['motion'] = self.probes['motion'].evaluate(z_clin, mean_fd)
        
        # Site probe
        if z_clin is not None and site_id is not None:
            results['site'] = self.probes['site'].evaluate(z_clin, site_id)
        
        return results
    
    def print_report(self, results: Dict[str, ProbeResult]):
        """Print a report of all probe results."""
        print("\n" + "=" * 60)
        print("CONFOUND PROBE RESULTS")
        print("=" * 60)
        
        for name, result in results.items():
            status = "ΓÜá∩╕Å CONFOUND" if result.is_confound else "Γ£ô OK"
            print(f"\n{name.upper()}")
            print(f"  Metric: {result.metric}")
            print(f"  Value: {result.value:.3f} (95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}])")
            print(f"  Status: {status}")
        
        print("\n" + "=" * 60)


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Compute AUROC."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_score)


def compute_auroc_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
) -> Tuple[float, float, float]:
    """
    Compute AUROC with bootstrap confidence intervals.
    """
    from sklearn.metrics import roc_auc_score
    
    # Point estimate
    auroc = roc_auc_score(y_true, y_score)
    
    # Bootstrap
    n = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        try:
            score = roc_auc_score(y_true[indices], y_score[indices])
            bootstrap_scores.append(score)
        except ValueError:
            pass
    
    bootstrap_scores = np.array(bootstrap_scores)
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    
    return auroc, ci_lower, ci_upper


def compute_per_modality_metrics(
    y_true: Dict[str, np.ndarray],
    y_score: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per modality.
    
    Returns:
        Dict[modality, Dict[metric, value]]
    """
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, 
        precision_score, recall_score, f1_score,
        confusion_matrix, roc_curve
    )
    
    results = {}
    
    for modality in y_true.keys():
        y_t = y_true[modality]
        y_s = y_score[modality]
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_t, y_s)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Binary predictions
        y_pred = (y_s >= optimal_threshold).astype(int)
        
        # Metrics
        results[modality] = {
            'auroc': roc_auc_score(y_t, y_s),
            'accuracy': accuracy_score(y_t, y_pred),
            'precision': precision_score(y_t, y_pred, zero_division=0),
            'recall': recall_score(y_t, y_pred, zero_division=0),
            'f1': f1_score(y_t, y_pred, zero_division=0),
            'optimal_threshold': optimal_threshold,
        }
    
    return results


# =============================================================================
# TEST
# =============================================================================

def test_confound_probes():
    """Test confound probes."""
    np.random.seed(42)
    
    # Create dummy data
    n = 100
    
    # Hardware probe
    z_eeg = np.random.randn(n, 256)
    hw_type = np.random.randint(0, 2, n)
    probe = HardwareProbe()
    result = probe.evaluate(z_eeg, hw_type)
    print(f"Hardware probe: {result.value:.3f} (confound={result.is_confound})")
    
    # Motion probe
    z_clin = np.random.randn(n, 256)
    mean_fd = np.random.randn(n)
    motion_probe = MotionProbe()
    result = motion_probe.evaluate(z_clin, mean_fd)
    print(f"Motion probe: {result.value:.3f} (confound={result.is_confound})")
    
    # Suite
    suite = ConfoundProbeSuite()
    results = suite.evaluate_all(z_eeg=z_eeg, hw_type=hw_type)
    suite.print_report(results)
    
    print("\nConfound probe tests passed!")


if __name__ == '__main__':
    test_confound_probes()
