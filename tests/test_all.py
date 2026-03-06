"""
Unit tests for ExplainMoE-ADHD v2.13.

Tests all core components:
- Encoders (EEG 19ch, 10ch, 5ch, Clinical, Actigraphy, Eye-tracking)
- Projection heads
- FuseMoE (routing, expert selection, load balancing, K-means init)
- Task heads (Diagnosis, Subtype, Severity)
- Loss functions
- Full model forward pass
- Phase configuration
- Training phases
- Data utilities
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# Encoder Tests
# ============================================================================

class TestEEGEncoders:
    """Test all EEG encoder variants."""

    def test_child_eeg_19ch_output_shape(self):
        from explainmoe_adhd.models.components.eeg_encoders import ChildEEG19chEncoder
        encoder = ChildEEG19chEncoder()
        x = torch.randn(4, 19, 256)  # batch=4, 19 channels, 256 timepoints (2s @ 128Hz)
        h = encoder(x, hw_id=0)
        assert h.shape == (4, 256), f"Expected (4, 256), got {h.shape}"

    def test_child_eeg_10ch_output_shape(self):
        from explainmoe_adhd.models.components.eeg_encoders import ChildEEG10chEncoder
        encoder = ChildEEG10chEncoder()
        x = torch.randn(4, 10, 256)
        h = encoder(x, hw_id=1)
        assert h.shape == (4, 256), f"Expected (4, 256), got {h.shape}"

    def test_adult_eeg_5ch_output_shape(self):
        from explainmoe_adhd.models.components.eeg_encoders import AdultEEG5chEncoder
        encoder = AdultEEG5chEncoder()
        x = torch.randn(4, 5, 512)  # 256Hz * 2s = 512 samples
        h = encoder(x, hw_id=2)
        assert h.shape == (4, 256), f"Expected (4, 256), got {h.shape}"

    def test_eeg_encoder_default_hw_id(self):
        from explainmoe_adhd.models.components.eeg_encoders import ChildEEG19chEncoder
        encoder = ChildEEG19chEncoder()
        x = torch.randn(2, 19, 256)
        h = encoder(x)  # hw_id defaults to hardware_token_id
        assert h.shape == (2, 256)

    def test_eeg_encoder_gradient_flow(self):
        from explainmoe_adhd.models.components.eeg_encoders import ChildEEG19chEncoder
        encoder = ChildEEG19chEncoder()
        x = torch.randn(2, 19, 256)
        h = encoder(x)
        loss = h.sum()
        loss.backward()
        # Check that gradients exist on at least one parameter
        has_grad = any(p.grad is not None for p in encoder.parameters())
        assert has_grad, "No gradients computed"

    def test_eeg_single_sample(self):
        from explainmoe_adhd.models.components.eeg_encoders import ChildEEG19chEncoder
        encoder = ChildEEG19chEncoder()
        x = torch.randn(1, 19, 256)
        h = encoder(x)
        assert h.shape == (1, 256)


class TestClinicalEncoder:
    """Test Clinical/fMRI encoder."""

    def test_output_shape_with_fmri(self):
        from explainmoe_adhd.models.components.clinical_encoder import ClinicalEncoder
        encoder = ClinicalEncoder()
        # Flat tensor: [age, sex, handedness, IQ, site_id, dataset_source]
        tabular = torch.tensor([[10.0, 0, 1, 100.0, 5, 2],
                                [12.0, 1, 0, 95.0, 3, 1]])
        fmri = torch.randn(2, 4005)
        mean_fd = torch.tensor([0.3, 0.5])
        h = encoder(tabular, fmri, mean_fd)
        assert h.shape == (2, 256), f"Expected (2, 256), got {h.shape}"

    def test_output_shape_without_fmri(self):
        from explainmoe_adhd.models.components.clinical_encoder import ClinicalEncoder
        encoder = ClinicalEncoder()
        tabular = torch.tensor([[10.0, 0, 1, 100.0, 5, 2]])
        h = encoder(tabular, None, None)
        assert h.shape == (1, 256), f"Expected (1, 256), got {h.shape}"

    def test_gradient_flow(self):
        from explainmoe_adhd.models.components.clinical_encoder import ClinicalEncoder
        encoder = ClinicalEncoder()
        tabular = torch.tensor([[10.0, 0, 1, 100.0, 5, 2]])
        fmri = torch.randn(1, 4005)
        mean_fd = torch.tensor([0.3])
        h = encoder(tabular, fmri, mean_fd)
        h.sum().backward()
        has_grad = any(p.grad is not None for p in encoder.parameters())
        assert has_grad


class TestActigraphyEncoder:
    """Test Actigraphy encoder."""

    def test_output_shape(self):
        from explainmoe_adhd.models.components.actigraphy_encoder import ActigraphyEncoder
        encoder = ActigraphyEncoder()
        ts = torch.randn(4, 4, 1000)  # 4 channels (3 accel + HR), 1000 timepoints
        age = torch.tensor([25.0, 30.0, 22.0, 45.0])
        sex = torch.tensor([0.0, 1.0, 1.0, 0.0])
        h = encoder(ts, age, sex)
        assert h.shape == (4, 256), f"Expected (4, 256), got {h.shape}"

    def test_gradient_flow(self):
        from explainmoe_adhd.models.components.actigraphy_encoder import ActigraphyEncoder
        encoder = ActigraphyEncoder()
        ts = torch.randn(2, 4, 500)
        age = torch.tensor([25.0, 30.0])
        sex = torch.tensor([0.0, 1.0])
        h = encoder(ts, age, sex)
        h.sum().backward()
        has_grad = any(p.grad is not None for p in encoder.parameters())
        assert has_grad


class TestEyeTrackingEncoder:
    """Test Eye-tracking encoder."""

    def test_output_shape(self):
        from explainmoe_adhd.models.components.eye_tracking_encoder import EyeTrackingEncoder
        encoder = EyeTrackingEncoder()
        gaze = torch.randn(4, 100, 3)  # batch=4, 100 timesteps, (x, y, pupil)
        h = encoder(gaze)
        assert h.shape == (4, 256), f"Expected (4, 256), got {h.shape}"

    def test_freeze_behavior(self):
        from explainmoe_adhd.models.components.eye_tracking_encoder import EyeTrackingEncoder
        encoder = EyeTrackingEncoder()
        for p in encoder.parameters():
            p.requires_grad = False
        gaze = torch.randn(2, 50, 3)
        h = encoder(gaze)
        assert h.shape == (2, 256)
        # No gradients should flow
        assert all(not p.requires_grad for p in encoder.parameters())


# ============================================================================
# Projection Head Tests
# ============================================================================

class TestProjectionHead:
    """Test projection heads."""

    def test_output_shape(self):
        from explainmoe_adhd.models.components.projection_heads import ProjectionHead
        head = ProjectionHead(input_dim=256, output_dim=256, depth=2)
        h = torch.randn(8, 256)
        z = head(h)
        assert z.shape == (8, 256)

    def test_depth_1(self):
        from explainmoe_adhd.models.components.projection_heads import ProjectionHead
        head = ProjectionHead(input_dim=256, output_dim=256, depth=1)
        h = torch.randn(4, 256)
        z = head(h)
        assert z.shape == (4, 256)

    def test_depth_3(self):
        from explainmoe_adhd.models.components.projection_heads import ProjectionHead
        head = ProjectionHead(input_dim=256, output_dim=256, depth=3)
        h = torch.randn(4, 256)
        z = head(h)
        assert z.shape == (4, 256)

    def test_xavier_init(self):
        from explainmoe_adhd.models.components.projection_heads import ProjectionHead
        head = ProjectionHead(input_dim=256, output_dim=256, depth=2)
        for m in head.modules():
            if isinstance(m, nn.Linear):
                # Xavier init produces weights with variance ~2/(fan_in+fan_out)
                weight_var = m.weight.data.var().item()
                assert weight_var > 0, "Weights should not be zero"
                assert weight_var < 1.0, "Weights variance unexpectedly large"


# ============================================================================
# FuseMoE Tests
# ============================================================================

class TestFuseMoE:
    """Test FuseMoE module."""

    def test_output_shape(self):
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        moe = FuseMoE(input_dim=256, num_experts=4, top_k=2, num_routers=5)
        z = torch.randn(8, 256)
        y_moe, expert_weights = moe(z, router_idx=0, return_expert_weights=True)
        assert y_moe.shape == (8, 256), f"Expected (8, 256), got {y_moe.shape}"
        assert expert_weights.shape == (4,), f"Expected (4,), got {expert_weights.shape}"

    def test_all_routers(self):
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        moe = FuseMoE(input_dim=256, num_experts=4, top_k=2, num_routers=5)
        z = torch.randn(4, 256)
        for router_idx in range(5):
            y_moe, ew = moe(z, router_idx, return_expert_weights=True)
            assert y_moe.shape == (4, 256)

    def test_expert_weights_sum(self):
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        moe = FuseMoE(input_dim=256, num_experts=4, top_k=2, num_routers=5)
        z = torch.randn(16, 256)
        y_moe, ew = moe(z, router_idx=0, return_expert_weights=True)
        # Expert weights should sum to approximately 1 (normalized)
        assert abs(ew.sum().item() - 1.0) < 0.01, f"Expert weights sum: {ew.sum().item()}"

    def test_load_balance_loss(self):
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        moe = FuseMoE(input_dim=256, num_experts=4, top_k=2, num_routers=5)
        z = torch.randn(32, 256)
        y_moe, ew = moe(z, router_idx=0, return_expert_weights=True)
        lb_loss = moe.load_balance_loss(ew)
        assert lb_loss.shape == (), "Load balance loss should be scalar"
        assert lb_loss.item() >= 0, "Load balance loss should be non-negative"

    def test_kmeans_initialization(self):
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        moe = FuseMoE(input_dim=256, num_experts=4, top_k=2, num_routers=5)
        z_all = torch.randn(100, 256)
        moe.initialize_centroids_kmeans(z_all)
        # Centroids should be updated
        for r in range(5):
            centroid = moe.centroids[r].data
            assert centroid.shape == (4, 256)
            # All routers should have identical centroids after init
            assert torch.allclose(moe.centroids[0].data, centroid)

    def test_temperature_positive(self):
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        moe = FuseMoE(input_dim=256, num_experts=4, top_k=2, num_routers=5)
        for r in range(5):
            tau = moe.get_temperature(r)
            assert tau.item() > 0, f"Temperature for router {r} should be positive"

    def test_gradient_flow(self):
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        moe = FuseMoE(input_dim=256, num_experts=4, top_k=2, num_routers=5)
        z = torch.randn(8, 256, requires_grad=True)
        y_moe, ew = moe(z, router_idx=0, return_expert_weights=True)
        loss = y_moe.sum()
        loss.backward()
        assert z.grad is not None, "Gradients should flow through FuseMoE"

    def test_residual_connection(self):
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        # With residual
        moe_res = FuseMoE(input_dim=256, num_experts=4, top_k=2, num_routers=5, use_residual=True)
        # Without residual
        moe_nores = FuseMoE(input_dim=256, num_experts=4, top_k=2, num_routers=5, use_residual=False)
        z = torch.randn(4, 256)
        y_res, _ = moe_res(z, 0)
        y_nores, _ = moe_nores(z, 0)
        # Just check both produce valid output shapes
        assert y_res.shape == (4, 256)
        assert y_nores.shape == (4, 256)


# ============================================================================
# Task Head Tests
# ============================================================================

class TestTaskHeads:
    """Test all task heads."""

    def test_diagnosis_head_shape(self):
        from explainmoe_adhd.models.components.task_heads import DiagnosisHead
        head = DiagnosisHead(input_dim=256)
        y = torch.randn(8, 256)
        logits = head(y)
        assert logits.shape == (8, 1)

    def test_diagnosis_head_predict_proba(self):
        from explainmoe_adhd.models.components.task_heads import DiagnosisHead
        head = DiagnosisHead(input_dim=256)
        y = torch.randn(4, 256)
        probs = head.predict_proba(y)
        assert probs.shape == (4, 1)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_subtype_head_shape(self):
        from explainmoe_adhd.models.components.task_heads import SubtypeHead
        head = SubtypeHead(input_dim=256, num_classes=3)
        y = torch.randn(8, 256)
        logits = head(y)
        assert logits.shape == (8, 3)

    def test_severity_head_shape(self):
        from explainmoe_adhd.models.components.task_heads import SeverityHead
        head = SeverityHead(input_dim=256)
        y = torch.randn(8, 256)
        preds = head(y)
        assert preds.shape == (8, 2)


# ============================================================================
# Loss Function Tests
# ============================================================================

class TestLosses:
    """Test all loss functions."""

    def test_diagnosis_loss(self):
        from explainmoe_adhd.training.losses import DiagnosisLoss
        loss_fn = DiagnosisLoss()
        logits = torch.randn(16, 1)
        targets = torch.randint(0, 2, (16, 1)).float()
        loss = loss_fn(logits, targets)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_subtype_loss_with_mask(self):
        from explainmoe_adhd.training.losses import SubtypeLoss
        loss_fn = SubtypeLoss()
        logits = torch.randn(16, 3)
        targets = torch.randint(0, 3, (16,))
        mask = torch.rand(16) > 0.5
        loss = loss_fn(logits, targets, mask)
        assert loss.shape == ()

    def test_subtype_loss_empty_mask(self):
        from explainmoe_adhd.training.losses import SubtypeLoss
        loss_fn = SubtypeLoss()
        logits = torch.randn(16, 3)
        targets = torch.randint(0, 3, (16,))
        mask = torch.zeros(16, dtype=torch.bool)  # No valid samples
        loss = loss_fn(logits, targets, mask)
        assert loss.item() == 0.0

    def test_severity_loss(self):
        from explainmoe_adhd.training.losses import SeverityLoss
        loss_fn = SeverityLoss()
        preds = torch.randn(16, 2)
        targets = torch.randn(16, 2)
        mask = torch.rand(16) > 0.3
        loss = loss_fn(preds, targets, mask)
        assert loss.shape == ()

    def test_mmd_loss(self):
        from explainmoe_adhd.training.losses import MMDLoss
        loss_fn = MMDLoss(min_batch_per_class=4)
        z_a_adhd = torch.randn(8, 256)
        z_b_adhd = torch.randn(8, 256)
        z_a_ctrl = torch.randn(8, 256)
        z_b_ctrl = torch.randn(8, 256)
        loss = loss_fn(z_a_adhd, z_b_adhd, z_a_ctrl, z_b_ctrl)
        assert loss.shape == ()
        assert torch.isfinite(loss), "MMD loss should be finite"

    def test_mmd_loss_insufficient_samples(self):
        from explainmoe_adhd.training.losses import MMDLoss
        loss_fn = MMDLoss(min_batch_per_class=16)
        z_a_adhd = torch.randn(4, 256)  # Too few
        z_b_adhd = torch.randn(4, 256)
        z_a_ctrl = torch.randn(4, 256)
        z_b_ctrl = torch.randn(4, 256)
        loss = loss_fn(z_a_adhd, z_b_adhd, z_a_ctrl, z_b_ctrl)
        assert loss.item() == 0.0, "Should return 0 when below min_batch_per_class"

    def test_combined_loss_phase4(self):
        from explainmoe_adhd.training.losses import CombinedLoss
        loss_fn = CombinedLoss(
            diagnosis_weight=1.0,
            subtype_weight=0.25,
            load_balance_weight=0.01,
            use_subtype=True,
        )
        diag_logits = torch.randn(16, 1)
        diag_targets = torch.randint(0, 2, (16, 1)).float()
        sub_logits = torch.randn(16, 3)
        sub_targets = torch.randint(0, 3, (16,))
        sub_mask = torch.ones(16, dtype=torch.bool)
        expert_weights = torch.ones(4) / 4

        total, loss_dict = loss_fn(
            diag_logits, diag_targets,
            sub_logits, sub_targets, sub_mask,
            expert_weights=expert_weights,
        )
        assert "diagnosis" in loss_dict
        assert "subtype" in loss_dict
        assert "total" in loss_dict

    def test_combined_loss_phase5(self):
        from explainmoe_adhd.training.losses import CombinedLoss
        loss_fn = CombinedLoss(
            diagnosis_weight=1.0,
            subtype_weight=0.25,
            severity_weight=0.1,
            mmd_weight=0.3,
            load_balance_weight=0.01,
            use_subtype=True,
            use_severity=True,
            use_mmd=True,
        )
        B = 16
        diag_logits = torch.randn(B, 1)
        diag_targets = torch.randint(0, 2, (B, 1)).float()
        sub_logits = torch.randn(B, 3)
        sub_targets = torch.randint(0, 3, (B,))
        sub_mask = torch.ones(B, dtype=torch.bool)
        sev_preds = torch.randn(B, 2)
        sev_targets = torch.randn(B, 2)
        sev_mask = torch.ones(B, dtype=torch.bool)
        mmd_pair = (torch.randn(8, 256), torch.randn(8, 256),
                     torch.randn(8, 256), torch.randn(8, 256))
        ew = torch.ones(4) / 4

        total, loss_dict = loss_fn(
            diag_logits, diag_targets,
            sub_logits, sub_targets, sub_mask,
            sev_preds, sev_targets, sev_mask,
            [mmd_pair], ew,
        )
        assert "diagnosis" in loss_dict
        assert "subtype" in loss_dict
        assert "severity" in loss_dict
        assert "mmd" in loss_dict


# ============================================================================
# Full Model Tests
# ============================================================================

class TestExplainMoEModel:
    """Test the full ExplainMoE model."""

    @pytest.fixture
    def model(self):
        from explainmoe_adhd.models.explainmoe_model import ExplainMoEModel
        return ExplainMoEModel(latent_dim=256, num_experts=4, top_k=2)

    def test_forward_child_eeg_19ch(self, model):
        batch = {
            "eeg": torch.randn(4, 19, 256),
            "hardware_token_id": torch.tensor([0, 0, 0, 0]),
            "label": torch.randint(0, 2, (4,)),
        }
        outputs = model(batch, "child_eeg_19ch")
        assert outputs["diagnosis_logits"].shape == (4, 1)
        assert outputs["subtype_logits"].shape == (4, 3)
        assert outputs["severity_preds"].shape == (4, 2)

    def test_forward_child_eeg_10ch(self, model):
        batch = {
            "eeg": torch.randn(4, 10, 256),
            "hardware_token_id": torch.tensor([1, 1, 1, 1]),
            "label": torch.randint(0, 2, (4,)),
        }
        outputs = model(batch, "child_eeg_10ch")
        assert outputs["diagnosis_logits"].shape == (4, 1)

    def test_forward_adult_eeg(self, model):
        batch = {
            "eeg": torch.randn(4, 5, 512),
            "hardware_token_id": torch.tensor([2, 2, 2, 2]),
            "label": torch.randint(0, 2, (4,)),
        }
        outputs = model(batch, "adult_eeg_5ch")
        assert outputs["diagnosis_logits"].shape == (4, 1)

    def test_forward_clinical(self, model):
        batch = {
            "tabular": {
                "categorical": torch.tensor([[0, 1, 5, 2], [1, 0, 3, 1], [0, 2, 7, 0], [1, 1, 2, 3]]),
                "continuous": torch.tensor([[10.0, 100.0], [12.0, 95.0], [8.0, 110.0], [15.0, 88.0]]),
            },
            "fmri": torch.randn(4, 4005),
            "mean_fd": torch.tensor([0.3, 0.5, 0.2, 0.4]),
            "label": torch.randint(0, 2, (4,)),
        }
        outputs = model(batch, "clinical")
        assert outputs["diagnosis_logits"].shape == (4, 1)

    def test_forward_actigraphy(self, model):
        batch = {
            "timeseries": torch.randn(4, 4, 1000),
            "age": torch.tensor([25.0, 30.0, 22.0, 45.0]),
            "sex": torch.tensor([0.0, 1.0, 1.0, 0.0]),
            "label": torch.randint(0, 2, (4,)),
        }
        outputs = model(batch, "actigraphy")
        assert outputs["diagnosis_logits"].shape == (4, 1)

    def test_return_intermediates(self, model):
        batch = {
            "eeg": torch.randn(2, 19, 256),
            "hardware_token_id": torch.tensor([0, 0]),
        }
        outputs = model(batch, "child_eeg_19ch", return_intermediates=True)
        assert "h_m" in outputs
        assert "z_m" in outputs
        assert "y_moe" in outputs
        assert outputs["h_m"].shape == (2, 256)
        assert outputs["z_m"].shape == (2, 256)
        assert outputs["y_moe"].shape == (2, 256)


# ============================================================================
# Phase Configuration Tests
# ============================================================================

class TestPhaseConfiguration:
    """Test model phase configuration."""

    @pytest.fixture
    def model(self):
        from explainmoe_adhd.models.explainmoe_model import ExplainMoEModel
        return ExplainMoEModel()

    def test_configure_phase2(self, model):
        model.configure_for_phase(2)
        # Encoders should be trainable
        for enc in model.encoders.values():
            assert any(p.requires_grad for p in enc.parameters())
        # Eye-tracking always frozen
        assert all(not p.requires_grad for p in model.eye_tracking_encoder.parameters())
        # Projection heads frozen
        for head in model.projection_heads.values():
            assert all(not p.requires_grad for p in head.parameters())
        # FuseMoE frozen
        assert all(not p.requires_grad for p in model.fusemoe.parameters())

    def test_configure_phase3(self, model):
        model.configure_for_phase(3)
        # Encoders frozen
        for enc in model.encoders.values():
            assert all(not p.requires_grad for p in enc.parameters())
        # Projection heads trainable
        for head in model.projection_heads.values():
            assert any(p.requires_grad for p in head.parameters())

    def test_configure_phase4(self, model):
        model.configure_for_phase(4)
        # Encoders frozen
        for enc in model.encoders.values():
            assert all(not p.requires_grad for p in enc.parameters())
        # Projection heads trainable (0.1x LR)
        for head in model.projection_heads.values():
            assert any(p.requires_grad for p in head.parameters())
        # FuseMoE trainable
        assert any(p.requires_grad for p in model.fusemoe.parameters())

    def test_configure_phase5(self, model):
        model.configure_for_phase(5)
        # Encoders trainable
        for enc in model.encoders.values():
            assert any(p.requires_grad for p in enc.parameters())
        # Eye-tracking still frozen
        assert all(not p.requires_grad for p in model.eye_tracking_encoder.parameters())
        # Everything else trainable
        assert any(p.requires_grad for p in model.fusemoe.parameters())

    def test_get_parameter_groups_phase4(self, model):
        model.configure_for_phase(4)
        groups = model.get_parameter_groups(4)
        assert len(groups) >= 2, "Phase 4 should have multiple param groups"
        # Check LR values
        lrs = set()
        for g in groups:
            if "lr" in g:
                lrs.add(g["lr"])
        assert 3e-4 in lrs or 3e-5 in lrs, f"Expected 3e-4 or 3e-5 in LRs, got {lrs}"

    def test_reinitialize_projection_heads(self, model):
        # Modify weights
        for head in model.projection_heads.values():
            for p in head.parameters():
                p.data.fill_(999.0)
        # Reinitialize
        model.reinitialize_projection_heads()
        # Check weights are no longer 999
        for head in model.projection_heads.values():
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    assert m.weight.data.mean().item() != 999.0

    def test_reinitialize_task_heads(self, model):
        model.reinitialize_task_heads()
        # Check task head weights are properly initialized
        for m in model.diagnosis_head.modules():
            if isinstance(m, nn.Linear):
                assert abs(m.bias.data.sum().item()) < 0.01


# ============================================================================
# Training Phase Tests
# ============================================================================

class TestTrainingPhases:
    """Test training phase implementations."""

    def test_phase1_instantiation(self):
        from explainmoe_adhd.training.phases.phase1 import Phase1Pretraining
        model = nn.Linear(256, 256)
        phase = Phase1Pretraining(model, {"mask_prob": 0.15}, device="cpu")
        assert not phase.is_maximize()

    def test_phase2_instantiation(self):
        from explainmoe_adhd.training.phases.phase2 import Phase2SupervisedEncoder
        model = nn.Linear(256, 256)
        phase = Phase2SupervisedEncoder(model, {}, device="cpu")
        assert phase.is_maximize()

    def test_phase3_instantiation(self):
        from explainmoe_adhd.training.phases.phase3 import Phase3MMDAlignment
        model = nn.Linear(256, 256)
        phase = Phase3MMDAlignment(model, {}, device="cpu")
        assert not phase.is_maximize()

    def test_phase4_instantiation(self):
        from explainmoe_adhd.training.phases.phase4 import Phase4FuseMoE
        model = nn.Linear(256, 256)
        phase = Phase4FuseMoE(model, {}, device="cpu")
        assert phase.is_maximize()

    def test_phase5_instantiation(self):
        from explainmoe_adhd.training.phases.phase5 import Phase5FineTuning
        model = nn.Linear(256, 256)
        phase = Phase5FineTuning(model, {}, device="cpu")
        assert phase.is_maximize()


# ============================================================================
# Cross-Validation Tests
# ============================================================================

class TestCrossValidation:
    """Test cross-validation protocol."""

    def test_cv_split_creation(self):
        from explainmoe_adhd.data.cross_validation import CrossValidator
        cv = CrossValidator(n_folds=5, seed=42)
        n = 100
        subjects = [f"sub_{i}" for i in range(n)]
        labels = np.array([i % 2 for i in range(n)])
        groups = np.arange(n)
        stratify = labels

        splits = cv.create_splits(subjects, labels, groups, stratify)
        assert len(splits) == 5

        # Check no overlap between test sets across folds
        all_test = set()
        for split in splits:
            test_set = set(split.test_subjects)
            assert len(all_test & test_set) == 0, "Test sets overlap across folds!"
            all_test |= test_set

    def test_cv_no_subject_leakage(self):
        from explainmoe_adhd.data.cross_validation import CrossValidator
        cv = CrossValidator(n_folds=5, seed=42)
        n = 100
        subjects = [f"sub_{i}" for i in range(n)]
        labels = np.array([i % 2 for i in range(n)])
        groups = np.arange(n)

        splits = cv.create_splits(subjects, labels, groups, labels)
        for split in splits:
            train_set = set(split.train_subjects)
            val_set = set(split.val_subjects)
            test_set = set(split.test_subjects)
            assert train_set.isdisjoint(val_set), "Train/Val overlap!"
            assert train_set.isdisjoint(test_set), "Train/Test overlap!"
            assert val_set.isdisjoint(test_set), "Val/Test overlap!"


# ============================================================================
# Data Constants Tests
# ============================================================================

class TestDataConstants:
    """Test data constants and dataset specifications."""

    def test_all_datasets_defined(self):
        from explainmoe_adhd.data.constants import DATASETS
        assert len(DATASETS) == 11  # D1-D11

    def test_group_a_datasets(self):
        from explainmoe_adhd.data.constants import DATASETS
        group_a = {k: v for k, v in DATASETS.items() if v.is_group_a}
        # D1-D3, D5-D8 (7 total, D4 is validation only)
        assert len(group_a) == 7, f"Expected 7 Group A datasets, got {len(group_a)}"

    def test_group_b_datasets(self):
        from explainmoe_adhd.data.constants import DATASETS
        group_b = {k: v for k, v in DATASETS.items() if v.is_group_b}
        assert len(group_b) == 3  # D9, D10, D11

    def test_modality_enum(self):
        from explainmoe_adhd.data.constants import Modality
        assert len(Modality) == 6

    def test_hardware_token_enum(self):
        from explainmoe_adhd.data.constants import HardwareToken
        assert HardwareToken.CLINICAL_19CH_WET.value == 0
        assert HardwareToken.EMOTIV_10CH_SALINE.value == 1
        assert HardwareToken.MENDELEY_5CH.value == 2
        assert HardwareToken.REPOD_VARIABLE.value == 3

    def test_dataset_subject_counts(self):
        from explainmoe_adhd.data.constants import DATASETS
        # Verify total counts per spec
        assert DATASETS["D1"].num_subjects == 121
        assert DATASETS["D2"].num_subjects == 103
        assert DATASETS["D3"].num_subjects == 79
        assert DATASETS["D5"].num_subjects == 973
        assert DATASETS["D6"].num_subjects == 57
        assert DATASETS["D7"].num_subjects == 103
        assert DATASETS["D8"].num_subjects == 50


# ============================================================================
# Config Tests
# ============================================================================

class TestConfigs:
    """Test configuration dataclasses."""

    def test_training_config_defaults(self):
        from explainmoe_adhd.config.training_config import TrainingConfig
        config = TrainingConfig()
        assert config.phase1.epochs == 30
        assert config.phase2.max_epochs == 50
        assert config.phase3.max_steps == 1000
        assert config.phase4.max_steps == 3000
        assert config.phase5.max_steps == 1000
        assert config.seed == 42

    def test_phase4_config(self):
        from explainmoe_adhd.config.training_config import Phase4Config
        config = Phase4Config()
        assert config.lr_main == 3e-4
        assert config.lr_projection == 3e-5
        assert config.severity_loss_weight == 0.0  # Inactive in Phase 4
        assert config.mmd_loss_weight == 0.0

    def test_phase5_config(self):
        from explainmoe_adhd.config.training_config import Phase5Config
        config = Phase5Config()
        assert config.lr == 1e-5
        assert config.severity_loss_weight == 0.1  # Active in Phase 5
        assert config.mmd_loss_weight == 0.3

    def test_loss_weights(self):
        from explainmoe_adhd.config.training_config import LOSS_WEIGHTS
        assert LOSS_WEIGHTS["phase4"]["severity"] == 0.0
        assert LOSS_WEIGHTS["phase5"]["severity"] == 0.1
        assert LOSS_WEIGHTS["phase5"]["mmd"] == 0.3

    def test_dataset_configs(self):
        from explainmoe_adhd.config.dataset_configs import (
            MMD_PAIRS, MODALITY_ROUTER_INDEX, GROUP_A_DATASETS, GROUP_B_DATASETS
        )
        assert len(MMD_PAIRS) == 4
        assert len(MODALITY_ROUTER_INDEX) == 5
        assert len(GROUP_A_DATASETS) >= 6
        assert len(GROUP_B_DATASETS) == 3


# ============================================================================
# Utility Tests
# ============================================================================

class TestUtilities:
    """Test utility functions."""

    def test_set_seed(self):
        from explainmoe_adhd.utils.common import set_seed
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_count_parameters(self):
        from explainmoe_adhd.utils.common import count_parameters
        model = nn.Linear(256, 128)
        total = count_parameters(model, trainable_only=False)
        assert total == 256 * 128 + 128  # weights + bias

    def test_get_device(self):
        from explainmoe_adhd.utils.common import get_device
        device = get_device(prefer_cuda=False)
        assert device == "cpu"

    def test_move_to_device(self):
        from explainmoe_adhd.utils.common import move_to_device
        batch = {"x": torch.randn(2, 3), "label": "test"}
        moved = move_to_device(batch, "cpu")
        assert moved["x"].device.type == "cpu"
        assert moved["label"] == "test"


# ============================================================================
# Evaluation Tests
# ============================================================================

class TestEvaluation:
    """Test evaluation utilities."""

    def test_compute_ece(self):
        from explainmoe_adhd.scripts.evaluate import compute_ece
        probs = np.array([0.1, 0.4, 0.6, 0.9])
        labels = np.array([0, 0, 1, 1])
        ece = compute_ece(probs, labels, n_bins=5)
        assert 0 <= ece <= 1

    def test_bootstrap_ci(self):
        from explainmoe_adhd.scripts.evaluate import bootstrap_ci
        labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        preds = np.array([0.1, 0.3, 0.2, 0.8, 0.7, 0.9, 0.4, 0.6, 0.3, 0.8])
        result = bootstrap_ci(labels, preds, n_bootstraps=100)
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert result["lower"] <= result["mean"] <= result["upper"]


# ============================================================================
# Dimension Chain Verification (Section 5.7)
# ============================================================================

class TestDimensionChain:
    """Verify the complete dimension chain from spec Section 5.7."""

    def test_full_eeg_19ch_chain(self):
        from explainmoe_adhd.models.components.eeg_encoders import ChildEEG19chEncoder
        from explainmoe_adhd.models.components.projection_heads import ProjectionHead
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        from explainmoe_adhd.models.components.task_heads import DiagnosisHead, SubtypeHead, SeverityHead

        encoder = ChildEEG19chEncoder()
        proj = ProjectionHead(256, 256, depth=2)
        moe = FuseMoE(256, 4, 2, 5)
        diag = DiagnosisHead(256)
        sub = SubtypeHead(256)
        sev = SeverityHead(256)

        x = torch.randn(2, 19, 256)
        h = encoder(x)                 # (2, 256)
        assert h.shape == (2, 256)

        z = proj(h)                    # (2, 256)
        assert z.shape == (2, 256)

        y, ew = moe(z, 0, True)        # (2, 256)
        assert y.shape == (2, 256)

        d = diag(y)                    # (2, 1)
        assert d.shape == (2, 1)

        s = sub(y)                     # (2, 3)
        assert s.shape == (2, 3)

        sv = sev(y)                    # (2, 2)
        assert sv.shape == (2, 2)

    def test_full_actigraphy_chain(self):
        from explainmoe_adhd.models.components.actigraphy_encoder import ActigraphyEncoder
        from explainmoe_adhd.models.components.projection_heads import ProjectionHead
        from explainmoe_adhd.models.components.fusemoe import FuseMoE
        from explainmoe_adhd.models.components.task_heads import DiagnosisHead

        encoder = ActigraphyEncoder()
        proj = ProjectionHead(256, 256, depth=2)
        moe = FuseMoE(256, 4, 2, 5)
        diag = DiagnosisHead(256)

        ts = torch.randn(2, 4, 500)
        age = torch.tensor([25.0, 30.0])
        sex = torch.tensor([0.0, 1.0])

        h = encoder(ts, age, sex)       # (2, 256)
        assert h.shape == (2, 256)

        z = proj(h)                     # (2, 256)
        assert z.shape == (2, 256)

        y, _ = moe(z, 4, True)          # (2, 256)
        assert y.shape == (2, 256)

        d = diag(y)                     # (2, 1)
        assert d.shape == (2, 1)


# ============================================================================
# Ablation Baseline Tests
# ============================================================================

class TestAblationBaselines:
    """Test ablation baselines (Conditions A, B, D)."""

    def test_condition_a_baseline_model(self):
        from explainmoe_adhd.models.ablation_baselines import BaselineModel
        model = BaselineModel(latent_dim=256)
        batch = {
            "eeg": torch.randn(2, 19, 256),
            "hardware_token_id": torch.tensor([0, 0]),
        }
        outputs = model(batch, "child_eeg_19ch")
        assert outputs["diagnosis_logits"].shape == (2, 1)

    def test_condition_b_moe_without_alignment(self):
        from explainmoe_adhd.models.ablation_baselines import MoEWithoutAlignmentModel
        model = MoEWithoutAlignmentModel()
        # Phase 3 should be rejected
        with pytest.raises(ValueError, match="Condition B"):
            model.configure_for_phase(3)
        # Phase 4 should work and give full LR to projection heads
        model.configure_for_phase(4)
        groups = model.get_parameter_groups(4)
        lrs = [g.get("lr") for g in groups]
        assert all(lr == 3e-4 for lr in lrs), f"All groups should use 3e-4, got {lrs}"

    def test_condition_d_simple_ml(self):
        from explainmoe_adhd.models.ablation_baselines import SimpleMLBaseline
        import numpy as np
        model = SimpleMLBaseline("logistic_regression")
        X = np.random.randn(50, 10)
        y = np.random.randint(0, 2, 50)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (50,)
        proba = model.predict_proba(X)
        assert proba.shape == (50,)


# ============================================================================
# Phase 2 Temporary Head Tests
# ============================================================================

class TestPhase2TempHeads:
    """Test Phase 2 uses temporary MLP heads instead of full pipeline."""

    def test_phase2_has_temp_heads(self):
        from explainmoe_adhd.training.phases.phase2 import Phase2SupervisedEncoder
        from explainmoe_adhd.models.explainmoe_model import ExplainMoEModel
        model = ExplainMoEModel()
        phase = Phase2SupervisedEncoder(model, {}, device="cpu")
        assert hasattr(phase, "temp_heads")
        assert len(phase.temp_heads) == 5  # One per modality

    def test_phase2_forward_uses_temp_head(self):
        from explainmoe_adhd.training.phases.phase2 import Phase2SupervisedEncoder
        from explainmoe_adhd.models.explainmoe_model import ExplainMoEModel
        model = ExplainMoEModel()
        model.configure_for_phase(2)
        phase = Phase2SupervisedEncoder(model, {}, device="cpu")
        batch = {
            "eeg": torch.randn(2, 19, 256),
            "hardware_token_id": torch.tensor([0, 0]),
            "label": torch.tensor([0.0, 1.0]),
            "modality": "child_eeg_19ch",
        }
        loss, metrics = phase.forward_step(batch)
        assert loss.item() > 0
        assert "accuracy" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
