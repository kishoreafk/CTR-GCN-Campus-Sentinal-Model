"""Tests for CTR-GCN model architecture."""
import torch
import pytest
from models.ctrgcn_ava import CTRGCNForAVA
from models.ctrgcn.graph import OpenPoseGraph


def test_forward_shape(device, dummy_skeleton):
    """(2,3,64,18,2) → (2, num_classes)."""
    model = CTRGCNForAVA(num_classes=15).to(device)
    x = dummy_skeleton.to(device)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 15)


def test_logits_not_bounded(device, dummy_skeleton):
    """Output should not be forced to [0,1] (no sigmoid in model)."""
    model = CTRGCNForAVA(num_classes=15).to(device)
    x = dummy_skeleton.to(device)
    with torch.no_grad():
        out = model(x)
    # Raw logits can be outside [0,1]
    # At least some values should be outside [0,1] for random weights
    assert out.min() < 0 or out.max() > 1


def test_freeze_backbone(device):
    """Backbone params should have requires_grad=False after freeze."""
    model = CTRGCNForAVA(num_classes=15).to(device)
    model.freeze_backbone()
    for p in model.backbone.parameters():
        assert not p.requires_grad


def test_gradual_unfreeze(device):
    """st_gcn_networks.9 should be unfrozen correctly by name."""
    model = CTRGCNForAVA(num_classes=15).to(device)
    model.freeze_backbone()

    # All backbone params should be frozen
    for p in model.backbone.parameters():
        assert not p.requires_grad

    # Unfreeze specific layer
    model.unfreeze_layers(["st_gcn_networks.9"])

    # At least some params in st_gcn_networks.9 should be unfrozen
    found_unfrozen = False
    for name, p in model.backbone.named_parameters():
        if "st_gcn_networks.9" in name:
            if p.requires_grad:
                found_unfrozen = True
    assert found_unfrozen


def test_head_rebuild(device):
    """Mismatch num_classes → head re-initialized, backbone ok."""
    model1 = CTRGCNForAVA(num_classes=400).to(device)
    sd = model1.state_dict()

    # Build model with different num_classes, loading pretrained
    model2 = CTRGCNForAVA(num_classes=15, pretrained_state_dict=sd).to(device)

    # Model 2 should have 15-class head
    assert model2.num_classes == 15
    x = torch.randn(1, 3, 64, 18, 2).to(device)
    with torch.no_grad():
        out = model2(x)
    assert out.shape == (1, 15)


def test_graph_shape():
    """OpenPose graph should have shape (3, 18, 18)."""
    g = OpenPoseGraph(strategy="spatial")
    assert g.A.shape == (3, 18, 18)
    assert g.num_nodes == 18
