"""
CTR-GCN adapted for multi-label AVA action recognition.
Head outputs raw logits (no sigmoid) — BCEWithLogitsLoss applies sigmoid.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from typing import List
from models.ctrgcn.graph import OpenPoseGraph

class CTRGCNForAVA(nn.Module):
    def __init__(self, num_classes: int,
                 pretrained_state_dict=None,
                 dropout: float = 0.1):
        super().__init__()
        from models.ctrgcn.ctrgcn import CTRGCN  # port of official repo
        graph = OpenPoseGraph(strategy="spatial")
        self.backbone = CTRGCN(graph=graph, in_channels=3)   # outputs (N,256)
        self.dropout  = nn.Dropout(p=dropout)
        self.head     = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
            # No sigmoid — BCEWithLogitsLoss is numerically more stable
        )
        self.num_classes = num_classes

        if pretrained_state_dict:
            self._load_pretrained(pretrained_state_dict)

    def _load_pretrained(self, state_dict: dict):
        """Load backbone weights, skip head (shape mismatch expected)."""
        own = self.state_dict()
        loaded, skipped = [], []
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        self.load_state_dict(own, strict=False)
        print(f"Loaded {len(loaded)} keys, skipped {len(skipped)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, 64, 18, 2) → logits: (N, num_classes)"""
        feat = self.backbone(x)   # (N, 256)
        feat = self.dropout(feat)
        return self.head(feat)    # (N, num_classes) — raw logits

    def get_param_groups(self, lr_backbone: float, lr_head: float):
        return [
            {"params": self.backbone.parameters(), "lr": lr_backbone},
            {"params": list(self.dropout.parameters()) +
                       list(self.head.parameters()),    "lr": lr_head},
        ]

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_layers(self, layer_names: List[str]):
        if "ALL" in layer_names:
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            return
        for name, module in self.backbone.named_modules():
            for target in layer_names:
                if target in name:
                    for p in module.parameters():
                        p.requires_grad_(True)
