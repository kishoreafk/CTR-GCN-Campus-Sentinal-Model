"""CTR-GCN model architecture modules."""

from models.ctrgcn_ava import CTRGCNForAVA
from models.model_factory import build_model

__all__ = [
    "CTRGCNForAVA",
    "build_model",
]
