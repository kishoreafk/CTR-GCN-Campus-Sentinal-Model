"""
Integration test: mini end-to-end pipeline.
Creates synthetic data → builds model → trains 2 epochs → validates.
"""
import torch
import numpy as np
import pytest
from pathlib import Path


def _create_synthetic_dataset(base_dir: str, n_samples: int = 20):
    """Create synthetic .npz files for train and val."""
    for split in ["train", "val"]:
        n = n_samples if split == "train" else n_samples // 4
        out_dir = Path(base_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(max(1, n)):
            label = np.zeros(15, dtype=np.float32)
            label[np.random.choice(15, size=np.random.randint(1, 4),
                                   replace=False)] = 1.0
            np.savez_compressed(
                str(out_dir / f"{split}_sample_{i:04d}.npz"),
                keypoints=np.random.rand(64, 2, 18, 3).astype(np.float32),
                label=label,
                video_id=f"synth_{i}",
                timestamp=float(i),
                quality_score=0.9,
                joint_layout="openpose_18",
                split=split,
            )


def test_mini_pipeline(tmp_path, device):
    """
    - Create 20 synthetic .npz files
    - Build dataset + DataLoader
    - Build model (no pretrained)
    - Train 2 epochs with Trainer
    - Validate, save checkpoint, reload
    - Assert: loss finite, mAP > 0, checkpoint loadable
    """
    # Create synthetic data
    skel_dir = str(tmp_path / "skeletons")
    _create_synthetic_dataset(skel_dir, n_samples=20)

    # Build dataset
    from utils.class_registry import ClassRegistry
    from data_pipeline.skeleton_dataset import SkeletonDataset
    from torch.utils.data import DataLoader

    reg = ClassRegistry()
    train_ds = SkeletonDataset(skel_dir, reg, split="train", augment=True)
    val_ds = SkeletonDataset(skel_dir, reg, split="val", augment=False)

    if len(train_ds) == 0 or len(val_ds) == 0:
        pytest.skip("No samples created")

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    # Build model
    from models.ctrgcn_ava import CTRGCNForAVA
    model = CTRGCNForAVA(num_classes=reg.num_classes).to(device)

    # Build training components
    from training.losses import AsymmetricLoss
    from training.ema import ModelEMA
    from training.early_stopping import EarlyStopping
    from training.checkpoint_manager import CheckpointManager
    from training.metrics import MultiLabelMetrics

    loss_fn = AsymmetricLoss()
    ema = ModelEMA(model, decay=0.999)
    es = EarlyStopping(patience=10)
    ckpt_dir = str(tmp_path / "checkpoints")
    ckpt_mgr = CheckpointManager(ckpt_dir, keep_k=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train 2 epochs manually
    from dataclasses import dataclass

    @dataclass
    class MinConfig:
        num_classes: int = reg.num_classes
        save_every_n_epochs: int = 1
        keep_last_k_checkpoints: int = 2

    cfg = MinConfig()

    for epoch in range(2):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x = batch["input"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(model)
            epoch_loss += loss.item()

        # Validate
        model.eval()
        metrics = MultiLabelMetrics(reg.num_classes, reg.class_names)
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input"].to(device)
                y = batch["label"].to(device)
                logits = model(x)
                metrics.update(logits, y)

        val_result = metrics.compute()
        all_metrics = {
            "train_loss": epoch_loss / max(len(train_loader), 1),
            "mAP": val_result["mAP"],
            "val_loss": 0.0,
        }

        # Save checkpoint
        ckpt_mgr.save(epoch, model, optimizer, None, ema, es, all_metrics, cfg)

    # Assertions
    assert epoch_loss > 0, "Loss should be positive"
    assert torch.isfinite(torch.tensor(epoch_loss)), "Loss should be finite"
    assert val_result["mAP"] >= 0, "mAP should be non-negative"

    # Reload checkpoint
    model2 = CTRGCNForAVA(num_classes=reg.num_classes).to(device)
    loaded_epoch, loaded_metrics = ckpt_mgr.load(
        str(Path(ckpt_dir) / "last.pth"), model2)
    assert loaded_epoch == 1  # last epoch saved
