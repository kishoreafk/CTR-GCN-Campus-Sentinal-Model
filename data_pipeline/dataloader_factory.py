import torch
from torch.utils.data import DataLoader
from data_pipeline.ava_kinetics_dataset import AVAKineticsDataset
from data_pipeline.ava_dataset import AVADataset

def create_dataloaders(config):
    DS = AVAKineticsDataset if config.dataset == "ava_kinetics" else AVADataset
    train_ds = DS(config, split="train")
    val_ds   = DS(config, split="val")

    loader_kwargs = dict(
        num_workers = config.num_workers,
        pin_memory  = config.pin_memory,
    )

    # persistent_workers + prefetch_factor need num_workers > 0
    if config.num_workers > 0:
        try:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"]    = config.prefetch_factor
        except Exception:
            pass  # Fall back gracefully on some platforms

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size * 2,
                              shuffle=False, **loader_kwargs)

    return train_loader, val_loader, train_ds
