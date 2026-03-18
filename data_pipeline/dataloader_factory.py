"""
DataLoader construction with stability guards:
  - Worker init function sets per-worker seeds (reproducibility)
  - Timeout on DataLoader prevents infinite hangs
  - Automatic fallback to num_workers=0 if multiprocessing fails
"""
import logging
import torch
from torch.utils.data import DataLoader
from data_pipeline.ava_kinetics_dataset import AVAKineticsDataset
from data_pipeline.ava_dataset import AVADataset

log = logging.getLogger("dataloader_factory")


def create_dataloaders(config, class_registry=None):
    """
    Create train and val DataLoaders with stability guards.

    Parameters
    ----------
    config          : TrainingConfig
    class_registry  : ClassRegistry subset (optional). If provided,
                      passed through to dataset constructors for
                      class-aware label building.
    """
    DS = AVAKineticsDataset if config.dataset == "ava_kinetics" else AVADataset
    train_ds = DS(config, split="train", class_registry=class_registry)
    val_ds   = DS(config, split="val",   class_registry=class_registry)

    def worker_init_fn(worker_id: int):
        """
        Each worker gets a unique seed derived from the global seed.
        Without this, all workers produce identical augmentation sequences.
        """
        import random, numpy as np
        seed = config.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    base_kwargs = dict(
        pin_memory     = config.pin_memory,
        worker_init_fn = worker_init_fn,
    )

    def _make_loader(dataset, batch_size, shuffle, drop_last):
        nw = config.num_workers
        kwargs = dict(**base_kwargs, num_workers=nw)

        if nw > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"]    = config.prefetch_factor

        try:
            loader = DataLoader(
                dataset,
                batch_size = batch_size,
                shuffle    = shuffle,
                drop_last  = drop_last,
                timeout    = 120,   # raise if batch not ready in 2 min
                **kwargs
            )
            # Smoke test: try fetching one batch
            _ = next(iter(loader))
            return loader

        except Exception as e:
            if nw > 0:
                log.warning(
                    f"DataLoader with num_workers={nw} failed ({e}). "
                    f"Falling back to num_workers=0."
                )
                kwargs["num_workers"] = 0
                kwargs.pop("persistent_workers", None)
                kwargs.pop("prefetch_factor", None)
                return DataLoader(
                    dataset,
                    batch_size = batch_size,
                    shuffle    = shuffle,
                    drop_last  = drop_last,
                    **kwargs
                )
            raise

    train_loader = _make_loader(train_ds, config.batch_size,
                                shuffle=True, drop_last=True)
    val_loader   = _make_loader(val_ds, config.batch_size * 2,
                                shuffle=False, drop_last=False)

    log.info(
        f"DataLoaders ready — "
        f"train={len(train_ds)} samples, "
        f"val={len(val_ds)} samples, "
        f"workers={config.num_workers}"
    )

    return train_loader, val_loader, train_ds
