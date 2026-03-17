"""
Builds CTRGCNForAVA with pretrained weights.
Handles Phase 1 → Phase 2 head rebuild when num_classes differs.
"""
import torch, logging
from models.ctrgcn_ava import CTRGCNForAVA
log = logging.getLogger("model_factory")

def build_model(config) -> CTRGCNForAVA:
    pretrained_sd = None
    if config.pretrained_ckpt and config.pretrained_ckpt != "auto":
        ckpt = torch.load(config.pretrained_ckpt, map_location="cpu",
                          weights_only=False)
        pretrained_sd = ckpt.get("state_dict",
                        ckpt.get("model_state_dict", ckpt))

        # Detect stored num_classes; rebuild head if mismatch
        for k, v in pretrained_sd.items():
            if "head" in k and "weight" in k and v.ndim == 2:
                stored_classes = v.shape[0]
                if stored_classes != config.num_classes:
                    log.warning(
                        f"num_classes mismatch: checkpoint={stored_classes}, "
                        f"config={config.num_classes}. Head will be re-initialised.")
                    # Drop head weights; backbone weights will still load
                    pretrained_sd = {k: v for k, v in pretrained_sd.items()
                                     if "head" not in k}
                break

    model = CTRGCNForAVA(
        num_classes         = config.num_classes,
        pretrained_state_dict = pretrained_sd,
        dropout             = config.dropout,
    ).to(config.device)

    # Freeze backbone initially if gradual fine-tuning
    if config.finetune_mode == "gradual":
        model.freeze_backbone()
        log.info("Backbone frozen. Unfreeze schedule will apply during training.")

    # torch.compile — ~20–30% throughput on Ada Lovelace
    if config.use_compile and "cuda" in config.device:
        try:
            model = torch.compile(model, mode=config.compile_mode)
            log.info("torch.compile enabled")
        except Exception as e:
            log.warning(f"torch.compile failed ({e}), continuing without it")

    total  = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {total/1e6:.1f}M params total, {active/1e6:.1f}M trainable")
    return model
