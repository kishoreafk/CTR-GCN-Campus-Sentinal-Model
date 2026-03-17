import copy, torch, torch.nn as nn

class ModelEMA:
    """Exponential moving average of model weights for more stable evaluation."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema   = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, m_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(m_p.data, alpha=1.0 - self.decay)

    def state_dict(self): return self.ema.state_dict()
    def load_state_dict(self, sd): self.ema.load_state_dict(sd)
