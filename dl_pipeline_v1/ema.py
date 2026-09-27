"""Optional exponential moving average of model state for validation and inference."""
from copy import deepcopy

import torch


class ModelEMA:
    def __init__(self, model, decay=0.999, update_after_step=0):
        if not 0 <= float(decay) < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        source = model.state_dict()
        shadow = self.shadow.state_dict()
        for name, old in shadow.items():
            new = source[name].detach()
            if old.is_floating_point() and self.steps > self.update_after_step:
                old.lerp_(new, 1.0 - self.decay)
            else:
                old.copy_(new)

    def state_dict(self):
        return {"shadow": self.shadow.state_dict(), "steps": self.steps}

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state["shadow"])
        self.steps = int(state["steps"])
