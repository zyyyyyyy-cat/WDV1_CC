"""Exponential moving average of model weights.

The EMA model is an evaluation-only copy.  It keeps the same state-dict keys
as the training model, so it can be loaded with the normal test entry point.
"""
import copy

import torch


class ModelEma:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, ema_v in ema_state.items():
            v = model_state[key].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(v, alpha=1.0 - d)
            else:
                ema_v.copy_(v)

    def state_dict(self):
        return self.module.state_dict()
