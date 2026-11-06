"""A true constant-LR scheduler.

Used by A0: continue the 120-epoch solution at its terminal LR with every
parameter group (including bias groups) fixed to the same value.
"""


class ConstantLRScheduler:
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = float(lr)
        self._set()

    def _set(self):
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr

    def step(self, epoch=None):
        self._set()

    def _get_lr(self, epoch=None):
        return [self.lr for _ in self.optimizer.param_groups]
