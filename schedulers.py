'''A wrapper class for scheduled optimizer '''
import numpy as np
import torch

class WarmupWithFrozenBackboneScheduler:
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_heads, lr_backbone, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_heads = lr_heads
        self.lr_backbone = lr_backbone
        self.n_warmup_steps = n_warmup_steps
        self._step_count = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_heads(self):
        lr_heads = self.lr_heads
        n_steps, n_warmup_steps = self._step_count, self.n_warmup_steps
        return min(lr_heads, lr_heads * n_steps / n_warmup_steps)

    def _get_lr_backbone(self):
        lr_backbone = self.lr_backbone
        if self.n_warmup_steps < self._step_count:
            frozen = 0.
        else:
            frozen = 1.
        return lr_backbone * frozen

    def _update_learning_rate(self):
        self._step_count += 1
        lr_h = self._get_lr_heads()
        lr_b = self._get_lr_backbone()

        for param_group in self._optimizer.param_groups:
            if param_group['is_back']:
                param_group['lr'] = lr_b
            else:
                param_group['lr'] = lr_h
    