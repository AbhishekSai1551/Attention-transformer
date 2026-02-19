import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import Transformer
import math

class NoamOpt:
    """Learning rate scheduler from the paper"""
    
    def __init__(self, model_size, warmup_steps, optimizer):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0
    
    def step(self):
        """for updating the learning rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
    
    def rate(self, step=None):
        """calculate the learning rate"""
        if step is None:
            step = self._step
        return (self.model_size ** (-0.5) * 
                min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))