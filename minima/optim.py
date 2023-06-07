# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/optim.ipynb.

# %% auto 0
__all__ = ['Optimizer', 'SGD', 'Adam']

# %% ../nbs/optim.ipynb 2
"""Optimization module"""
import minima as mi
from .nn import Parameter
from .autograd import Tensor
from . import init
import numpy as np

# %% ../nbs/optim.ipynb 3
class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def zero_grad(self):
        for p in self.params:
            p.grad = None

# %% ../nbs/optim.ipynb 4
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, wd=0.0):
        super().__init__(params)

        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.wd = wd

    def step(self):
        for self.idx, p in enumerate(self.params):
            self._reg_step(p)
            self._opt_step(p)
            
                
    def _opt_step(self, p):
        if self.idx not in self.u:
            self.u[self.idx] = init.zeros(*p.shape)
        self.u[self.idx] = self.momentum * self.u[self.idx] + (1 - self.momentum) * p.grad.data
        p.data = p.data - self.lr * self.u[self.idx]

    def _reg_step(self, p):
        if self.wd != 0:
            p.data *= (1 - self.lr * self.wd)
        # all same :3
        # p.data *= (1 - self.lr * self.weight_decay)
        # p.data = p.data - self.lr * self.weight_decay * p.data
        # p.data -= self.lr * self.weight_decay * p.data
    
    def zero_grad(self):
        for p in self.params:
            p.grad = None

# %% ../nbs/optim.ipynb 7
class Adam(Optimizer):
    def __init__(
        self,
        params, # `params` is the list of parameters
        lr=0.01, # `lr` is the learning rate $\alpha$
        beta1=0.9, #
        beta2=0.999, #
        eps=1e-8, # `eps` is $\hat{\epsilon}$ or $\epsilon$ based on `optimized_update`
        weight_decay=0.0, # is an instance of class `WeightDecay` defined in [`__init__.py`](index.html)
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = weight_decay
        self.t = 0

        self.exp_avg = {}
        self.exp_avg_sq = {}

    def step(self):
        for self.idx, p in enumerate(self.params):
            self._reg_step(p)
            self._opt_step(p)
            
                
    def _opt_step(self, p):
        if self.idx not in self.exp_avg:
            self.exp_avg[self.idx] = init.zeros(*p.shape)
            self.exp_avg_sq[self.idx] = init.zeros(*p.shape)
        
        # Update biased first and second moment estimates
        self.exp_avg[self.idx] = self.beta1 * self.exp_avg[self.idx] + (1 - self.beta1) * p.grad.data
        self.exp_avg_sq[self.idx] = self.beta2 * self.exp_avg_sq[self.idx] + (1 - self.beta2) * p.grad.data**2
        
        # Compute bias-corrected first and second moment estimates
        exp_avg_hat = self.exp_avg[self.idx] / (1 - self.beta1 ** (self.idx + 1))
        exp_avg_sq_hat = self.exp_avg_sq[self.idx] / (1 - self.beta2 ** (self.idx + 1))
        p.data = p.data - self.lr * exp_avg_hat / (exp_avg_sq_hat ** 0.5 + self.eps)

    def _reg_step(self, p):
        if self.wd != 0:
            p.data *= (1 - self.lr * self.wd)
        # all same :3
        # p.data *= (1 - self.lr * self.weight_decay)
        # p.data = p.data - self.lr * self.weight_decay * p.data
        # p.data -= self.lr * self.weight_decay * p.data
