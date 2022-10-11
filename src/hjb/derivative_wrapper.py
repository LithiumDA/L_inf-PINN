'''
Classes to wrap the neural network function and support the following derivatives:
df_dt
df_dx
d2f_dx2
'''
from typing import Any
import torch
from torch import nn
from .equation import HJBEquation



def build_wrapper(cfg, g):
    w_type = cfg.model.derivative
    if w_type == 'gt':
        wrapper = GroundTruthWrapper(cfg.equation.mu, cfg.equation.T, cfg.equation.x_dim, 
                                     cfg.model.sample_cnt)
    elif w_type == 'pinn':
        wrapper = PinnWrapper(g, cfg.equation.x_dim)
    else:
        raise NotImplementedError
    return wrapper

    
class Wrapper:
    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
      
class PinnWrapper(Wrapper):
    def __init__(self, g:nn.Module, x_dim) -> None:
        super().__init__()
        self.g:nn.Module = g
        self.x_dim = x_dim    

    def eval(self):
        self.g.eval()

    def train(self):
        self.g.train()

    def __call__(self, X, sample_cnt=None):
        with torch.set_grad_enabled(self.g.training):
            return self.g(X)
    
    def dx(self, X, sample_cnt=None):
        # x.shape: (batch_size, x_dim)
        x, t = X[:, :-1], X[:, -1:]
        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.g(X)
        df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        return df_dx

    def dt(self, X, sample_cnt=None):
        # x.shape: (batch_size, x_dim)
        x, t = X[:, :-1], X[:, -1:]
        t.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.g(X)
        df_dx = torch.autograd.grad(f.sum(), t, create_graph=True)[0]
        return df_dx

    def dx2(self, X, sample_cnt=None):
        # x.shape: (batch_size, x_dim)
        x, t = X[:, :-1], X[:, -1:]
        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.g(X)

        df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        d2f_dx2 = []
        for i in range(self.x_dim):
            # (batch_size, 1)
            d2f_dxidxi = torch.autograd.grad(df_dx[:, i].sum(), x, create_graph=True)[0][:, i:i+1]
            d2f_dx2.append(d2f_dxidxi)
        # (batch_size, x_dim)
        d2f_dx2 = torch.concat(d2f_dx2, dim=1)
        return d2f_dx2
        
class GroundTruthWrapper(Wrapper):
    def __init__(self, mu, T, x_dim, sample_cnt) -> None:
        super().__init__()
        self.sample_cnt = sample_cnt
        self.training = True
        self.hjb = HJBEquation(x_dim=x_dim, T=T, mu=mu)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, X, sample_cnt=None) -> Any:
        if sample_cnt is None:
            sample_cnt = self.sample_cnt

        # (batch_size, 1)
        return self.hjb.ground_truth(X, sample_cnt=sample_cnt).unsqueeze(1)

    def dx(self, X, sample_cnt=None):
        if sample_cnt is None:
            sample_cnt = self.sample_cnt

        # x.shape: (batch_size, x_dim)
        x, t = X[:, :-1], X[:, -1:]
        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.hjb.ground_truth(X, sample_cnt)

        df_dx = torch.autograd.grad(f.sum(), x)[0]
        return df_dx

    def dx2(self, X, sample_cnt=None):
        if sample_cnt is None:
            sample_cnt = self.sample_cnt

        # x.shape: (batch_size, x_dim)
        x, t = X[:, :-1], X[:, -1:]
        x.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.hjb.ground_truth(X, sample_cnt)

        df_dx = torch.autograd.grad(f.sum(), x, create_graph=True)[0]
        d2f_dx2 = []
        for i in range(self.hjb.xdim):
            # (batch_size, 1)
            d2f_dxidxi = torch.autograd.grad(df_dx[:, i].sum(), x, retain_graph=True)[0][:, i:i+1]
            d2f_dx2.append(d2f_dxidxi)
        # (batch_size, x_dim)
        d2f_dx2 = torch.concat(d2f_dx2, dim=1)
        return d2f_dx2

    def dt(self, X, sample_cnt=None):
        if sample_cnt is None:
            sample_cnt = self.sample_cnt
        
        # batch_size = X.shape[0]
        # t.shape: (batch_size, 1)
        x, t = X[:, :-1], X[:, -1:]
        t.requires_grad_()
        X = torch.cat([x, t], dim=1)
        f = self.hjb.ground_truth(X, sample_cnt)

        df_dt = torch.autograd.grad(f.sum(), t)[0]
        return df_dt