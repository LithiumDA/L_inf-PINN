import torch

class HJBEquation:
    def __init__(self, x_dim, T, mu) -> None:
        self.xdim = x_dim
        self.mu = mu
        self.T = T
        self.sqrt_2 = 2**0.5

    def domain_loss(self, X, f, sample_cnt=None):
        # dt: (batch, 1)
        # dx: (batch, 100)
        # dx2: (batch, 100)
        dt = f.dt(X, sample_cnt)
        dx2 = f.dx2(X, sample_cnt)
        dx = f.dx(X, sample_cnt)

        residual = dt.squeeze(1) + torch.sum(dx2, dim=1) - self.mu*torch.sum(dx**2, dim=1)
        loss = torch.mean(residual**2)
        return loss

    def boundary_loss(self, X, f, sample_cnt=None):
        # terminal state: g(x) = log((1+x**2)/2)
        # (batch, 1)
        y = f(X, sample_cnt=sample_cnt).squeeze(1)
        x = X[:, :-1]
        gt = torch.log((1+ torch.sum(x**2, dim=1))/2)
        return torch.mean((y-gt)**2)

    def spatial_boundary_loss(self, X, f, sample_cnt=None):
        # terminal state: g(x) = log((1+x**2)/2)
        # (batch, 1)
        y = f(X, sample_cnt=sample_cnt).squeeze(1)
        gt = self.ground_truth(X, sample_cnt)
        return torch.mean((y-gt)**2)

    def ground_truth(self, X, sample_cnt):
        batch_size = X.shape[0]
        # x.shape: (1, batch_size, 100)
        # t.shape: (1, batch_size, 1)
        x, t = X[:, :-1].unsqueeze(0), X[:, -1:].unsqueeze(0)

        # w ~ (0, sqrt(t))
        sample_w = torch.normal(mean=0, std=1.0, size=(sample_cnt, batch_size, self.xdim), 
                                device=x.device, dtype=x.dtype)*torch.sqrt(self.T-t)
        # sample_x.shape: (sample_cnt, batch_size, 100)
        sample_x = x + self.sqrt_2*sample_w
        # (sample_cnt, batch_size)
        sample_x2 = (1+torch.sum(sample_x**2, dim=2))/2
        g = torch.log(sample_x2)
        # Note: if mu is large, it is possible the value diminishes to zero, so it would be better to use float64
        # (batch_size,)
        E = torch.mean(torch.exp(-self.mu*g), dim=0)
        u = -torch.log(E)/self.mu
        return u
