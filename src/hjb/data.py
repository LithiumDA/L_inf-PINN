import torch

class HJBDataset():
    def __init__(self, domain_bsz, bound_bsz, xdim=250, T=1.0, rank=0):
        self.domain_bsz = domain_bsz
        self.bound_bsz = bound_bsz
        self.xdim = xdim
        self.T = T
        self.rank = rank

    def get_online_data(self):
        domain_X = torch.concat(
            [torch.randn((self.domain_bsz, self.xdim), device=self.rank), # x ~ N(0,1)
             torch.rand((self.domain_bsz, 1), device=self.rank)*self.T, # t ~ U(0,T)
            ],
            dim=1
        )
        
        boundary_X = torch.concat(
            [torch.randn((self.domain_bsz, self.xdim), device=self.rank), # x ~ N(0,1)
             torch.ones((self.domain_bsz, 1), device=self.rank)*self.T, # t = T
            ],
            dim=1
        )

        return domain_X, boundary_X