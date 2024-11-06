import torch
import torch.nn as nn
import manifolds


class ToEuclidean(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Euclidean space
    """
    def __init__(self,):
        super(ToEuclidean, self).__init__()

    def forward(self, x):
        return x

class ToSphere(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Poincare space
    """
    def __init__(self,):
        super(ToSphere, self).__init__()

    def forward(self, x):
        x = x - x.mean(dim=0)
        z = x / x.norm(dim=1)[:, None]
        # latent = F.normalize(latent, dim=0)
        # a = torch.arcsin(latent[:,2]).unsqueeze(1)
        # b = torch.arcsin(latent[:,1].unsqueeze(1) / torch.cos(a))
        # latent = torch.cat((a, b), 1)
        return z

class ToSphere_tao(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Poincare space
    """
    def __init__(self, manifold):
        super(ToSphere_tao, self).__init__()
        self.manifold = manifold # 'Sphere'
        self.manifold = getattr(manifolds, self.manifold)()

    def forward(self, x):
        z = self.manifold.exp_map(x)
        return z

class ToPoincare(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Poincare space
    """
    def __init__(self, c, manifold):
        super(ToPoincare, self).__init__()
        self.c = c
        self.manifold = manifold # 'PoincareBall'
        self.manifold = getattr(manifolds, self.manifold)()

    def forward(self, x):
        z = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        return z

class FromPoincare(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Poincare space
    """
    def __init__(self, c, manifold):
        super(FromPoincare, self).__init__()
        self.c = c
        self.manifold = manifold # 'PoincareBall'
        self.manifold = getattr(manifolds, self.manifold)()

    def forward(self, x):
        z = self.manifold.logmap0(x, c=self.c)
        return z

class ToLorentz(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Lorentz space
    """
    def __init__(self, c, manifold):
        super(ToLorentz, self).__init__()
        self.c = c
        self.manifold = manifold # 'Hyperboloid'
        self.manifold = getattr(manifolds, self.manifold)()

    def forward(self, x):
        z = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
        # z = self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c)
        return z