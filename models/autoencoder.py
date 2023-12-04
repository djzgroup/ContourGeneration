import torch
from torch.nn import Module

from .encoders import *
from .diffusion import *


class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = PointEncoder(args,zdim=args.latent_dim)
        # self.encoder = PointNetEncoder(zdim=args.latent_dim)
        self.diffusion = DiffusionPoint(
            net = Embed(point_dim=3, context_dim=args.latent_dim, residual=args.residual),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        x=x.float()
        code, _= self.encoder(x)
        return code

    def decode(self, code, num_points, flexibility=0.0, ret_traj=False):
        return self.diffusion.sample(num_points, code, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x,raw_x):
        x=x.float()
        raw_x=raw_x.float()
        code = self.encode(raw_x)
        loss = self.diffusion.get_loss(x, code)
        return loss
