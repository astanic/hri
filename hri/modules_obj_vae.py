import torch.nn as nn

from modules_common import MLP


class ObjectSlotVAE(nn.Module):
    def __init__(self, n_dims, obj_vae):
        super(ObjectSlotVAE, self).__init__()
        self.n_dims = n_dims
        if obj_vae['net'] == 'linear':
            self.mu_net = nn.Linear(n_dims, n_dims)
            self.logvar_net = nn.Linear(n_dims, n_dims)
        elif obj_vae['net'] == 'mlp':
            self.mu_net = MLP(n_dims, n_dims, n_dims,
                              out_nl=obj_vae['net_non_linearity'],
                              out_bn=obj_vae['net_bn'])
            self.logvar_net = MLP(n_dims, n_dims, n_dims,
                                  out_nl=obj_vae['net_non_linearity'],
                                  out_bn=obj_vae['net_bn'])
        else:
            raise NotImplementedError

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def kl_divergence_mu0_var1(self, mu, logvar):
        kld = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1).mean()
        return kld

    def forward(self, objs):
        mu = self.mu_net(objs)
        logvar = self.logvar_net(objs)
        objs = self.reparametrize(mu, logvar)
        kl_obj = self.kl_divergence_mu0_var1(mu, logvar)
        return objs, kl_obj

