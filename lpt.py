import torch
from torch import nn

from gpt import GPT_CA_SF
from unet1d import Unet1D

class MLP(nn.Module):
    def __init__(self, args, mol_property):
        super(MLP, self).__init__()
        self.args = args
        self.prior_dim = args.latent_dim * args.n_latent
        self.mlp_hidden_dim = args.mlp_hidden_dim
        self.dropout_rate = args.mlp_dropout_rate
        self.mol_property = mol_property

        self.mlp = nn.Sequential(
            nn.Linear(self.prior_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

        if mol_property in ["qed", "sas"]:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()

    def forward(self, z):
        b = z.shape[0]
        z = z.view(b, -1)
        score = self.final_activation(self.mlp(z))
        return score.squeeze()

class LatentPromptTransformer(nn.Module):
    def __init__(self, args, vocab_size=10000,
                 dec_word_dim=256,
                 n_latent=8,
                 latent_dim=32,
                 max_sequence_length=40):
        super(LatentPromptTransformer, self).__init__()
        self.args = args # z_n_iters, z_with_noise, noise_factor, mol_property, prop_coefficient
        self.embedding_size = dec_word_dim
        self.latent_dim = latent_dim
        self.n_latent = n_latent
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        
        model_config = GPT_CA_SF.get_default_config() # model_type = 'gpt-nano'
        model_config.vocab_size = self.vocab_size
        model_config.block_size = self.max_sequence_length
        self.dec_trans = GPT_CA_SF(model_config)

        self.dim = latent_dim * n_latent
        self.channel = 1
        self.unet = Unet1D(dim=16, channels=1, dim_mults=(1, 2, 4, 8))
        self.property_mlps = nn.ModuleDict({p: MLP(args, p) for p in args.mol_property})

    @property
    def device(self):
        return next(self.parameters()).device

    def unet_forward(self, z_0):
        z = z_0.view(-1, self.channel, self.latent_dim) # batch * n_latent, 1, latent_dim
        z = self.unet(z)
        z = z.view(-1, self.n_latent * self.channel, self.latent_dim)
        return z

    def forward(self, x, z, targets=None, inference=False):
        z = self.unet_forward(z)
        if self.args.debug:
            print('z: mean={:.6f}, var={:.6f}, norm={:.6f}'.format(z.mean().item(), z.var().item(), z.norm().item()))
        logits, loss = self.dec_trans(x, z, targets=targets, inference=inference)
        return logits, loss

    def sample_p_0(self, x):
        # x should be an int indicating num batches, or a tensor with batch dimension B
        return torch.randn(*[x.shape[0] if hasattr(x, 'shape') else x, self.channel*self.n_latent, self.latent_dim], device=self.device, requires_grad=False)

    def infer_z(self, z, x, targets=None, y=None, step_size=0.3, debug=False):
        if z is None:
            z = self.sample_p_0(x)

        args = self.args
        z_f_grads_norm = []
        z_nll_grads_norm = []

        for i in range(args.z_n_iters):
            z = z.detach().requires_grad_(True)
            assert z.grad is None

            _, nll = self.forward(x, z, targets=targets, inference=True) # logits, nll

            z_grad_nll = torch.autograd.grad(nll, z)[0]
            _z_grad_nll = z_grad_nll.detach()
            z = z - 0.5 * step_size * step_size * (z_grad_nll + z)

            if args.z_with_noise:
                z += args.noise_factor * step_size * torch.randn_like(z)

            if y:
                z_prior = self.unet_forward(z)
                mse = 0
                for p, yval in y.items():
                    y_z = self.property_mlps[p](z_prior)
                    mse += nn.functional.mse_loss(y_z, yval, reduction='sum')
                z_grad_mse = torch.autograd.grad(mse, z)[0]
                z = z - 0.5 * step_size * step_size * z_grad_mse * args.prop_coefficient

            z_nll_grads_norm.append(torch.norm(_z_grad_nll, dim=1).mean().cpu().numpy())

            if debug:
                print('iter={:2d}, nll={:.6f}, grad_norm={:.6f}'.format(
                    i, nll.cpu().item(), torch.norm(_z_grad_nll, dim=1).mean().cpu().numpy()))

        z = z.detach()

        return z, (z_f_grads_norm, z_nll_grads_norm)

    def infer_z_given_y(self, z, y, n_iter=20, step_size=0.8):
        args = self.args
        z_mse_grads_norm = []
        
        for i in range(n_iter):
            z = z.detach().requires_grad_()
            assert z.grad is None

            z_prior = self.unet_forward(z)
            mse = 0
            for p, yval in y.items():
                y_z = self.property_mlps[p](z_prior)
                mse += nn.functional.mse_loss(y_z, yval, reduction='sum')
            z_grad_mse = torch.autograd.grad(mse, z)[0]
            _z_grad_mse = z_grad_mse.detach()
            z = z - 0.5 * step_size * step_size * (z_grad_mse * args.prop_coefficient + z)

            if args.z_with_noise:
                z += args.noise_factor * step_size * torch.randn_like(z)
            
            z_mse_grads_norm.append(torch.norm(_z_grad_mse, dim=1).mean().cpu().numpy())

        z = z.detach()
        return z, (z_mse_grads_norm)

    def generate(self, sos_idx=33, max_new_tokens=None, x=None, z=None, do_sample=False, top_k=None):
        if z is None:
            z = self.sample_p_0(x)
        z = self.unet_forward(z)
        # unconditional generation only uses idx, pass in x as int
        idx = torch.ones([x.shape[0] if hasattr(x, 'shape') else x, 1], dtype=torch.long, device=self.device) * sos_idx
        # conditional generation, assuming that x has shape [BATCH, SEQ_LEN] and the sos token is not already prepended
        if hasattr(x, 'shape'):
            max_new_tokens -= x.shape[-1] # length of the prefix sequence we condition on
            idx = torch.concatenate([idx, torch.as_tensor(x, dtype=torch.long, device=self.device)], dim=1)
        x_hat = self.dec_trans.generate(idx=idx, z=z, max_new_tokens=max_new_tokens, temperature=1.0, do_sample=do_sample, top_k=top_k)
        return x_hat
