"""
models/nets.py
- StateEncoder: small Transformer over N UAV tokens
- ConditionEncoder: factorized (per-factor tokens + dropout) OR monolithic
- DiffusionDenoiser: predicts noise eps_theta(Y_t, t, state_ctx, cond_ctx)
- MLPBaseline: direct regression, no diffusion (Model A)
"""
import torch
import torch.nn as nn
import math

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_F = 32       # per-factor embedding dim (factorized)
D_C_MONO = 32  # monolithic condition embedding dim


class StateEncoder(nn.Module):
    def __init__(self, d_state, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.embed = nn.Linear(d_state, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                            batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, n_layers)

    def forward(self, S):  # S: (B, N, d_state)
        x = self.embed(S)          # (B, N, d_model)
        x = self.encoder(x)        # self-attention across N UAV tokens
        ctx = x.mean(dim=1)        # pooled context (B, d_model)
        return ctx


class FactorizedConditionEncoder(nn.Module):
    """Each failure factor gets its own small nonlinear MLP (not just a linear
    projection), independently dropped to a learnable null-token during
    training so the model must learn each factor's effect separately, then
    compose them (additive sum) at test time -- including for factor
    combinations never jointly seen in training. The nonlinear per-factor MLP
    plus strictly additive composition makes this a generalized-additive-model
    constraint: it can represent an arbitrary nonlinear effect for each factor
    individually, but structurally cannot represent a cross-factor
    interaction term -- unlike MonolithicConditionEncoder, which processes
    all factors jointly and can fit interactions."""
    def __init__(self, n_factors=4, d_f=D_F, dropout_p=0.2):
        super().__init__()
        self.n_factors = n_factors
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(1, d_f), nn.ReLU(), nn.Linear(d_f, d_f))
            for _ in range(n_factors)
        ])
        self.null_token = nn.Parameter(torch.zeros(n_factors, d_f))
        self.dropout_p = dropout_p
        self.out_dim = d_f

    def forward(self, C, training=True):  # C: (B, n_factors)
        B = C.shape[0]
        toks = []
        for k in range(self.n_factors):
            e_k = self.proj[k](C[:, k:k+1])                # (B, d_f)
            if training and self.dropout_p > 0:
                mask = (torch.rand(B, 1, device=C.device) > self.dropout_p).float()
                e_k = mask * e_k + (1 - mask) * self.null_token[k]
            toks.append(e_k)
        return torch.stack(toks, dim=0).sum(dim=0)  # additive composition -> (B, d_f)


class MonolithicConditionEncoder(nn.Module):
    """Baseline: all failure factors processed jointly through a nonlinear MLP
    (matched nonlinearity to FactorizedConditionEncoder, so the two encoders
    differ in whether cross-factor interactions are representable, not in
    whether either one is linear) -- this is what Model B uses."""
    def __init__(self, n_factors=4, d_c=D_C_MONO):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_factors, d_c), nn.ReLU(), nn.Linear(d_c, d_c))
        self.out_dim = d_c

    def forward(self, C, training=True):
        return self.net(C)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionDenoiser(nn.Module):
    """eps_theta(Y_t, t, state_ctx, cond_ctx) -> predicted noise, same shape as Y."""
    def __init__(self, d_out, d_state_ctx, d_cond_ctx, d_hidden=256, t_dim=32):
        super().__init__()
        self.t_dim = t_dim
        in_dim = d_out + d_state_ctx + d_cond_ctx + t_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, Y_t, t, state_ctx, cond_ctx):
        temb = timestep_embedding(t, self.t_dim)
        x = torch.cat([Y_t, state_ctx, cond_ctx, temb], dim=-1)
        return self.net(x)


class MLPBaseline(nn.Module):
    """Model A: direct regression S,C -> Y (no diffusion, single forward pass)."""
    def __init__(self, d_state, n_uav, n_factors, d_out, d_hidden=256):
        super().__init__()
        in_dim = n_uav * d_state + n_factors
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, S, C):  # S: (B,N,d_state), C: (B,n_factors)
        B = S.shape[0]
        x = torch.cat([S.reshape(B, -1), C], dim=-1)
        return self.net(x)
