"""
models/diffusion_utils.py
Minimal DDPM scheduler: linear beta schedule, forward noising, reverse sampling loop.
"""
import torch

class DDPMScheduler:
    def __init__(self, T=100, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, Y0, t, noise):
        # Y0: (B, D), t: (B,) long, noise: (B, D)
        ab = self.alpha_bars[t].unsqueeze(-1)               # (B,1)
        return torch.sqrt(ab) * Y0 + torch.sqrt(1 - ab) * noise

    @torch.no_grad()
    def sample(self, denoiser, state_ctx, cond_ctx, shape, device, n_steps=None,
               clip_range=(-1.0, 2.0)):
        """Full reverse diffusion loop -> returns Y0 estimate.
        NOTE: DDPM's ancestral update formula assumes CONSECUTIVE timesteps,
        so we always walk every step from T-1 down to 0 (n_steps is ignored --
        kept as an argument only for API compatibility with Model D later,
        which will do single/few-shot sampling differently).

        At each step we derive the implied clean-sample estimate x0_hat from
        eps_pred and clip it to clip_range (targets are one-hot {0,1} plus
        trajectories normalized to ~[0,1]). Without this, a single bad
        eps_pred from an undertrained denoiser compounds over all T reverse
        steps and can diverge to inf/nan by t=0."""
        Y = torch.randn(shape, device=device)
        step_indices = torch.arange(self.T - 1, -1, -1, device=device).long()
        for t_val in step_indices:
            t = torch.full((shape[0],), t_val.item(), device=device, dtype=torch.long)
            eps_pred = denoiser(Y, t, state_ctx, cond_ctx)
            alpha = self.alphas[t_val]
            alpha_bar = self.alpha_bars[t_val]
            alpha_bar_prev = self.alpha_bars[t_val - 1] if t_val > 0 else torch.ones_like(alpha_bar)
            beta = self.betas[t_val]

            x0_hat = (Y - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
            x0_hat = torch.clamp(x0_hat, clip_range[0], clip_range[1])

            coef_x0 = torch.sqrt(alpha_bar_prev) * beta / (1 - alpha_bar)
            coef_Yt = torch.sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
            mean = coef_x0 * x0_hat + coef_Yt * Y
            if t_val > 0:
                noise = torch.randn_like(Y)
                Y = mean + torch.sqrt(beta) * noise
            else:
                Y = mean
        return Y
