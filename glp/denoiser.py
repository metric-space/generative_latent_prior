import einops
from einops import repeat
from huggingface_hub import snapshot_download
from itertools import chain
import math
from omegaconf import OmegaConf
from pathlib import Path
from safetensors.torch import load_file, save_file
import torch
import torch.nn as nn
from types import SimpleNamespace

from glp import flow_matching

# ==========================
#     Normalizer Class
# ==========================
class Normalizer(nn.Module):
    def __init__(self, mean, var):
        super().__init__()
        self.mean = nn.Buffer(mean)
        self.var = nn.Buffer(var)
    
    def get_layer_stat(self, stat, layer_idx=None):
        if stat.ndim > 1 and stat.shape[0] != 1:
            assert layer_idx is not None, "Layer index must be provided for multi-layer normalization"
        if layer_idx is not None and stat.ndim == 2:
            stat = stat[layer_idx]
            if stat.ndim == 1:
                stat = stat[None, None, :]
            elif stat.ndim == 2:
                stat = stat[:, None, :]
            return stat
        else:
            return stat

    def normalize(self, rep, layer_idx=None):
        mean = self.get_layer_stat(self.mean, layer_idx)
        var = self.get_layer_stat(self.var, layer_idx)
        return (rep.to(mean.device) - mean) / torch.sqrt(var)
    
    def denormalize(self, rep, layer_idx=None):
        mean = self.get_layer_stat(self.mean, layer_idx)
        var = self.get_layer_stat(self.var, layer_idx)
        return rep.to(var.device) * torch.sqrt(var) + mean
    
    def check_normalized(self, rep, atol=2.0):
        # the tolerance is lenient to catch egregious cases
        rep_mean = rep.view(-1, rep.shape[-1]).mean(dim=0)
        rep_var = rep.view(-1, rep.shape[-1]).var(dim=0, unbiased=False)
        ref_mean = torch.zeros(rep.shape[-1], device=rep.device, dtype=rep.dtype)
        ref_var = torch.ones(rep.shape[-1], device=rep.device, dtype=rep.dtype)
        is_normalized = torch.isclose(rep_mean, ref_mean, atol=atol).all() and torch.isclose(rep_var, ref_var, atol=atol).all()
        if not is_normalized:
            print(
                f"WARNING: Latents may not be normalized "
                f"(expected mean=0 and var=1, got mean={rep_mean.mean().item():.4f} and var={rep_var.mean().item():.4f}). "
                f"Small deviations are expected, but variances much larger than 1 are unusual."
            )

    @classmethod
    def from_config(cls, rep_statistic):
        rep_statistic_pt = torch.load(rep_statistic, map_location="cpu")
        rep_mean = rep_statistic_pt["mean"]
        rep_var = rep_statistic_pt["var"]
        return cls(rep_mean, rep_var)

    def save_config(self, path):
        path = Path(path)
        torch.save({"mean": self.mean, "var": self.var}, path / f"rep_statistics.pt")

# ==========================
#     Denoiser Classes
# ==========================
def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.    
    Reference: https://github.com/facebookresearch/DiT/blob/ed81ce2229091fd4ecc9a223645f95cf379d582b/models.py#L41
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

class TransformerMLPBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_mlp,
        d_input,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_input = d_input

        self.up_proj = nn.Linear(d_model, d_mlp)
        self.down_proj = nn.Linear(d_mlp, d_model)
        self.gate_proj = nn.Linear(d_model, d_mlp)
        self.time_proj = nn.Linear(d_model, d_mlp)
        self.act = nn.SiLU()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, t_emb):
        resid_x = x
        post_ln_x = self.ln(x)
        # project up
        interm_x = self.up_proj(post_ln_x)
        # start SwiGLU gate
        g = self.gate_proj(post_ln_x)
        # multiplicative timestep conditioning
        t_emb = self.time_proj(t_emb)
        merged = g * t_emb
        # continue SwiGLU gate
        x = self.act(merged) * interm_x
        # project down
        x = self.down_proj(x)
        return x + resid_x

class TransformerMLPDenoiser(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_mlp=1536,
        d_input=1536,
        n_layers=12,
        multi_layer_n_layers=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_input = d_input
        self.n_layers = n_layers
        self.multi_layer_n_layers = multi_layer_n_layers

        self.layers = nn.ModuleList([
            TransformerMLPBlock(
                d_model=d_model,
                d_mlp=d_mlp,
                d_input=d_input
            ) for _ in range(n_layers)
        ])
        self.in_proj = nn.Linear(d_input, d_model)
        self.out_proj = nn.Linear(d_model, d_input)

        self.time_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        if multi_layer_n_layers is not None:
            self.layer_embed = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.layer_embed = nn.Identity()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, latents, timesteps, layer_idx=None, **kwargs):
        assert latents.ndim == 2, f"Expected (batch, dim), got shape {latents.shape}"
        x = latents
        # prepare sinusoidal timestep embedding
        timesteps = timesteps.flatten().to(x.device)
        assert timesteps.shape == (x.shape[0],)
        t_emb = timestep_embedding(timesteps, self.d_model, repeat_only=False)
        emb = self.time_embed(t_emb)
        # prepare sinusoidal layer depth embedding
        layer_depth = None if layer_idx is None else layer_idx.float() / (self.multi_layer_n_layers - 1)
        if layer_depth is not None:
            layer_emb = timestep_embedding(layer_depth, self.d_model, repeat_only=False)
            emb += self.layer_embed(layer_emb)
        # apply MLP blocks
        x = self.in_proj(x)
        for layer in self.layers:
            x = layer(x, emb)
        x = self.ln(x)
        x = self.out_proj(x)
        return x

class Denoiser(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TransformerMLPDenoiser(**kwargs)
        self.device, self.dtype = None, None

    def forward(self, latents, layer_idx=None, **kwargs):
        layer_idx = torch.full((latents.shape[0],), layer_idx, device=latents.device) if isinstance(layer_idx, int) else layer_idx
        # move device and dtype
        device, dtype = latents.device, latents.dtype
        latents = latents.to(device=self.device, dtype=self.dtype)
        # reshape to (batch*seq, dim) 
        # since denoiser does single-token modeling
        b, s, d = latents.shape
        latents = einops.rearrange(latents, "b s d -> (b s) d")
        latents = self.model(latents, layer_idx=layer_idx, **kwargs)
        # reshape back to (batch, seq, dim)
        latents = einops.rearrange(latents, "(b s) d -> b s d", b=b, s=s)
        latents = latents.to(device=device, dtype=dtype)
        return latents
    
    def save_pretrained(self, path, name=None):
        path = Path(path)
        name = name or "mlp"
        save_file(self.state_dict(), path / f"{name}.safetensors")
        
    def load_pretrained(self, path, name=None):
        path = Path(path)
        name = name or "mlp"
        self.load_state_dict(load_file(path / f"{name}.safetensors"))

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        param = next(chain(self.model.parameters(), self.model.buffers()), None)
        self.device = param.device if param is not None else None
        self.dtype = param.dtype if param is not None else None
        return result

# ==========================
#    GLP Wrapper Class
# ==========================
class GLP(nn.Module):
    def __init__(self, normalizer_config, denoiser_config, tracedict_config=None):
        super().__init__()
        self.normalizer = Normalizer.from_config(**normalizer_config)
        self.denoiser = Denoiser(**denoiser_config)
        self.scheduler = flow_matching.fm_scheduler()
        self.tracedict_config = tracedict_config

    def save_pretrained(self, path, name=None):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)
        self.denoiser.save_pretrained(path, name=name)
        self.normalizer.save_config(path)

    def load_pretrained(self, path, name=None):
        path = Path(path)
        self.denoiser.load_pretrained(path, name=name)

    def forward(
        self,
        *,
        latents: torch.FloatTensor,                       # (batch, seq, dim)
        u: torch.FloatTensor | float | None = None,       # (batch,) or scalar
        layer_idx: torch.LongTensor | int | None = None,  # (batch,) or scalar
        loss_kwargs: dict = {},
        generator: torch.Generator | None = None,
        **kwargs
    ) -> SimpleNamespace:
        # prepare extra params
        assert latents.ndim == 3, f"Expected (batch, seq, dim), got shape {latents.shape}"
        self.normalizer.check_normalized(latents)
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)
        u = torch.full((latents.shape[0],), u, device=latents.device) if isinstance(u, float) else u

        # prepare flow matching inputs and target
        noise = torch.randn(latents.shape, dtype=latents.dtype, generator=generator).to(latents.device)
        noisy_latents, target, timesteps, meta = flow_matching.fm_prepare(
            self.scheduler,
            latents,
            noise,
            u=u,
            generator=generator
        )
        # compute denoiser forward pass
        outputs = self.denoiser(
            latents=noisy_latents,
            timesteps=timesteps,
            layer_idx=layer_idx,
            **kwargs
        )
        # compute loss
        loss = torch.nn.functional.mse_loss(outputs, target, **loss_kwargs)
        return SimpleNamespace(
            latents=outputs,
            timesteps=timesteps,
            loss=loss,
        )

def load_glp(weights_folder, device="cuda:0", checkpoint="final"):
    weights_path = Path(weights_folder)
    checkpoint_file = weights_path / f"{checkpoint}.safetensors"
    if weights_path.is_dir() and not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    if not weights_path.is_dir():
        # speed up downloading the main checkpoint
        ignore_patterns = ["checkpoints/*"] if checkpoint == "final" else None
        local_dir = snapshot_download(
            repo_id=weights_folder,
            ignore_patterns=ignore_patterns
        )
        weights_path = Path(local_dir)
        checkpoint_file = weights_path / f"{checkpoint}.safetensors"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    weights_folder = str(weights_path)
    config = OmegaConf.load(f"{weights_folder}/config.yaml")
    config.rep_statistic = f"{weights_folder}/rep_statistics.pt"
    OmegaConf.resolve(config)
    model = GLP(**config.glp_kwargs)
    model.to(device)
    model.load_pretrained(weights_folder, name=checkpoint)
    return model
