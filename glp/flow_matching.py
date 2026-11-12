import torch
from tqdm import tqdm

from diffusers import FlowMatchEulerDiscreteScheduler

# ==========================
#  Flow Matching Functions
# ==========================
def fm_scheduler():
    return FlowMatchEulerDiscreteScheduler()

def fm_prepare(scheduler, model_input, noise, u=None, generator=None):
    """
    Prepare inputs for flow matching training.
    Reference: https://github.com/huggingface/diffusers/blob/9f48394bf7ab75a43435d3ebb96649665e09c98b/examples/dreambooth/train_dreambooth_lora_flux.py#L1736
    """
    # sanity check
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler), "Only FlowMatchEulerDiscreteScheduler is supported"
    assert model_input.ndim == 3, f"Expected (batch, seq, dim), got shape {model_input.shape}"
    # use uniform weighting_scheme; doesn't implement logit_normal
    if u is None:
        batch_size = model_input.shape[0]
        u = torch.rand(size=(batch_size,), generator=generator)
    indices = (u * len(scheduler.timesteps)).long()
    timesteps = scheduler.timesteps[indices]
    sigmas = scheduler.sigmas[indices].flatten()
    timesteps = timesteps.to(model_input.device)
    sigmas = sigmas.to(model_input.device)
    timesteps = timesteps[:, None, None]
    sigmas = sigmas[:, None, None]
    # interpolate between model_input and noise
    noisy_model_input = (1.0 - sigmas) * model_input.to(sigmas.dtype) + sigmas * noise
    noisy_model_input = noisy_model_input.to(model_input.dtype)
    # the target in flow matching is the "velocity"
    target = noise - model_input
    return noisy_model_input, target, timesteps, {"sigmas": sigmas, "noise": noise, "u": u}

def fm_clean_estimate(scheduler, latents, noise_pred, timesteps):
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler), "Only FlowMatchEulerDiscreteScheduler is supported"
    step_indices = [(scheduler.timesteps == t).nonzero().item() for t in timesteps]
    sigma = scheduler.sigmas[step_indices]
    sigma = sigma.to(device=latents.device, dtype=latents.dtype)
    pred_x0 = latents - sigma * noise_pred
    return pred_x0

def guided_velocity(
    x_t,
    t,
    velocity,
    edited_activation,
    steering_vector,
    gamma=0.2,
):
    """Guide reverse flow toward the edited activation's steering projection.

    All tensors must use the GLP's normalized activation coordinates. The
    minus sign accounts for reverse integration over decreasing sigma when
    the trained velocity target is ``noise - clean_activation``.
    """
    del t  # reserved for timestep-dependent guidance schedules
    direction = torch.nn.functional.normalize(steering_vector, dim=-1)
    target = (edited_activation * direction).sum(dim=-1, keepdim=True)
    current = (x_t * direction).sum(dim=-1, keepdim=True)
    guidance = -gamma * (target - current) * direction
    return velocity + guidance

# ==========================
#   Generic Sampling Code
# ==========================
@torch.no_grad()
def sample(
    model,
    latents,
    num_timesteps=20,
    **kwargs
):
    """
    Generate activations from pure noise.
    We recommend setting `num_timesteps` based on your priorities:
    - 20: moderate quality at fast speed
    - 100: good quality at reasonable speed
    - 1000: best quality for diffusion purists
    """
    model.scheduler.set_timesteps(num_timesteps)
    model.scheduler.timesteps = model.scheduler.timesteps.to(latents.device)
    for i, timestep in tqdm(enumerate(model.scheduler.timesteps)):
        timesteps = timestep.repeat(latents.shape[0], 1)
        noise_pred = model.denoiser(
            latents=latents,
            timesteps=timesteps,
            **kwargs
        )
        latents = model.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
    return latents

@torch.no_grad()
def sample_on_manifold(
    model, 
    latents, 
    num_timesteps=20, 
    start_timestep=None,
    step_callback=None,
    max_denoising_steps=None,
    velocity_transform=None,
    **kwargs
):
    """
    Post-process activations into their on-manifold counterpart.
    See the `sample` function above for recommendations on `num_timesteps`.
    This is essentially the activation-space analogue of SDEdit (Meng et. al., 2022).

    If supplied, ``step_callback`` is called after every applied denoising
    step as ``step_callback(step_index, timestep, latents)``. Timesteps that
    are skipped because they precede ``start_timestep`` do not trigger it.
    """
    start_latents = latents.clone()
    model.scheduler.set_timesteps(num_timesteps)
    applied_steps = 0
    for i, timestep in tqdm(enumerate(model.scheduler.timesteps)):
        if start_timestep is not None and torch.is_tensor(start_timestep):
            # inject original latents until start_timestep
            timestep_mask = start_timestep[:, 0, 0] <= timestep
            latents[timestep_mask] = start_latents[timestep_mask]
        elif start_timestep is not None and timestep > start_timestep:
            continue
        if max_denoising_steps is not None and applied_steps >= max_denoising_steps:
            break
        timesteps = timestep[None, ...]
        noise_pred = model.denoiser(
            latents=latents,
            timesteps=timesteps.repeat(latents.shape[0], 1, 1),
            **kwargs
        )
        if velocity_transform is not None:
            noise_pred = velocity_transform(latents, timesteps, noise_pred)
        latents = model.scheduler.step(noise_pred, timesteps, latents, return_dict=False)[0]
        if step_callback is not None:
            step_callback(i, timestep, latents)
        applied_steps += 1
    return latents
