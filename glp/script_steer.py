from baukit import TraceDict
import einops
import json
import pandas as pd
import torch
import transformers

from glp import flow_matching

# =========================
#   Diffusion Functions
# =========================
class OnManifoldPostprocessor:
    """Callable GLP postprocessor with explicit optional tracing state."""

    def __init__(
        self,
        model,
        u=0.5,
        num_timesteps=20,
        layer_idx=None,
        record_direction_trace=False,
        max_denoising_steps=None,
        guidance_gamma=0.0,
    ):
        self.model = model
        self.u = u
        self.num_timesteps = num_timesteps
        self.layer_idx = layer_idx
        self.record_direction_trace = record_direction_trace
        self.max_denoising_steps = max_denoising_steps
        if guidance_gamma < 0:
            raise ValueError("guidance_gamma must be non-negative")
        self.guidance_gamma = guidance_gamma
        self.direction_traces = []

    @property
    def uses_intervention_context(self):
        return self.record_direction_trace or self.guidance_gamma > 0

    def clear_direction_traces(self):
        self.direction_traces.clear()

    def _to_activation_space(self, latents, *, has_seq_dim, batch_size):
        acts = self.model.normalizer.denormalize(
            latents, layer_idx=self.layer_idx
        ).float()
        if has_seq_dim:
            return einops.rearrange(acts, "(b s) 1 d -> b s d", b=batch_size)
        return einops.rearrange(acts, "b 1 d -> b d")[:, None, :]

    def __call__(self, acts_edit, reference_acts=None, steering_vector=None):
        self.model.scheduler.set_timesteps(self.num_timesteps)
        has_seq_dim = len(acts_edit.shape) == 3
        b = acts_edit.shape[0]
        latents = acts_edit
        if has_seq_dim:
            latents = einops.rearrange(latents, "b s d -> (b s) 1 d")
        else:
            latents = einops.rearrange(latents, "b d -> b 1 d")
        latents = self.model.normalizer.normalize(latents, layer_idx=self.layer_idx)
        edited_latents = latents

        if self.guidance_gamma > 0:
            if reference_acts is None or steering_vector is None:
                raise ValueError(
                    "Velocity guidance requires reference_acts and steering_vector"
                )
            normalized_reference = reference_acts
            if has_seq_dim:
                normalized_reference = einops.rearrange(
                    normalized_reference, "b s d -> (b s) 1 d"
                )
            else:
                normalized_reference = einops.rearrange(
                    normalized_reference, "b d -> b 1 d"
                )
            normalized_reference = self.model.normalizer.normalize(
                normalized_reference, layer_idx=self.layer_idx
            )
            normalized_steering = edited_latents - normalized_reference

            def velocity_transform(x_t, timestep, velocity):
                return flow_matching.guided_velocity(
                    x_t=x_t,
                    t=timestep,
                    velocity=velocity,
                    edited_activation=edited_latents,
                    steering_vector=normalized_steering,
                    gamma=self.guidance_gamma,
                )
        else:
            velocity_transform = None

        noise = torch.randn_like(latents)
        noisy_latents, _, timesteps, _ = flow_matching.fm_prepare(
            self.model.scheduler,
            latents,
            noise,
            u=torch.ones(latents.shape[0]) * self.u,
        )

        step_records = []
        if self.record_direction_trace:
            if reference_acts is None or steering_vector is None:
                raise ValueError(
                    "Direction tracing requires reference_acts and steering_vector"
                )

            reference = reference_acts.detach().float()
            direction = steering_vector.detach().to(reference.device).float()
            if direction.ndim == 1:
                direction = direction[None, None, :]
            elif direction.ndim == 2:
                direction = direction[:, None, :]
            direction = direction.expand(reference.shape[0], reference.shape[1], -1)
            unit_direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            intended_projection = (
                (acts_edit.detach().float() - reference) * unit_direction
            ).sum(dim=-1).mean(dim=-1)

            previous = self._to_activation_space(
                noisy_latents, has_seq_dim=has_seq_dim, batch_size=b
            )

            def trace_step(step_index, timestep, step_latents):
                nonlocal previous
                current = self._to_activation_space(
                    step_latents, has_seq_dim=has_seq_dim, batch_size=b
                )
                step_delta = current - previous
                cumulative_delta = current - reference
                step_norm = step_delta.norm(dim=-1)
                cumulative_norm = cumulative_delta.norm(dim=-1)
                step_projection = (step_delta * unit_direction).sum(dim=-1)
                cumulative_projection = (cumulative_delta * unit_direction).sum(dim=-1)
                step_orthogonal_norm = torch.sqrt(
                    (step_norm.square() - step_projection.square()).clamp_min(0)
                )
                cumulative_orthogonal_norm = torch.sqrt(
                    (cumulative_norm.square() - cumulative_projection.square()).clamp_min(0)
                )
                step_records.append({
                    "step": len(step_records),
                    "scheduler_step": int(step_index),
                    "timestep": float(timestep.detach().float().item()),
                    "step_projection": step_projection.mean(dim=-1).detach().cpu().tolist(),
                    "cumulative_projection": cumulative_projection.mean(dim=-1).detach().cpu().tolist(),
                    "step_norm": step_norm.mean(dim=-1).detach().cpu().tolist(),
                    "cumulative_norm": cumulative_norm.mean(dim=-1).detach().cpu().tolist(),
                    "step_orthogonal_norm": step_orthogonal_norm.mean(dim=-1).detach().cpu().tolist(),
                    "cumulative_orthogonal_norm": cumulative_orthogonal_norm.mean(dim=-1).detach().cpu().tolist(),
                    "step_cosine": (
                        step_projection / step_norm.clamp_min(1e-12)
                    ).mean(dim=-1).detach().cpu().tolist(),
                    "cumulative_cosine": (
                        cumulative_projection / cumulative_norm.clamp_min(1e-12)
                    ).mean(dim=-1).detach().cpu().tolist(),
                    "steering_retention": (
                        cumulative_projection.mean(dim=-1)
                        / intended_projection.clamp_min(1e-12)
                    ).detach().cpu().tolist(),
                })
                previous = current
        else:
            trace_step = None

        latents = flow_matching.sample_on_manifold(
            self.model,
            noisy_latents,
            start_timestep=timesteps[0].item(),
            num_timesteps=self.num_timesteps,
            layer_idx=self.layer_idx,
            step_callback=trace_step,
            max_denoising_steps=self.max_denoising_steps,
            velocity_transform=velocity_transform,
        )
        if self.record_direction_trace:
            self.direction_traces.append(step_records)
        latents = self.model.normalizer.denormalize(
            latents, layer_idx=self.layer_idx
        )
        if has_seq_dim:
            latents = einops.rearrange(latents, "(b s) 1 d -> b s d", b=b)
        else:
            latents = einops.rearrange(latents, "b 1 d -> b d")
        latents = latents.to(device=acts_edit.device, dtype=acts_edit.dtype)
        return latents


def postprocess_on_manifold_wrapper(
    model,
    u=0.5,
    num_timesteps=20,
    layer_idx=None,
    record_direction_trace=False,
    max_denoising_steps=None,
    guidance_gamma=0.0,
):
    """Compatibility factory returning a callable GLP postprocessor."""
    return OnManifoldPostprocessor(
        model=model,
        u=u,
        num_timesteps=num_timesteps,
        layer_idx=layer_idx,
        record_direction_trace=record_direction_trace,
        max_denoising_steps=max_denoising_steps,
        guidance_gamma=guidance_gamma,
    )

# =========================
#    Steering Functions
# =========================
def addition_intervention(w=None, alphas=None, postprocess_fn=None):
    if postprocess_fn is None:
        postprocess_fn = lambda x: x
    def rep_act(output, layer_name, inputs):
        nonlocal w, alphas
        use_tuple = isinstance(output, tuple)
        act = output[0] if use_tuple else output
        if w is not None:
            # move to device
            w = w.to(device=act.device, dtype=act.dtype)
            alphas = alphas.to(device=act.device, dtype=act.dtype)
            # reshape based on if batched / unbatched
            if w.ndim == 1:
                w = w[None, None, :]
            elif w.ndim == 2:
                w = w[:, None, :]
            if alphas.ndim == 1:
                alphas = alphas[:, None, None]
            # only apply to every new generated token
            reference_acts = act[:, [-1], :].clone()
            edited_acts = reference_acts + alphas * w
            if (
                isinstance(postprocess_fn, OnManifoldPostprocessor)
                and postprocess_fn.uses_intervention_context
            ):
                edited_acts = postprocess_fn(
                    edited_acts,
                    reference_acts=reference_acts,
                    steering_vector=alphas * w,
                )
            else:
                edited_acts = postprocess_fn(edited_acts)
            act[:, [-1], :] = edited_acts
        return (act, *output[1:]) if use_tuple else act
    return rep_act

def steer_sweep(hf_model, hf_tokenizer, prefixes, alphas, w, layer, act_norm, postprocess_fn=None, batch_size=100, max_new_tokens=20, temperature=0, seed=42):
    hf_tokenizer.padding_side = "left"
    rows = []
    for i in range(0, len(prefixes), batch_size):
        batch = prefixes[i:i + batch_size]
        batch_text = sum([[text] * len(alphas) for text in batch], [])
        batch_alphas = alphas.repeat(len(batch))
        tracing_directions = (
            isinstance(postprocess_fn, OnManifoldPostprocessor)
            and postprocess_fn.record_direction_trace
        )
        if tracing_directions:
            postprocess_fn.clear_direction_traces()
        generate_with_intervention = generate_with_intervention_wrapper(seed=seed)
        gen_text = generate_with_intervention(
            batch_text,
            hf_model,
            hf_tokenizer,
            layers=[f"model.layers.{layer}"],
            intervention_wrapper=addition_intervention,
            # save interpretable alpha but apply act_norm
            intervention_kwargs={"w": w, "alphas": batch_alphas * act_norm, "postprocess_fn": postprocess_fn},
            generate_kwargs={"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "temperature": temperature},
        )
        batch_rows = [
            {"alpha": a.item(), "prefix": p, "text": t}
            for a, p, t in zip(batch_alphas, batch_text, gen_text)
        ]
        if tracing_directions:
            for batch_index, row in enumerate(batch_rows):
                per_token = []
                for generation_step, diffusion_trace in enumerate(postprocess_fn.direction_traces):
                    per_token.append({
                        "generation_step": generation_step,
                        "diffusion_steps": [
                            {
                                key: value[batch_index] if isinstance(value, list) else value
                                for key, value in step.items()
                            }
                            for step in diffusion_trace
                        ],
                    })
                row["direction_trace_json"] = json.dumps(per_token)
                final_steps = [
                    token["diffusion_steps"][-1]
                    for token in per_token
                    if token["diffusion_steps"]
                ]
                for metric in (
                    "cumulative_projection",
                    "cumulative_norm",
                    "cumulative_orthogonal_norm",
                    "cumulative_cosine",
                    "steering_retention",
                ):
                    row[f"activation_{metric}_mean"] = (
                        sum(step[metric] for step in final_steps) / len(final_steps)
                        if final_steps else None
                    )
        rows += batch_rows
    return pd.DataFrame(rows)

def generate(model, processor, inputs, remove_input=True, **generate_kwargs):
    with torch.no_grad():
        output = model.generate(**inputs, **generate_kwargs)
        if remove_input:
            input_len = inputs["input_ids"].shape[1]
            output = output[:, input_len:]
        output = processor.batch_decode(output, skip_special_tokens=True)
    return output
    
def generate_with_intervention_wrapper(seed=42):
    def generate_with_intervention(text, hf_model, hf_processor, generate_kwargs={"max_new_tokens": 10}, layers=[], intervention_wrapper=None, intervention_kwargs={}, forward_only=False):
        if seed is not None:
            transformers.set_seed(seed)
        inputs = hf_processor(text, return_tensors="pt", padding=True).to(hf_model.device)
        if intervention_wrapper is not None:
            intervention_fn = intervention_wrapper(**intervention_kwargs)
        else:
            intervention_fn = None
        with TraceDict(hf_model, layers=layers, edit_output=intervention_fn) as ret:
            if forward_only:
                outputs = hf_model(**inputs)
                output_text = hf_processor.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
            else:
                output_text = generate(hf_model, hf_processor, inputs, **generate_kwargs)
        return output_text
    return generate_with_intervention
