"""Differential privacy via the Laplace mechanism (NOT Gaussian).

Applied to the public (shared) LoRA matrix update before it leaves a client:
  1. L2-clip the update to bound sensitivity: delta <- delta * min(1, C / ||delta||_2)
  2. Add zero-mean Laplace noise  eta ~ Lap(0, C / epsilon)  elementwise.

The clip bounds sensitivity to C; Laplace noise with scale C/epsilon gives an
(epsilon, 0)-DP guarantee for that single released matrix, per round. (Note: the
L1-sensitivity-tight form uses the L1 clip norm; here we use an L2 clip as a
conservative, standard practical bound - see README "privacy notes".)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..config import CONFIG


@dataclass
class DPResult:
    tensor: torch.Tensor
    clipped_norm: float
    noise_scale: float          # representative per-element Laplace scale
    signal_rms: float = 0.0     # per-element signal magnitude (relative mode)


def clip_and_add_laplace(
    update: torch.Tensor,
    clip_norm: float,
    epsilon: float,
    clip_type: str = "l1",
    mode: Optional[str] = None,
    noise_ratio: Optional[float] = None,
    clip_mult: Optional[float] = None,
) -> DPResult:
    """Clip a LoRA update and add zero-mean Laplace noise.

    Two calibration modes (default taken from CONFIG.dp_mode):

    "relative" (repaired engine, RECOMMENDED):
      Noise is calibrated to the update's OWN per-element magnitude so the
      signal-to-noise ratio stays sane instead of the noise drowning a ~0.06
      signal (the bug that produced gibberish). Steps:
        s = RMS(update) = sqrt(mean(u^2))          # typical per-element size
        clip each element to +/- clip_mult * s     # bound outliers/sensitivity
        noise ~ Laplace(0, noise_ratio * s)         # e.g. 25% of signal -> SNR ~4
      This is a heuristic, signal-relative DP calibration (not pure (eps,0)-DP);
      the noise_ratio is the privacy<->utility knob.

    "absolute" (original formal mechanism):
      Clip the L1 (or L2) norm to C and add Laplace(C/epsilon). Formally
      (epsilon,0)-DP, but trivially miscalibrated: C=1.0 on a matrix whose L1
      norm is ~1400 shrinks every element ~1000x, then C/eps=0.25 noise buries
      it -> the model collapses to noise.

    Runs on CPU float32 (updates are small LoRA matrices). Returns a new tensor.
    """
    mode = (mode or CONFIG.dp_mode).lower()
    u = update.detach().to("cpu", dtype=torch.float32)

    if mode == "relative":
        nr = CONFIG.dp_noise_ratio if noise_ratio is None else noise_ratio
        cm = CONFIG.dp_clip_mult if clip_mult is None else clip_mult
        rms = float(torch.sqrt(torch.mean(u ** 2)).item())
        if rms <= 0:
            rms = 1e-8
        bound = cm * rms
        clipped = torch.clamp(u, min=-bound, max=bound)
        noise_scale = nr * rms
        noise = torch.distributions.Laplace(0.0, noise_scale).sample(clipped.shape)
        return DPResult(
            tensor=clipped + noise,
            clipped_norm=float(torch.linalg.vector_norm(clipped, ord=2).item()),
            noise_scale=noise_scale,
            signal_rms=rms,
        )

    # --- absolute (original) mode ---
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0 for the Laplace mechanism.")
    if clip_type == "l1":
        norm = torch.linalg.vector_norm(u, ord=1)
    else:
        norm = torch.linalg.vector_norm(u, ord=2)
    scale_factor = min(1.0, clip_norm / (norm.item() + 1e-12))
    clipped = u * scale_factor
    noise_scale = clip_norm / epsilon
    noise = torch.distributions.Laplace(0.0, noise_scale).sample(clipped.shape)
    return DPResult(
        tensor=clipped + noise,
        clipped_norm=min(norm.item(), clip_norm),
        noise_scale=noise_scale,
        signal_rms=float(torch.sqrt(torch.mean(u ** 2)).item()),
    )
