"""Attack simulator: a live Gradient Inversion Attack (GIA) DEMO prop.

This is NOT a real optimizer. It is an honest demo: real GIA genuinely succeeds
when it captures a full, unprotected LoRA update (both A and B, full precision,
no noise) and genuinely fails when the update is fragmented by FedRand (only A or
only B). So faking the *outcome* is faithful to the underlying truth, and it lets
the dashboard show the privacy mechanism working without a GPU or a real attack.

Real optimization-based GIA is a post-hackathon 'if time' item; it slots in here
behind the same `attack(update, original_text)` interface.
"""
from __future__ import annotations

import random

_NOISE_GLYPHS = "▓░█▒▀▄▌▐"


def _is_reconstructable(update: dict) -> bool:
    """Real GIA needs the full delta-W = B @ A, i.e. BOTH matrices, clean.

    Returns True only when the captured update is fully unprotected:
      - both A and B present, AND
      - no DP noise (epsilon == 0), AND
      - full precision (bits >= 32)
    """
    keys = update["tensors"].keys()
    has_a = any(k.endswith(".A") for k in keys)
    has_b = any(k.endswith(".B") for k in keys)
    meta = update["meta"]
    return has_a and has_b and meta["epsilon"] == 0.0 and meta["bits"] >= 32


def _garble(text: str, keep_ratio: float = 0.4, seed: int | None = None) -> str:
    """Replace most characters with noise glyphs -> indecipherable output."""
    rng = random.Random(seed)
    return "".join(
        c if (c.isspace() or rng.random() < keep_ratio) else rng.choice(_NOISE_GLYPHS)
        for c in text
    )


def attack(update: dict, original_text: str, seed: int | None = None) -> dict:
    """Attempt reconstruction of the training text from a captured update.

    Returns {"success": bool, "reconstruction": str} so the dashboard can render
    a clear pass/fail. On the unprotected path it returns the original text
    (GIA succeeds); on the protected path it returns garble (GIA fails).
    """
    if _is_reconstructable(update):
        return {"success": True, "reconstruction": original_text}
    return {"success": False, "reconstruction": _garble(original_text, seed=seed)}


if __name__ == "__main__":
    import copy
    import numpy as np
    from .schema import make_update
    from .filters import FedRandFilter

    text = "Patient reports increased thirst and fatigue over two weeks."
    raw = make_update(
        "h0", 0,
        {"layer0.A": np.zeros((8, 64)), "layer0.B": np.zeros((64, 8))},
        num_samples=5,
    )
    unprotected = attack(copy.deepcopy(raw), text, seed=0)
    protected = attack(FedRandFilter()(copy.deepcopy(raw)), text, seed=0)
    assert unprotected["success"] is True
    assert protected["success"] is False
    print("UNPROTECTED ->", unprotected["reconstruction"])
    print("PROTECTED   ->", protected["reconstruction"])
