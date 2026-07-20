"""Central configuration for FedNeMo.

All tunables live here. Values were chosen to fit the hard constraints:
  - Single RTX 4050 laptop GPU, total project VRAM must stay < 3 GB.
  - Nemotron-Mini-4B-Instruct loaded once, 4-bit NF4 body on GPU, 256k-vocab
    embed_tokens + lm_head kept on CPU (float32) to keep the vocab projection
    off-GPU. Verified peak training footprint ~2.3 GB.
  - Clients train strictly sequentially (one at a time) - never in parallel.
"""
from __future__ import annotations

import os

# Reduce CUDA reserved-memory fragmentation so "reserved" tracks "allocated"
# closely (the large, variable-size lm_head backward tensors otherwise fragment
# the caching allocator). Must be set before the CUDA context is created.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# This file lives at e:\FedNeMo\v1\fednemo\config.py  ->  V1_DIR = e:\FedNeMo\v1
V1_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = V1_DIR / "Datasets"

PTBXL_DIR = DATASETS_DIR / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"
PTBXL_DATABASE_CSV = PTBXL_DIR / "ptbxl_database.csv"
PTBXL_SCP_CSV = PTBXL_DIR / "scp_statements.csv"
UCI_CSV = DATASETS_DIR / "heart_disease_uci.csv"

# Local Nemotron checkpoint (full HF weights, already downloaded).
MODEL_PATH = os.environ.get("FEDNEMO_MODEL_PATH", r"D:\Code\models\Nemotron-Mini-4B-Instruct")

# Artifacts (trained global adapter, logs) written here.
ARTIFACTS_DIR = V1_DIR / "artifacts"

# --------------------------------------------------------------------------- #
# Federation
# --------------------------------------------------------------------------- #
NUM_CLIENTS = 5
NUM_ROUNDS = 3
DIRICHLET_ALPHA = 0.4          # in [0.3, 0.5] per spec -> strong non-IID skew
MIN_RECORDS_PER_CLIENT = 40    # every shard must be large enough to fine-tune
SEED = 42

# --------------------------------------------------------------------------- #
# Local training (per client, per round)
# --------------------------------------------------------------------------- #
LOCAL_EPOCHS = 1
LOCAL_MAX_STEPS = 0            # 0 => no cap: train on the FULL shard each round
LOCAL_BATCH_SIZE = 1          # forward batch (manual split uses 1)
GRAD_ACCUM_STEPS = 4          # accumulate grads over N records -> effective batch=N
SHUFFLE_EACH_ROUND = True     # shuffle each node's records every round (coverage)
LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 256
# Class-weighted loss: scale each sample's loss by inverse global class frequency
# (helps minority classes: PTB-XL NORM 9257 vs HYP 1309). Off by default so a
# baseline can be measured before/after (see gap-analysis item 8).
CLASS_WEIGHTED_LOSS = False

# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
LORA_RANK = 16          # optimal for ~5.85GB VRAM + ~1k records (r=8->16 doubles
LORA_ALPHA = 32         # capacity, ~+90MB VRAM, low overfit risk; r=32 overfits small data)
LORA_DROPOUT = 0.0
# NemotronForCausalLM is a dense transformer (squared-ReLU MLP, no gate_proj).
LORA_TARGET_MODULES: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]

# --------------------------------------------------------------------------- #
# FedRand (randomized LoRA subparameter split)
# --------------------------------------------------------------------------- #
FEDRAND_SHARE_PROB = 0.5       # rho: P(share A) per layer per round

# --------------------------------------------------------------------------- #
# Differential privacy (Laplace mechanism, NOT Gaussian)
# --------------------------------------------------------------------------- #
DP_CLIP_NORM = 1.0             # C: clip bound on the public matrix update (absolute mode)
DP_EPSILON = 4.0               # per-round privacy budget for the shared matrix
DP_CLIP_TYPE = "l1"            # "l1" (formally correct for Laplace) or "l2"
DP_DELTA = 1e-5                # delta for (eps_total, delta)-DP conversion
DP_EPSILON_MAX = None          # optional cumulative-epsilon budget ceiling (RDP)
# DP calibration mode:
#   "relative" (default, the repaired engine): per-element Laplace noise scaled to
#     a fraction (DP_NOISE_RATIO) of the update's own per-element magnitude (RMS),
#     with per-element clipping at DP_CLIP_MULT x RMS. Keeps signal-to-noise sane
#     so the private model stays functional. This is a heuristic (signal-relative)
#     DP calibration, not pure (eps,0)-DP - see README privacy notes.
#   "absolute": the original C/epsilon mechanism (formally (eps,0)-DP but easily
#     miscalibrated; with C=1.0 it destroys LoRA weights of magnitude ~0.06).
DP_MODE = "relative"
DP_NOISE_RATIO = 0.25          # relative mode: noise std ~= 25% of signal (SNR ~4:1)
DP_CLIP_MULT = 4.0             # relative mode: clip per-element to +/- 4 x RMS
# Distributed DP: the DP guarantee is defined on the AGGREGATE. Nodes add
# independent noise (per-transmission privacy preserved); averaging N nodes
# reduces the noise the GLOBAL model sees by ~sqrt(N). When on, we report the
# effective aggregate noise so the utility benefit is explicit/honest.
DP_DISTRIBUTED = True

# --------------------------------------------------------------------------- #
# Adaptive quantization (fixed 2-bit width, adaptive per-tensor scale/zero-point)
# --------------------------------------------------------------------------- #
QUANT_BITS = 2

# --------------------------------------------------------------------------- #
# Device placement / VRAM budget
# --------------------------------------------------------------------------- #
# GPU VRAM budget in GiB. With ~4.8 GB we can keep lm_head on the GPU (4-bit),
# avoiding the slow CPU 256k-vocab matmul and speeding up ~3-5x. embed_tokens
# stays on CPU regardless (a cheap gather; moving it to GPU costs ~1.5 GB bf16).
GPU_BUDGET_GB = 4.8
# "cuda" -> lm_head on GPU as 4-bit (fast, ~4.4 GB peak).
# "cpu"  -> lm_head on CPU float32 (slow matmul, ~2.4 GB peak).
LM_HEAD_DEVICE = "cuda"

# --------------------------------------------------------------------------- #
# Inference (nemotron-parse hosted API - the ONLY network call in the system)
# --------------------------------------------------------------------------- #
NEMOTRON_PARSE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NEMOTRON_PARSE_MODEL = "nvidia/nemoretriever-parse"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"


@dataclass
class FedNeMoConfig:
    """Runtime config bundle (importable snapshot of the constants above)."""
    num_clients: int = NUM_CLIENTS
    num_rounds: int = NUM_ROUNDS
    dirichlet_alpha: float = DIRICHLET_ALPHA
    min_records_per_client: int = MIN_RECORDS_PER_CLIENT
    seed: int = SEED

    local_epochs: int = LOCAL_EPOCHS
    local_max_steps: int = LOCAL_MAX_STEPS
    local_batch_size: int = LOCAL_BATCH_SIZE
    grad_accum_steps: int = GRAD_ACCUM_STEPS
    shuffle_each_round: bool = SHUFFLE_EACH_ROUND
    learning_rate: float = LEARNING_RATE
    max_seq_len: int = MAX_SEQ_LEN
    class_weighted_loss: bool = CLASS_WEIGHTED_LOSS

    lora_rank: int = LORA_RANK
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]
    )

    fedrand_share_prob: float = FEDRAND_SHARE_PROB
    dp_clip_norm: float = DP_CLIP_NORM
    dp_epsilon: float = DP_EPSILON
    dp_clip_type: str = DP_CLIP_TYPE
    dp_delta: float = DP_DELTA
    dp_epsilon_max: object = DP_EPSILON_MAX
    dp_mode: str = DP_MODE
    dp_noise_ratio: float = DP_NOISE_RATIO
    dp_clip_mult: float = DP_CLIP_MULT
    dp_distributed: bool = DP_DISTRIBUTED
    quant_bits: int = QUANT_BITS

    gpu_budget_gb: float = GPU_BUDGET_GB
    lm_head_device: str = LM_HEAD_DEVICE

    model_path: str = MODEL_PATH


CONFIG = FedNeMoConfig()
