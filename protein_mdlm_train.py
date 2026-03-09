#!/usr/bin/env python3
"""
protein_mdlm_train.py — Training script for protein Masked Diffusion LM.

Imports model and shared components from protein_mdlm_core.py.
"""

import argparse
import math
import random
import sys
from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_mdlm_core import (
    AMINO_ACIDS,
    DEVICE,
    DROPOUT,
    DTYPE,
    D_MODEL,
    HEAD_DIM,
    MASK_ID,
    N_DECODER_LAYERS,
    N_LAYERS,
    PAD_ID,
    ROPE_DIM,
    SCHEDULE_EPS,
    VOCAB_SIZE,
    TOKEN_TO_ID,
    ProteinMDLM,
    RMSNorm,
    alpha_cosine,
    encode_sequence,
    maybe_softcap_logits,
)

# ============================================================
# Training-specific config
# ============================================================

T_MIN = 1e-3

RNG_SEED = 1337
TOTAL_STEPS = 5000
TOKENS_PER_BATCH_TARGET = 65536
TOKENS_PER_BATCH_TOL = 1024

ADAM_LR = 3e-4
ADAM_BETAS = (0.9, 0.95)
ADAM_WEIGHT_DECAY = 0.10

MUON_LR = 2e-2
MUON_WEIGHT_DECAY = 0.01
MUON_BETA2 = 0.95
MUON_MOMENTUM_MIN = 0.85
MUON_MOMENTUM_MAX = 0.95
MUON_WARMUP_STEPS = 300
MUON_COOLDOWN_STEPS = 200

LR_COOLDOWN_FRAC = 0.50
LR_FLOOR_MULT = 0.10
LR_PLATEAU_1 = 1.52
LR_PLATEAU_2 = 1.73

DROPOUT = 1 / 128
GRAD_CLIP_NORM = None

USE_LOGIT_SOFTCAP = True
SOFTCAP_CAP = 23.0
SOFTCAP_SHIFT = 5.0
SOFTCAP_SCALE = 7.5

MMD_WEIGHT = 1.0
MMD_BANDWIDTHS_MULT = [0.2, 0.5, 1.0, 2.0, 5.0]

VAL_SPLIT_FRAC = 0.05
EVAL_EVERY = 500
EVAL_BATCHES = 10
EVAL_MASK_RATE = 0.20


# ============================================================
# FASTA loading
# ============================================================

def load_fasta(path: str) -> List[str]:
    """Load protein sequences from FASTA, keeping only canonical-20-AA sequences."""
    valid_aas = set(AMINO_ACIDS)
    sequences: List[str] = []
    parts: List[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    seq = "".join(parts)
                    if seq and all(c in valid_aas for c in seq):
                        sequences.append(seq)
                parts = []
            else:
                parts.append(line.upper())
        if parts:
            seq = "".join(parts)
            if seq and all(c in valid_aas for c in seq):
                sequences.append(seq)

    return sequences


# ============================================================
# MDLM schedule & masking (training only)
# ============================================================

def mdlm_weight(t: torch.Tensor) -> torch.Tensor:
    """ELBO weight: -alpha'(t) / (1 - alpha(t))."""
    alpha = alpha_cosine(t)
    sigma = -torch.log(alpha.clamp(min=1e-8))
    neg_alpha_prime = (1.0 - SCHEDULE_EPS) * (math.pi / 2) * torch.sin((math.pi / 2) * t)
    dsigma = neg_alpha_prime / alpha.clamp(min=1e-8)
    return dsigma / torch.expm1(sigma).clamp(min=1e-8)


def apply_mdlm_masking(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MDLM masking to a packed batch."""
    B = cu_seqlens.shape[0] - 1
    T_total = input_ids.shape[0]
    device = input_ids.device

    t_vals = torch.rand(B, device=device) * (1.0 - T_MIN) + T_MIN
    alpha_vals = alpha_cosine(t_vals)
    mask_rates = 1.0 - alpha_vals

    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_idx = torch.repeat_interleave(torch.arange(B, device=device), lens)

    pos_mask_rate = mask_rates[seq_idx]
    is_masked = torch.rand(T_total, device=device) < pos_mask_rate

    masked_ids = input_ids.clone()
    masked_ids[is_masked] = MASK_ID

    w_vals = mdlm_weight(t_vals)
    weights = torch.zeros(T_total, dtype=torch.float32, device=device)
    weights[is_masked] = w_vals[seq_idx[is_masked]].float()

    return masked_ids, is_masked, weights


# ============================================================
# Optimizers
# ============================================================

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def polar_express(G: torch.Tensor) -> torch.Tensor:
    assert G.ndim == 2
    X = G.to(torch.bfloat16)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True
    X = X / (X.norm() * (1 + 2e-2) + 1e-6)
    for a, b, c in polar_express_coeffs:
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + (B @ X)
    if transposed:
        X = X.t()
    return X


def apply_normuon_variance_reduction(v, buf, beta2, red_dim):
    v_mean = v.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()
    buf.lerp_(v_mean.to(dtype=buf.dtype), 1 - beta2)
    step_size = buf.clamp_min(1e-10).rsqrt_()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
    return v.mul_(final_scale.to(dtype=v.dtype))


class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            m = float(group["momentum"])
            beta2 = float(group["beta2"])
            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue
                g = p.grad
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(g)
                mom = st["mom"]
                mom.lerp_(g, 1 - m)
                v = g.lerp(mom, m)
                v = polar_express(v)
                M, K = p.shape
                red_dim = -1 if (M >= K) else -2
                if "vbuf" not in st:
                    st["vbuf"] = (
                        torch.zeros((M, 1), device=p.device, dtype=torch.float32)
                        if red_dim == -1
                        else torch.zeros((1, K), device=p.device, dtype=torch.float32)
                    )
                v = apply_normuon_variance_reduction(v, st["vbuf"], beta2, red_dim)
                mask = (v * p) >= 0
                if wd != 0.0:
                    p.add_(p * mask, alpha=-(wd * lr * lr))
                p.add_(v, alpha=-lr)
        return loss


def split_params_for_optimizers(model: nn.Module):
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and ("tok_emb" not in name) and ("lm_head" not in name) and ("mask_embed" not in name):
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


def speedrun_lr_factor(step: int, total_steps: int) -> float:
    if step > total_steps:
        return LR_FLOOR_MULT
    x = step / max(1, total_steps)
    lr_max = 1.0
    if x > 1 / 3:
        lr_max = LR_PLATEAU_1
    if x > 2 / 3:
        lr_max = LR_PLATEAU_2
    if x >= 1.0 - LR_COOLDOWN_FRAC:
        w = (1.0 - x) / LR_COOLDOWN_FRAC
        return lr_max * w + (1.0 - w) * LR_FLOOR_MULT
    return lr_max


def muon_momentum_schedule(step: int, total_steps: int) -> float:
    cd_start = total_steps - MUON_COOLDOWN_STEPS
    if step < MUON_WARMUP_STEPS:
        frac = step / max(1, MUON_WARMUP_STEPS)
        return MUON_MOMENTUM_MIN + frac * (MUON_MOMENTUM_MAX - MUON_MOMENTUM_MIN)
    if step > cd_start:
        frac = (step - cd_start) / max(1, MUON_COOLDOWN_STEPS)
        return MUON_MOMENTUM_MAX - frac * (MUON_MOMENTUM_MAX - MUON_MOMENTUM_MIN)
    return MUON_MOMENTUM_MAX


def compute_mmd_imq(z: torch.Tensor, d_model: int) -> torch.Tensor:
    B = z.shape[0]
    prior = torch.randn_like(z)

    zz = torch.cdist(z, z).pow(2)
    pp = torch.cdist(prior, prior).pow(2)
    zp = torch.cdist(z, prior).pow(2)

    mmd = torch.zeros(1, device=z.device, dtype=z.dtype)
    for mult in MMD_BANDWIDTHS_MULT:
        C = mult * d_model
        k_zz = C / (C + zz)
        k_pp = C / (C + pp)
        k_zp = C / (C + zp)
        mmd = mmd + k_zz.mean() + k_pp.mean() - 2.0 * k_zp.mean()

    return mmd.squeeze()


# ============================================================
# Token-band batching
# ============================================================

class DeferredTokenBatcher:
    def __init__(self, sample_lens: List[int], target_tokens: int, tol_tokens: int, rng_seed: int):
        self.sample_lens = sample_lens
        self.N = len(sample_lens)
        self.min_tokens = int(max(1, target_tokens - tol_tokens))
        self.max_tokens = int(target_tokens + tol_tokens)
        self.rng = random.Random(rng_seed)
        self.deferred: deque = deque()

    def next_batch_indices(self) -> List[int]:
        if self.deferred and self.sample_lens[self.deferred[0]] > self.max_tokens:
            return [self.deferred.popleft()]

        batch: List[int] = []
        tok = 0
        spins = 0
        while tok < self.min_tokens:
            if self.deferred and tok + self.sample_lens[self.deferred[0]] <= self.max_tokens:
                idx = self.deferred.popleft()
            else:
                idx = self.rng.randrange(self.N)
            L = self.sample_lens[idx]
            if tok == 0 and L > self.max_tokens:
                return [idx]
            if tok + L <= self.max_tokens:
                batch.append(idx)
                tok += L
            else:
                self.deferred.append(idx)
            spins += 1
            if spins > 200_000:
                return batch if batch else [self.rng.randrange(self.N)]
        return batch


# ============================================================
# Collation
# ============================================================

def collate_packed(
    sequences: List[List[int]],
    sample_indices: List[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    seqs = [sequences[i] for i in sample_indices]
    lens = [len(s) for s in seqs]
    max_seqlen = max(lens)

    flat_input: List[int] = []
    flat_positions: List[int] = []
    for s in seqs:
        flat_input.extend(s)
        flat_positions.extend(range(len(s)))

    cu = [0]
    running = 0
    for L in lens:
        running += L
        cu.append(running)

    input_ids = torch.tensor(flat_input, dtype=torch.long, device=device)
    positions = torch.tensor(flat_positions, dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    return input_ids, positions, cu_seqlens, max_seqlen, int(input_ids.numel())


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def apply_fixed_masking(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    mask_rate: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T_total = input_ids.shape[0]
    is_masked = torch.rand(T_total, device=input_ids.device) < mask_rate
    masked_ids = input_ids.clone()
    masked_ids[is_masked] = MASK_ID
    return masked_ids, is_masked


def _unwrap(model):
    return getattr(model, "_orig_mod", model)


@torch.no_grad()
def _eval_decoder_pass(m, z_dec, seq_idx, input_ids, is_masked, cu_seqlens, max_seqlen, positions, autocast_ctx):
    z_dec_broadcast = z_dec[seq_idx]
    emb_masked = m.mask_embed.weight[1]
    emb_revealed = m.dec_tok_emb(input_ids)
    pos_emb = torch.where(is_masked.unsqueeze(1), emb_masked, emb_revealed)
    pos_emb = m.decoder_input_norm(pos_emb)
    dec_input = z_dec_broadcast + pos_emb
    with autocast_ctx:
        for blk in m.decoder_blocks:
            dec_input = blk(dec_input, cu_seqlens, max_seqlen, positions)
        dec_input = m.norm_f(dec_input)
        logits = m.lm_head(dec_input)
        logits = maybe_softcap_logits(logits)
        logits[:, PAD_ID] = -1e9
        logits[:, MASK_ID] = -1e9
    return logits


@torch.no_grad()
def evaluate(model, val_sequences, val_batcher, d_model, device, autocast_ctx, n_batches=10):
    model.eval()
    m = _unwrap(model)

    total_ce = 0.0
    total_masked_toks = 0
    first_batch = None

    for i in range(n_batches):
        sample_indices = val_batcher.next_batch_indices()
        input_ids, positions, cu_seqlens, max_seqlen, T_total = collate_packed(
            val_sequences, sample_indices, device=device,
        )
        masked_ids, is_masked = apply_fixed_masking(input_ids, cu_seqlens, EVAL_MASK_RATE)

        B = cu_seqlens.shape[0] - 1
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        seq_idx = torch.repeat_interleave(torch.arange(B, device=device), lens)

        with autocast_ctx:
            x = m.tok_emb(masked_ids)
            for blk in m.blocks:
                x = blk(x, cu_seqlens, max_seqlen, positions)

        z_sum = torch.zeros(B, m.d_model, device=device, dtype=x.dtype)
        z_sum.scatter_add_(0, seq_idx.unsqueeze(1).expand_as(x), x)
        z_mean = z_sum / lens.unsqueeze(1).to(x.dtype)
        z_dec = z_mean * m.scalar_dec.to(x.dtype)

        logits = _eval_decoder_pass(
            m, z_dec, seq_idx, input_ids, is_masked,
            cu_seqlens, max_seqlen, positions, autocast_ctx,
        )

        if is_masked.any():
            ce = F.cross_entropy(
                logits[is_masked].float(), input_ids[is_masked], reduction="sum",
            ).item()
            total_ce += ce
            total_masked_toks += int(is_masked.sum().item())

        if i == 0:
            first_batch = (z_mean, seq_idx, input_ids, is_masked,
                           cu_seqlens, max_seqlen, positions)

    val_ce = total_ce / max(total_masked_toks, 1)

    # Shuffle test on first batch
    z_mean, seq_idx, input_ids, is_masked, cu_seqlens, max_seqlen, positions = first_batch
    B = z_mean.shape[0]
    perm = torch.randperm(B, device=device)
    z_dec_shuffled = z_mean[perm] * m.scalar_dec.to(z_mean.dtype)

    logits_shuf = _eval_decoder_pass(
        m, z_dec_shuffled, seq_idx, input_ids, is_masked,
        cu_seqlens, max_seqlen, positions, autocast_ctx,
    )

    shuffle_ce = float("nan")
    if is_masked.any():
        shuffle_ce = F.cross_entropy(
            logits_shuf[is_masked].float(), input_ids[is_masked],
        ).item()

    model.train()
    return {"val_ce": val_ce, "shuffle_ce": shuffle_ce, "delta": shuffle_ce - val_ce}


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Protein Masked Diffusion LM (MDLM)")
    parser.add_argument("fasta_path", type=str, help="Path to input FASTA file")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS, help="Training steps")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--checkpoint", type=str, default="protein_mdlm.pt", help="Output checkpoint path")
    parser.add_argument("--mmd-weight", type=float, default=MMD_WEIGHT, help="MMD loss weight")
    parser.add_argument("--d-model", type=int, default=D_MODEL, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=N_LAYERS, help="Encoder layers")
    parser.add_argument("--n-decoder-layers", type=int, default=N_DECODER_LAYERS, help="Decoder layers")
    parser.add_argument("--head-dim", type=int, default=HEAD_DIM, help="Attention head dimension")
    parser.add_argument("--rope-dim", type=int, default=ROPE_DIM, help="RoPE dimensions per head")
    parser.add_argument("--val-fasta", type=str, default=None, help="Validation FASTA (auto-split if omitted)")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT_FRAC, help="Fraction held out for val if no --val-fasta")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY, help="Evaluate every N steps")
    parser.add_argument("--eval-batches", type=int, default=EVAL_BATCHES, help="Batches per evaluation")
    args = parser.parse_args()

    total_steps = args.steps
    mmd_weight = args.mmd_weight
    d_model = args.d_model
    n_layers = args.n_layers
    n_decoder_layers = args.n_decoder_layers
    head_dim = args.head_dim
    rope_dim = args.rope_dim

    print(f"DEVICE: {DEVICE} | DTYPE: {DTYPE}")

    # ---- Data ----
    print(f"Loading FASTA: {args.fasta_path}")
    sequences_str = load_fasta(args.fasta_path)
    print(f"Loaded {len(sequences_str)} valid sequences (canonical 20 AA only)")
    if not sequences_str:
        print("ERROR: No valid sequences found. Exiting.")
        sys.exit(1)

    all_sequences = [encode_sequence(s) for s in sequences_str]

    # ---- Train / val split ----
    if args.val_fasta:
        val_str = load_fasta(args.val_fasta)
        print(f"Validation FASTA: {args.val_fasta} ({len(val_str)} sequences)")
        val_sequences = [encode_sequence(s) for s in val_str]
        sequences = all_sequences
    else:
        rng_split = random.Random(args.seed)
        rng_split.shuffle(all_sequences)
        n_val = max(1, int(len(all_sequences) * args.val_split))
        val_sequences = all_sequences[:n_val]
        sequences = all_sequences[n_val:]
        print(f"Auto-split: {len(sequences)} train, {len(val_sequences)} val ({args.val_split:.0%})")

    sample_lens = [len(s) for s in sequences]

    lens_t = torch.tensor(sample_lens, dtype=torch.int64)
    p50 = int(lens_t.kthvalue(max(1, int(0.50 * len(lens_t)))).values.item())
    p90 = int(lens_t.kthvalue(max(1, int(0.90 * len(lens_t)))).values.item())
    print(
        f"Train seq lengths: min={int(lens_t.min())}  p50={p50}  "
        f"p90={p90}  max={int(lens_t.max())}"
    )

    # ---- Model ----
    model = ProteinMDLM(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_layers=n_layers,
        n_decoder_layers=n_decoder_layers,
        head_dim=head_dim,
        rope_dim=rope_dim,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.3f}M")

    if DEVICE == "cuda":
        model = torch.compile(model)

    # ---- Optimizers ----
    muon_params, adam_params = split_params_for_optimizers(model)
    adam = torch.optim.AdamW(
        adam_params, lr=ADAM_LR, betas=ADAM_BETAS, weight_decay=ADAM_WEIGHT_DECAY
    )
    muon = NorMuon(
        muon_params, lr=MUON_LR, weight_decay=MUON_WEIGHT_DECAY,
        momentum=MUON_MOMENTUM_MAX, beta2=MUON_BETA2,
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=DTYPE)
        if DEVICE == "cuda"
        else torch.cpu.amp.autocast(enabled=False)
    )

    # ---- Batchers ----
    batcher = DeferredTokenBatcher(
        sample_lens=sample_lens,
        target_tokens=TOKENS_PER_BATCH_TARGET,
        tol_tokens=TOKENS_PER_BATCH_TOL,
        rng_seed=args.seed,
    )
    val_sample_lens = [len(s) for s in val_sequences]
    val_batcher = DeferredTokenBatcher(
        sample_lens=val_sample_lens,
        target_tokens=TOKENS_PER_BATCH_TARGET,
        tol_tokens=TOKENS_PER_BATCH_TOL,
        rng_seed=args.seed + 1,
    )

    # ---- Seed ----
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ---- Training loop ----
    loss_ema = None
    nll_ema = None

    for step in range(total_steps):
        # Schedule
        lr_mult = speedrun_lr_factor(step, total_steps)
        mu_mom = muon_momentum_schedule(step, total_steps)
        for pg in adam.param_groups:
            pg["lr"] = ADAM_LR * lr_mult
        for pg in muon.param_groups:
            pg["lr"] = MUON_LR * lr_mult
            pg["momentum"] = mu_mom

        # Batch
        sample_indices = batcher.next_batch_indices()
        input_ids, positions, cu_seqlens, max_seqlen, T_total = collate_packed(
            sequences, sample_indices, device=DEVICE
        )

        # MDLM masking
        masked_ids, is_masked, weights = apply_mdlm_masking(input_ids, cu_seqlens)

        # Forward
        with autocast_ctx:
            logits, z_mmd = model(masked_ids, cu_seqlens, max_seqlen, positions, is_masked)
            logits = maybe_softcap_logits(logits)

            logits[:, PAD_ID] = -1e9
            logits[:, MASK_ID] = -1e9

            if is_masked.any():
                masked_logits = logits[is_masked]
                masked_targets = input_ids[is_masked]
                masked_weights = weights[is_masked]

                per_tok_ce = F.cross_entropy(masked_logits, masked_targets, reduction="none")
                loss = (per_tok_ce * masked_weights).sum() / T_total
            else:
                loss = logits.sum() * 0.0

            mmd_loss = compute_mmd_imq(z_mmd.float(), d_model)
            loss = loss + mmd_weight * mmd_loss

        # Backward + step
        adam.zero_grad(set_to_none=True)
        muon.zero_grad(set_to_none=True)
        loss.backward()

        if GRAD_CLIP_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        adam.step()
        muon.step()

        # Logging
        loss_val = float(loss.detach().cpu().item())
        mmd_val = float(mmd_loss.detach().cpu().item())
        n_masked = int(is_masked.sum().item())
        loss_ema = loss_val if loss_ema is None else (0.98 * loss_ema + 0.02 * loss_val)

        with torch.no_grad():
            B = cu_seqlens.shape[0] - 1
            lens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
            seq_idx = torch.repeat_interleave(torch.arange(B, device=DEVICE), lens)
            masked_per_seq = torch.zeros(B, device=DEVICE)
            masked_per_seq.scatter_add_(0, seq_idx, is_masked.float())
            mask_frac = masked_per_seq / lens.float()
            qual_seqs = mask_frac < 0.20
            if qual_seqs.any():
                qual_positions = qual_seqs[seq_idx] & is_masked
                if qual_positions.any():
                    nll_val = F.cross_entropy(
                        logits[qual_positions].detach().float(),
                        input_ids[qual_positions],
                    ).item()
                    nll_ema = nll_val if nll_ema is None else (0.98 * nll_ema + 0.02 * nll_val)

        if (step + 1) % 50 == 0 or step == 0:
            pct = 100 * n_masked / max(T_total, 1)
            nll_str = f"{nll_ema:.4f}" if nll_ema is not None else "n/a"
            print(
                f"step {step + 1:5d}/{total_steps} | "
                f"loss {loss_val:.4f} | ema {loss_ema:.4f} | "
                f"mmd {mmd_val:.4f} | nll<20 {nll_str} | "
                f"toks {T_total} | masked {n_masked}/{T_total} ({pct:.0f}%) | "
                f"B {len(sample_indices)} | lr {adam.param_groups[0]['lr']:.2e}"
            )

        # Validation
        if (step + 1) % args.eval_every == 0:
            val_res = evaluate(
                model, val_sequences, val_batcher, d_model, DEVICE,
                autocast_ctx, n_batches=args.eval_batches,
            )
            print(
                f"  [VAL step {step + 1}] "
                f"ce {val_res['val_ce']:.4f} | "
                f"shuffle_ce {val_res['shuffle_ce']:.4f} | "
                f"delta {val_res['delta']:+.4f}"
            )

    # ---- Save ----
    print(f"Saving checkpoint: {args.checkpoint}")
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": TOKEN_TO_ID,
            "config": {
                "d_model": d_model,
                "n_layers": n_layers,
                "n_decoder_layers": n_decoder_layers,
                "head_dim": head_dim,
                "rope_dim": rope_dim,
                "dropout": DROPOUT,
                "vocab_size": VOCAB_SIZE,
                "total_steps": total_steps,
                "mmd_weight": mmd_weight,
            },
        },
        args.checkpoint,
    )
    print("Done.")


if __name__ == "__main__":
    main()
