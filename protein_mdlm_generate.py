#!/usr/bin/env python3
"""
protein_mdlm_generate.py — Generate protein sequences from a trained MDLM checkpoint.

Uses MDLM ancestral sampling (DDPM-style reverse process):
  1. Sample z ~ N(0, I)  (prior, matched by MMD training)
  2. Start with all [MASK] tokens
  3. Iteratively unmask via the reverse transition:
       For each masked position at step t -> s:
         p(stay masked) = (1 - alpha_s) / (1 - alpha_t)
         p(unmask to v)  = [(alpha_s - alpha_t) / (1 - alpha_t)] * p_theta(v | x_t)
  4. Final argmax for any remaining masks

Supports two nucleus sampling modes:
  --top-p: Standard per-position nucleus (filter vocab per position independently)
  --spatial-top-p: Joint nucleus across all (position, token) pairs — high-confidence
                   positions compete directly with uncertain ones
"""

import argparse
import random
import sys

import torch
import torch.nn.functional as F

from protein_mdlm_core import (
    DEVICE,
    DTYPE,
    MASK_ID,
    PAD_ID,
    VOCAB_SIZE,
    ProteinMDLM,
    SCHEDULE_EPS,
    alpha_cosine,
    decode_ids,
    encode_sequence,
    maybe_softcap_logits,
)


def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Parse a FASTA file into a list of (header, sequence) tuples."""
    entries: list[tuple[str, str]] = []
    header = ""
    seq_parts: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header or seq_parts:
                    entries.append((header, "".join(seq_parts)))
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line.upper())
    if header or seq_parts:
        entries.append((header, "".join(seq_parts)))
    return entries


@torch.no_grad()
def encode_seed_sequence(
    model: ProteinMDLM, seq: str, device: str = DEVICE,
) -> torch.Tensor:
    """Encode a protein sequence through the model's encoder to get a latent z.

    Returns:
        z: (1, d_model) latent vector (before scalar_dec — matches generate()'s z input).
    """
    ids = encode_sequence(seq)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    length = len(ids)
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=device)
    positions = torch.arange(length, dtype=torch.int32, device=device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=DTYPE)
        if device.startswith("cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )

    with autocast_ctx:
        z_dec = model.encode(input_ids, cu_seqlens, length, positions)
    # model.encode returns z_mean * scalar_dec; undo scalar_dec so generate()
    # can re-apply it consistently
    z = z_dec / model.scalar_dec.to(z_dec.dtype)
    return z


def load_model(checkpoint_path: str, device: str = DEVICE) -> ProteinMDLM:
    """Load a ProteinMDLM from a training checkpoint (.pt file)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = ProteinMDLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_decoder_layers=cfg["n_decoder_layers"],
        head_dim=cfg["head_dim"],
        rope_dim=cfg["rope_dim"],
        dropout=0.0,  # no dropout at inference
    ).to(device)

    # torch.compile saves keys with "_orig_mod." prefix — strip it
    state_dict = ckpt["model"]
    cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.eval()
    return model


# ============================================================
# Nucleus filtering
# ============================================================

def _nucleus_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Per-position nucleus (top-p) filtering.

    For each row, sort tokens by descending probability, keep the smallest
    set whose cumulative mass >= p, zero out the rest, renormalize.

    Args:
        probs: (*, V) probability distributions.
        p: Nucleus threshold in (0, 1].

    Returns:
        Filtered and renormalized probabilities, same shape.
    """
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    # Keep the first token that pushes cumsum past p, zero everything after
    remove = (cumsum - sorted_probs) > p
    sorted_probs[remove] = 0.0
    result = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return result / result.sum(dim=-1, keepdim=True).clamp(min=1e-8)


def _spatial_nucleus_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Joint nucleus filtering across all masked (position, token) pairs.

    Flattens the (M, V) matrix into a single distribution of M*V entries
    (total mass = M since each row sums to 1). Keeps the top entries whose
    cumulative mass reaches p * M, zeros out the rest, then reshapes back
    and renormalizes per position.

    Positions whose entire row is zeroed out will have no valid tokens to
    sample — they effectively stay masked in the reverse step.

    Args:
        probs: (M, V) predicted p(x_0) for M masked positions.
        p: Fraction of total mass to keep (e.g. 0.9).

    Returns:
        Filtered and per-position renormalized probabilities, shape (M, V).
    """
    M, V = probs.shape
    flat = probs.reshape(-1)  # (M*V,)
    threshold = p * M  # total mass is M

    sorted_probs, sorted_idx = flat.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=0)
    remove = (cumsum - sorted_probs) > threshold
    sorted_probs[remove] = 0.0

    result = torch.zeros_like(flat).scatter_(0, sorted_idx, sorted_probs)
    result = result.reshape(M, V)

    # Renormalize per position; all-zero rows stay zero (→ position stays masked)
    row_sums = result.sum(dim=-1, keepdim=True)
    return torch.where(row_sums > 1e-8, result / row_sums.clamp(min=1e-8), result)


# ============================================================
# Decoder pass helper
# ============================================================

def _decoder_pass(
    model: ProteinMDLM,
    z_dec: torch.Tensor,
    x: torch.Tensor,
    is_masked: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    positions: torch.Tensor,
    seq_idx: torch.Tensor,
    autocast_ctx,
) -> torch.Tensor:
    """Run the decoder with full reveal of unmasked positions. Returns logits."""
    z_broadcast = z_dec[seq_idx]
    emb_masked = model.mask_embed.weight[1]
    emb_revealed = model.dec_tok_emb(x)
    pos_emb = torch.where(is_masked.unsqueeze(1), emb_masked, emb_revealed)
    pos_emb = model.decoder_input_norm(pos_emb)
    dec_input = z_broadcast + pos_emb

    with autocast_ctx:
        for blk in model.decoder_blocks:
            dec_input = blk(dec_input, cu_seqlens, max_seqlen, positions)
        dec_input = model.norm_f(dec_input)
        logits = model.lm_head(dec_input)

    logits = maybe_softcap_logits(logits)
    logits[:, PAD_ID] = -1e9
    logits[:, MASK_ID] = -1e9
    return logits


# ============================================================
# Generation
# ============================================================

@torch.no_grad()
def generate(
    model: ProteinMDLM,
    length: int,
    *,
    num_sequences: int = 1,
    num_steps: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    spatial_top_p: float | None = None,
    reencode: bool = False,
    device: str = DEVICE,
    z: torch.Tensor | None = None,
) -> list[str]:
    """Generate protein sequences via MDLM ancestral sampling.

    Args:
        model: Trained ProteinMDLM (already on device, eval mode).
        length: Length of each generated sequence.
        num_sequences: How many sequences to generate in parallel.
        num_steps: Number of reverse diffusion steps.
        temperature: Softmax temperature (lower = more greedy).
        top_p: Per-position nucleus threshold (1.0 = disabled).
        spatial_top_p: Joint (position, token) nucleus threshold (None = disabled).
                       Mutually exclusive with top_p < 1.
        reencode: If True, re-encode the current partial sequence through the
                  encoder after each step to update z_dec. Step 0 uses the
                  prior sample; steps 1+ use the encoder output.
        device: Device string.
        z: Optional latent vectors (num_sequences, d_model). Sampled from
           N(0, I) if not provided.

    Returns:
        List of amino-acid strings.
    """
    d_model = model.d_model

    # ---- Latent vectors ----
    if z is None:
        z = torch.randn(num_sequences, d_model, device=device)
    z = z.to(DTYPE)
    z_dec = z * model.scalar_dec.to(z.dtype)  # (B, d_model)

    # ---- Packed varlen setup (all same length) ----
    B = num_sequences
    total_len = length * B
    x = torch.full((total_len,), MASK_ID, dtype=torch.long, device=device)
    positions = torch.arange(length, dtype=torch.int32, device=device).repeat(B)
    cu_seqlens = torch.arange(0, total_len + length, length, dtype=torch.int32, device=device)
    seq_idx = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.long), length
    )

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=DTYPE)
        if device.startswith("cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )

    use_nucleus = top_p < 1.0
    use_spatial = spatial_top_p is not None

    # ---- Reverse diffusion ----
    timesteps = torch.linspace(1.0, SCHEDULE_EPS, num_steps + 1, device=device)

    for i in range(num_steps):
        t = timesteps[i]
        s = timesteps[i + 1]
        alpha_t = alpha_cosine(t)
        alpha_s = alpha_cosine(s)

        # Re-encode partial sequence to update z_dec (skip step 0: use prior)
        if reencode and i > 0:
            with autocast_ctx:
                z_dec = model.encode(x, cu_seqlens, length, positions)

        is_masked = (x == MASK_ID)
        if not is_masked.any():
            break

        logits = _decoder_pass(
            model, z_dec, x, is_masked,
            cu_seqlens, length, positions, seq_idx, autocast_ctx,
        )

        # Model's predicted clean-token distribution
        p_x0 = F.softmax(logits / temperature, dim=-1)

        # Apply nucleus filtering
        if use_spatial:
            # Joint filtering across all masked (position, token) pairs
            masked_p = p_x0[is_masked]  # (M, V)
            masked_p = _spatial_nucleus_filter(masked_p, spatial_top_p)
            p_x0 = p_x0.clone()
            p_x0[is_masked] = masked_p
        elif use_nucleus:
            p_x0 = _nucleus_filter(p_x0, top_p)

        # Reverse transition categorical
        unmask_prob = (alpha_s - alpha_t) / (1.0 - alpha_t)
        stay_prob = (1.0 - alpha_s) / (1.0 - alpha_t)

        q_xs = p_x0 * unmask_prob  # (total_len, V)
        q_xs[:, MASK_ID] = stay_prob

        sampled = torch.multinomial(q_xs, 1).squeeze(-1)
        x = torch.where(is_masked, sampled, x)

    # ---- Final greedy decode for any remaining masks ----
    remaining = (x == MASK_ID)
    if remaining.any():
        logits = _decoder_pass(
            model, z_dec, x, remaining,
            cu_seqlens, length, positions, seq_idx, autocast_ctx,
        )
        x[remaining] = logits[remaining].argmax(dim=-1)

    # ---- Decode to strings ----
    sequences = x.view(B, length)
    return [decode_ids(seq) for seq in sequences]


def main():
    parser = argparse.ArgumentParser(
        description="Generate protein sequences from a trained MDLM checkpoint"
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("--length", type=int, default=None,
                        help="Sequence length to generate (required unless --fasta is given)")
    parser.add_argument("--num-sequences", "-n", type=int, default=10, help="Number of sequences")
    parser.add_argument("--num-steps", type=int, default=200, help="Reverse diffusion steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Per-position nucleus threshold (1.0 = disabled)")
    parser.add_argument("--spatial-top-p", type=float, default=None,
                        help="Joint (position,token) nucleus threshold (e.g. 0.9)")
    parser.add_argument("--reencode", action="store_true",
                        help="Re-encode partial sequence through encoder each step after the first")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output FASTA file (stdout if omitted)")
    parser.add_argument("--device", type=str, default=DEVICE, help="Device (cuda/cpu)")
    parser.add_argument("--fasta", type=str, default=None,
                        help="Input FASTA file; a random sequence is selected and encoded "
                             "as the latent initialization (replaces random z)")
    args = parser.parse_args()

    if args.top_p < 1.0 and args.spatial_top_p is not None:
        parser.error("--top-p and --spatial-top-p are mutually exclusive")
    if args.fasta is None and args.length is None:
        parser.error("--length is required unless --fasta is given")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if args.device.startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)

    print(f"Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    model = load_model(args.checkpoint, device=args.device)
    print(
        f"Model: d={model.d_model}, enc={len(model.blocks)}L, "
        f"dec={len(model.decoder_blocks)}L, {sum(p.numel() for p in model.parameters())/1e6:.2f}M params",
        file=sys.stderr,
    )

    # ---- Optional FASTA seed ----
    z_init = None
    if args.fasta is not None:
        entries = parse_fasta(args.fasta)
        if not entries:
            parser.error(f"No sequences found in {args.fasta}")
        header, seed_seq = random.choice(entries)
        print(f"Seed protein: {header} (length {len(seed_seq)})", file=sys.stderr)
        if args.length is None:
            args.length = len(seed_seq)
        z_seed = encode_seed_sequence(model, seed_seq, device=args.device)  # (1, d_model)
        z_init = z_seed.expand(args.num_sequences, -1)  # broadcast to batch

    sampling_desc = f"{args.num_steps} steps, temp={args.temperature}"
    if args.top_p < 1.0:
        sampling_desc += f", top-p={args.top_p}"
    if args.spatial_top_p is not None:
        sampling_desc += f", spatial-top-p={args.spatial_top_p}"
    if args.reencode:
        sampling_desc += ", reencode"
    if args.fasta is not None:
        sampling_desc += ", fasta-seeded"

    print(
        f"Generating {args.num_sequences} sequences of length {args.length} "
        f"({sampling_desc})",
        file=sys.stderr,
    )
    sequences = generate(
        model,
        args.length,
        num_sequences=args.num_sequences,
        num_steps=args.num_steps,
        temperature=args.temperature,
        top_p=args.top_p,
        spatial_top_p=args.spatial_top_p,
        reencode=args.reencode,
        device=args.device,
        z=z_init,
    )

    # ---- Output as FASTA ----
    out = open(args.output, "w") if args.output else sys.stdout
    for i, seq in enumerate(sequences):
        out.write(f">generated_{i}\n{seq}\n")
    if args.output:
        out.close()
        print(f"Wrote {len(sequences)} sequences to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
