#!/usr/bin/env python3
"""
protein_mdlm_encode.py — Encode protein sequences from a FASTA file to latent
vectors using a trained MDLM encoder. Saves as a .npy file of shape (N, d_model).
"""

import argparse
import sys

import numpy as np
import torch

from protein_mdlm_core import (
    DEVICE,
    DTYPE,
    ProteinMDLM,
    encode_sequence,
)


def load_fasta(path: str) -> list[tuple[str, str]]:
    """Load FASTA, returning (header, sequence) pairs for canonical-20-AA sequences."""
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    results = []
    header = ""
    parts = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    seq = "".join(parts)
                    if seq and all(c in valid_aas for c in seq):
                        results.append((header, seq))
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line.upper())
        if parts:
            seq = "".join(parts)
            if seq and all(c in valid_aas for c in seq):
                results.append((header, seq))

    return results


def load_model(checkpoint_path: str, device: str = DEVICE) -> ProteinMDLM:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = ProteinMDLM(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_decoder_layers=cfg["n_decoder_layers"],
        head_dim=cfg["head_dim"],
        rope_dim=cfg["rope_dim"],
        dropout=0.0,
    ).to(device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def encode_fasta(
    model: ProteinMDLM,
    sequences: list[str],
    device: str = DEVICE,
    batch_max_tokens: int = 65536,
) -> np.ndarray:
    """Encode sequences through the encoder, returning (N, d_model) numpy array."""
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=DTYPE)
        if device.startswith("cuda")
        else torch.autocast(device_type="cpu", enabled=False)
    )

    encoded = [encode_sequence(s) for s in sequences]
    all_z = []
    batch_seqs = []
    batch_tokens = 0

    def flush(seqs):
        if not seqs:
            return
        lens = [len(s) for s in seqs]
        flat = [tok for s in seqs for tok in s]
        positions = [p for s in seqs for p in range(len(s))]
        cu = [0]
        r = 0
        for l in lens:
            r += l
            cu.append(r)

        input_ids = torch.tensor(flat, dtype=torch.long, device=device)
        pos = torch.tensor(positions, dtype=torch.int32, device=device)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
        max_seqlen = max(lens)

        with autocast_ctx:
            z_dec = model.encode(input_ids, cu_seqlens, max_seqlen, pos)
        all_z.append(z_dec.float().cpu())

    for seq in encoded:
        if batch_tokens + len(seq) > batch_max_tokens and batch_seqs:
            flush(batch_seqs)
            batch_seqs = []
            batch_tokens = 0
        batch_seqs.append(seq)
        batch_tokens += len(seq)
    flush(batch_seqs)

    return torch.cat(all_z, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Encode FASTA sequences to latent vectors using a trained MDLM encoder"
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    parser.add_argument("fasta", type=str, help="Input FASTA file")
    parser.add_argument("-o", "--output", type=str, default="latents.npy",
                        help="Output .npy file (default: latents.npy)")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--headers", type=str, default=None,
                        help="Save sequence headers to this text file (one per line)")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}", file=sys.stderr)
    model = load_model(args.checkpoint, device=args.device)
    print(f"Model: d={model.d_model}, {len(model.blocks)}L encoder", file=sys.stderr)

    entries = load_fasta(args.fasta)
    if not entries:
        print("ERROR: No valid sequences found.", file=sys.stderr)
        sys.exit(1)
    headers, sequences = zip(*entries)
    print(f"Loaded {len(sequences)} sequences from {args.fasta}", file=sys.stderr)

    z = encode_fasta(model, list(sequences), device=args.device)
    np.save(args.output, z)
    print(f"Saved latents {z.shape} to {args.output}", file=sys.stderr)

    if args.headers:
        with open(args.headers, "w") as f:
            for h in headers:
                f.write(h + "\n")
        print(f"Saved headers to {args.headers}", file=sys.stderr)


if __name__ == "__main__":
    main()
