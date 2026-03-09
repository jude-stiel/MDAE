# MDAE — Masked Diffusion Autoencoder for Proteins

> **Disclaimer:** This project was built at the **YC AI/Bio Hackathon**. It is an experimental prototype — not an official product, not production-ready, and provided as-is with no warranties of any kind. Use at your own risk.

MDAE is a PyTorch implementation of a masked diffusion language model for protein sequence generation and representation learning. It combines a bidirectional transformer encoder-decoder with a cosine-scheduled masking diffusion process to learn compact latent representations of proteins and generate novel sequences via reverse diffusion.

## How It Works

1. **Training**: Protein sequences are corrupted by randomly masking amino acid positions according to a cosine schedule. An encoder compresses each sequence into a fixed-length latent vector, and a decoder reconstructs the masked positions. An MMD (Maximum Mean Discrepancy) penalty regularizes the latent space toward a Gaussian prior.
2. **Encoding**: A trained model maps any protein sequence to a dense latent vector — useful for clustering, search, or downstream prediction tasks.
3. **Generation**: Starting from noise and a fully masked sequence, the model iteratively unmasks positions through ancestral sampling (reverse diffusion) to produce novel protein sequences.

## Architecture

| Component | Details |
|---|---|
| Encoder | 6 transformer blocks (configurable) with FlashAttention + RoPE |
| Bottleneck | Average-pooled latent vector (d_model dimensions), learnable scale factors |
| Decoder | 3 transformer blocks (configurable) |
| Output head | Logits over 22-token vocab (20 amino acids + PAD + MASK) |
| Normalization | RMSNorm |
| Activation | ReLU-squared in MLP layers |
| Positional encoding | Rotary (log-spaced wavelengths, 2 to 2048) |

Default hyperparameters: `d_model=128`, `head_dim=32`, `rope_dim=24`, `dropout=1/128`.

## Requirements

- Python 3.10+
- PyTorch
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) (`flash-attn`)
- NumPy

```bash
pip install torch flash-attn numpy
```

A CUDA GPU is strongly recommended. The code automatically selects bfloat16 on Ampere+ GPUs (SM >= 80) and falls back to float16 or CPU otherwise.

## Usage

### Training

```bash
python protein_mdlm_train.py data.fasta \
  --steps 5000 \
  --seed 1337 \
  --checkpoint model.pt \
  --d-model 128 \
  --n-layers 6 \
  --n-decoder-layers 3
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--steps` | 5000 | Total training steps |
| `--d-model` | 128 | Embedding dimension |
| `--n-layers` | 6 | Encoder transformer blocks |
| `--n-decoder-layers` | 3 | Decoder transformer blocks |
| `--mmd-weight` | 1.0 | Weight on MMD latent regularization loss |
| `--val-fasta` | — | Separate validation FASTA (otherwise 5% auto-split) |
| `--checkpoint` | — | Output checkpoint path |
| `--seed` | 1337 | Random seed |

Input must be a FASTA file containing protein sequences using only the 20 canonical amino acids (`ACDEFGHIKLMNPQRSTVWY`).

Training uses a dual-optimizer setup:
- **AdamW** (lr=3e-4) for embeddings and 1D parameters
- **NorMuon** (lr=2e-2) — a custom SVD-approximation optimizer — for 2D weight matrices

Sequences are packed into token-banded batches (~65k tokens each) for efficient variable-length processing.

### Encoding

Map protein sequences to latent vectors:

```bash
python protein_mdlm_encode.py model.pt sequences.fasta \
  -o latents.npy \
  --headers headers.txt
```

Outputs a NumPy array of shape `(N, d_model)`. The optional `--headers` flag saves the corresponding FASTA headers to a text file.

### Generation

Generate novel protein sequences via reverse diffusion:

```bash
python protein_mdlm_generate.py model.pt \
  --length 100 \
  -n 10 \
  --num-steps 200 \
  --temperature 1.0 \
  -o generated.fasta
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--length` | — | Target sequence length (required unless `--fasta` is given) |
| `-n` | 10 | Number of sequences to generate |
| `--num-steps` | 200 | Reverse diffusion steps |
| `--temperature` | 1.0 | Softmax temperature |
| `--top-p` | 1.0 | Per-position nucleus filtering threshold |
| `--spatial-top-p` | — | Joint nucleus filtering across all positions |
| `--reencode` | off | Re-encode partial sequence at each step for consistency |
| `--fasta` | — | Seed FASTA to initialize from (encodes a random entry) |
| `--seed` | — | Random seed for reproducibility |

Output goes to stdout if `-o` is not specified.

## Project Structure

```
protein_mdlm_core.py      Core model architecture (transformer, RoPE, RMSNorm, etc.)
protein_mdlm_train.py      Training loop, optimizers, data loading, batching
protein_mdlm_encode.py     Inference: encode sequences to latent vectors
protein_mdlm_generate.py   Inference: generate sequences via ancestral sampling
```

## Checkpoint Format

Saved `.pt` files contain:

```python
{
    "model":  state_dict,          # Model weights
    "vocab":  {token: id, ...},    # Token-to-ID mapping
    "config": {                    # Architecture + training config
        "d_model", "n_layers", "n_decoder_layers",
        "head_dim", "rope_dim", "dropout",
        "vocab_size", "total_steps", "mmd_weight"
    }
}
```

