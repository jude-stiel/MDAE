#!/usr/bin/env python3
"""
protein_mdlm_core.py — Shared model, vocabulary, schedule, and components
for the protein Masked Diffusion Language Model.

Split from protein_mdlm_ae.py so that both training and generation can
import the model without pulling in optimizer / batching code.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Performance knobs
# ============================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ============================================================
# FlashAttention
# ============================================================
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

# ============================================================
# Vocabulary
# ============================================================
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"

VOCAB_TOKENS = [PAD_TOKEN, MASK_TOKEN] + list(AMINO_ACIDS)
TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB_TOKENS)}
ID_TO_TOKEN = {i: tok for tok, i in TOKEN_TO_ID.items()}
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
MASK_ID = TOKEN_TO_ID[MASK_TOKEN]
VOCAB_SIZE = len(VOCAB_TOKENS)  # 22

# ============================================================
# Device / dtype
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
    else torch.float16
)

# ============================================================
# Model architecture defaults
# ============================================================
D_MODEL = 128
N_LAYERS = 6
N_DECODER_LAYERS = 3
HEAD_DIM = 32
ROPE_DIM = 24
DROPOUT = 1 / 128

# ============================================================
# Logit softcapping
# ============================================================
USE_LOGIT_SOFTCAP = True
SOFTCAP_CAP = 23.0
SOFTCAP_SHIFT = 5.0
SOFTCAP_SCALE = 7.5

# ============================================================
# MDLM schedule
# ============================================================
SCHEDULE_EPS = 1e-3


def alpha_cosine(t: torch.Tensor) -> torch.Tensor:
    """Cosine masking schedule: alpha(0) ~ 1 (clean), alpha(1) ~ eps (fully masked)."""
    return SCHEDULE_EPS + (1.0 - SCHEDULE_EPS) * torch.cos((math.pi / 2) * t)


# ============================================================
# Helpers
# ============================================================

def encode_sequence(seq: str) -> list:
    return [TOKEN_TO_ID[aa] for aa in seq]


def decode_ids(ids) -> str:
    return "".join(ID_TO_TOKEN[int(i)] for i in ids)


def maybe_softcap_logits(logits: torch.Tensor) -> torch.Tensor:
    if not USE_LOGIT_SOFTCAP:
        return logits
    return SOFTCAP_CAP * torch.sigmoid((logits + SOFTCAP_SHIFT) / SOFTCAP_SCALE)


# ============================================================
# Model components
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x_float / rms).to(x.dtype) * self.weight.to(x.dtype)


class CustomRoPE(nn.Module):
    """Partial RoPE with log-spaced wavelengths (2 -> 2048)."""

    def __init__(
        self,
        head_dim: int,
        rope_dim: int,
        min_wavelength: float = 2.0,
        max_wavelength: float = 2048.0,
    ):
        super().__init__()
        assert rope_dim % 2 == 0
        assert 0 <= rope_dim <= head_dim
        self.rope_dim = rope_dim
        self.n_freq = rope_dim // 2

        if self.n_freq > 0:
            wl = torch.logspace(
                math.log10(min_wavelength),
                math.log10(max_wavelength),
                steps=self.n_freq,
            ).to(torch.float32)
        else:
            wl = torch.empty((0,), dtype=torch.float32)
        self.register_buffer("wavelengths", wl, persistent=False)

    def forward(self, q, k, positions):
        if self.rope_dim == 0:
            return q, k

        pos = positions.to(torch.float32)
        angles = (2.0 * math.pi) * pos[:, None] / self.wavelengths[None, :]
        cos = torch.cos(angles).to(dtype=q.dtype)
        sin = torch.sin(angles).to(dtype=q.dtype)

        def _rotate(x: torch.Tensor) -> torch.Tensor:
            x_rope = x[..., : self.rope_dim].contiguous()
            x_pass = x[..., self.rope_dim :]
            x_rope = x_rope.view(x_rope.shape[0], x_rope.shape[1], self.n_freq, 2)
            c = cos[:, None, :]
            s = sin[:, None, :]
            x1, x2 = x_rope[..., 0], x_rope[..., 1]
            y1 = x1 * c - x2 * s
            y2 = x1 * s + x2 * c
            y = torch.stack([y1, y2], dim=-1).reshape(
                x_rope.shape[0], x_rope.shape[1], self.rope_dim
            )
            return torch.cat([y, x_pass], dim=-1)

        return _rotate(q), _rotate(k)


def qk_rms_norm(x: torch.Tensor, eps: float = 1e-6, scale: float = 1.0) -> torch.Tensor:
    x_float = x.float()
    rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return ((x_float / rms) * float(scale)).to(x.dtype)


# ============================================================
# Attention (bidirectional)
# ============================================================

class FlashBidirectionalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        dropout: float,
        rope_dim: int,
        rope_min_wl: float = 2.0,
        rope_max_wl: float = 2048.0,
        qk_norm_scale: float = 1.0,
    ):
        super().__init__()
        assert d_model % head_dim == 0
        self.d_model = d_model
        self.head_dim = head_dim
        self.n_heads = d_model // head_dim
        self.dropout = float(dropout)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        bound = math.sqrt(3.0) * (d_model**-0.5)
        with torch.no_grad():
            self.qkv.weight.uniform_(-bound, bound)
            self.out.weight.zero_()

        self.rope = CustomRoPE(head_dim, rope_dim, rope_min_wl, rope_max_wl)
        self.qk_norm_eps = 1e-6
        self.qk_norm_scale = float(qk_norm_scale)

    def forward(self, x, cu_seqlens, max_seqlen, positions):
        T = x.shape[0]
        qkv = self.qkv(x).view(T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = qk_rms_norm(q, eps=self.qk_norm_eps, scale=self.qk_norm_scale)
        k = qk_rms_norm(k, eps=self.qk_norm_eps, scale=self.qk_norm_scale)
        q, k = self.rope(q, k, positions)

        qkv_packed = torch.stack([q, k, v], dim=1).contiguous()
        if qkv_packed.dtype not in (torch.float16, torch.bfloat16):
            qkv_packed = qkv_packed.to(DTYPE)

        out = flash_attn_varlen_qkvpacked_func(
            qkv_packed,
            cu_seqlens,
            max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )
        return self.out(out.reshape(T, self.d_model))


# ============================================================
# MLP & Block
# ============================================================

class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        d_hidden = d_model * hidden_mult
        self.fc1 = nn.Linear(d_model, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_model, bias=False)
        self.dropout = float(dropout)

        bound = math.sqrt(3.0) * (0.5 * (d_model**-0.5))
        with torch.no_grad():
            self.fc1.weight.uniform_(-bound, bound)
            self.fc2.weight.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x).pow(2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc2(x)


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        head_dim: int,
        dropout: float,
        rope_dim: int,
        rope_min_wl: float,
        rope_max_wl: float,
        qk_norm_scale: float,
        mlp_mult: int,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = FlashBidirectionalSelfAttention(
            d_model, head_dim, dropout, rope_dim, rope_min_wl, rope_max_wl, qk_norm_scale
        )
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, hidden_mult=mlp_mult, dropout=dropout)
        self.dropout = float(dropout)

    def forward(self, x, cu_seqlens, max_seqlen, positions):
        x = x + F.dropout(
            self.attn(self.norm1(x), cu_seqlens, max_seqlen, positions),
            p=self.dropout,
            training=self.training,
        )
        x = x + F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=self.training)
        return x


# ============================================================
# Protein MDLM model
# ============================================================

class ProteinMDLM(nn.Module):
    """Bidirectional transformer for masked diffusion on protein sequences."""

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 128,
        n_layers: int = 6,
        n_decoder_layers: int = 3,
        head_dim: int = 32,
        rope_dim: int = 24,
        rope_min_wl: float = 2.0,
        rope_max_wl: float = 2048.0,
        dropout: float = 1 / 128,
        mlp_mult: int = 4,
        qk_norm_scale: float = 1.0,
        tok_emb_std: float = 0.005,
        zero_init_lm_head: bool = True,
    ):
        super().__init__()
        assert d_model % head_dim == 0
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=float(tok_emb_std))
        self.drop = nn.Dropout(dropout)

        # Encoder blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    d_model, head_dim, dropout, rope_dim,
                    rope_min_wl, rope_max_wl, qk_norm_scale, mlp_mult,
                )
                for _ in range(n_layers)
            ]
        )

        # Bottleneck components
        self.scalar_enc = nn.Parameter(torch.ones(1))
        self.scalar_dec = nn.Parameter(torch.ones(1))
        self.mask_embed = nn.Embedding(2, d_model)  # 0=not-masked-hidden, 1=masked
        nn.init.normal_(self.mask_embed.weight, mean=0.0, std=0.005)
        self.dec_tok_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.dec_tok_emb.weight, mean=0.0, std=float(tok_emb_std))
        self.decoder_input_norm = RMSNorm(d_model)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    d_model, head_dim, dropout, rope_dim,
                    rope_min_wl, rope_max_wl, qk_norm_scale, mlp_mult,
                )
                for _ in range(n_decoder_layers)
            ]
        )

        self.norm_f = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if zero_init_lm_head:
            with torch.no_grad():
                self.lm_head.weight.zero_()

    def encode(self, input_ids, cu_seqlens, max_seqlen, positions):
        """Run encoder only -> average pool -> return z_dec (B, d_model)."""
        B = cu_seqlens.shape[0] - 1
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        seq_idx = torch.repeat_interleave(torch.arange(B, device=input_ids.device), lens)

        x = self.tok_emb(input_ids)
        for blk in self.blocks:
            x = blk(x, cu_seqlens, max_seqlen, positions)

        z_sum = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
        z_sum.scatter_add_(0, seq_idx.unsqueeze(1).expand_as(x), x)
        z_mean = z_sum / lens.unsqueeze(1).to(x.dtype)
        return z_mean * self.scalar_dec.to(x.dtype)

    def forward(self, input_ids, cu_seqlens, max_seqlen, positions, is_masked):
        B = cu_seqlens.shape[0] - 1
        lens = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        seq_idx = torch.repeat_interleave(torch.arange(B, device=input_ids.device), lens)

        # --- Encoder ---
        x = self.tok_emb(input_ids)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, cu_seqlens, max_seqlen, positions)

        # --- Varlen average pooling ---
        z_sum = torch.zeros(B, self.d_model, device=x.device, dtype=x.dtype)
        z_sum.scatter_add_(0, seq_idx.unsqueeze(1).expand_as(x), x)
        z_mean = z_sum / lens.unsqueeze(1).to(x.dtype)

        # --- Bottleneck scalars ---
        z_mmd = z_mean * self.scalar_enc.to(x.dtype)
        z_dec = z_mean * self.scalar_dec.to(x.dtype)

        # --- Broadcast z_dec back to token positions ---
        z_dec_broadcast = z_dec[seq_idx]

        # --- Decoder per-position input ---
        T_total = input_ids.shape[0]
        p_reveal = torch.rand(B, device=input_ids.device)
        reveal_prob = p_reveal[seq_idx]
        is_revealed = (~is_masked) & (torch.rand(T_total, device=input_ids.device) < reveal_prob)

        emb_hidden = self.mask_embed.weight[0]
        emb_masked = self.mask_embed.weight[1]
        emb_revealed = self.dec_tok_emb(input_ids)

        pos_emb = torch.where(is_masked.unsqueeze(1), emb_masked, emb_hidden)
        pos_emb = torch.where(is_revealed.unsqueeze(1), emb_revealed, pos_emb)
        pos_emb = self.decoder_input_norm(pos_emb)

        dec_input = z_dec_broadcast + pos_emb

        # --- Decoder ---
        for blk in self.decoder_blocks:
            dec_input = blk(dec_input, cu_seqlens, max_seqlen, positions)
        dec_input = self.norm_f(dec_input)
        logits = self.lm_head(dec_input)

        return logits, z_mmd
