"""Le même modèle, écrit de zéro en MLX pour le Mac (Apple Silicon).
Mêmes noms et formes de poids que modele_jax.py : les checkpoints passent de l'un à l'autre."""
import math

import mlx.core as mx
import numpy as np

from config_zero import Archi


def depuis_numpy(params: dict) -> dict:
    return {k: depuis_numpy(v) if isinstance(v, dict) else mx.array(v) for k, v in params.items()}


def rmsnorm(x, poids, eps):
    return mx.fast.rms_norm(x, poids, eps)


def rope_tables(a: Archi, seq: int):
    inv = 1.0 / (a.rope_theta ** (np.arange(0, a.dim_tete, 2, dtype=np.float32) / a.dim_tete))
    angles = np.outer(np.arange(seq, dtype=np.float32), inv)
    angles = np.concatenate([angles, angles], axis=-1)
    return mx.array(np.cos(angles)), mx.array(np.sin(angles))


def appliquer_rope(x, cos, sin):
    # x : [B, H, T, hd]
    moitie = x.shape[-1] // 2
    tourne = mx.concatenate([-x[..., moitie:], x[..., :moitie]], axis=-1)
    return (x * cos + tourne * sin).astype(x.dtype)


def couche(a: Archi, h, p, cos, sin):
    B, T, _ = h.shape
    hd = a.dim_tete
    x = rmsnorm(h, p["norm_attn"], a.eps)
    q = (x @ p["q"]).reshape(B, T, a.tetes, hd).transpose(0, 2, 1, 3)
    k = (x @ p["k"]).reshape(B, T, a.tetes_kv, hd).transpose(0, 2, 1, 3)
    v = (x @ p["v"]).reshape(B, T, a.tetes_kv, hd).transpose(0, 2, 1, 3)
    q, k = appliquer_rope(q, cos, sin), appliquer_rope(k, cos, sin)
    # noyau d'attention optimisé Metal, gère le GQA et le masque causal
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(hd), mask="causal")
    h = h + o.transpose(0, 2, 1, 3).reshape(B, T, a.tetes * hd) @ p["o"]
    x = rmsnorm(h, p["norm_mlp"], a.eps)
    g = x @ p["gate"]
    return h + (g * mx.sigmoid(g) * (x @ p["up"])) @ p["down"]  # SwiGLU


def logits(a: Archi, params: dict, tokens, dtype=mx.bfloat16, remat: bool = True):
    T = tokens.shape[1]
    cos, sin = rope_tables(a, T)
    cos, sin = cos.astype(dtype), sin.astype(dtype)
    c = params["couches"]
    h = params["embed"].astype(dtype)[tokens]
    f = mx.checkpoint(lambda h, *pc: couche(a, h, dict(zip(c.keys(), pc)), cos, sin)) if remat else \
        (lambda h, *pc: couche(a, h, dict(zip(c.keys(), pc)), cos, sin))
    for i in range(a.couches):
        h = f(h, *[c[k][i].astype(dtype) for k in c])
    h = rmsnorm(h, params["norm_finale"].astype(dtype), a.eps)
    return (h @ params["sortie"].astype(dtype)).astype(mx.float32)


def perte(a: Archi, params, x, y, dtype=mx.bfloat16):
    lg = logits(a, params, x, dtype)
    return mx.mean(mx.logsumexp(lg, axis=-1) - mx.take_along_axis(lg, y[..., None], axis=-1).squeeze(-1))
