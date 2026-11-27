"""Le modèle, écrit de zéro en JAX pur (TPU Google / GPU NVIDIA).

Paramètres = dict de tableaux. Les couches sont empilées ([couches, ...]) et parcourues
avec lax.scan : compilation rapide et mémoire maîtrisée (rematérialisation par couche).
Convention des poids : x @ W, donc W a la forme [entrée, sortie].
"""
import jax
import jax.numpy as jnp
import numpy as np

from config_zero import Archi


def init_parametres(a: Archi, graine: int = 0) -> dict:
    """Initialisation façon GPT-2/Llama : normale(0, 0.02), sorties résiduelles réduites par sqrt(2*couches)."""
    rng = np.random.default_rng(graine)
    L, d, hd = a.couches, a.dim, a.dim_tete
    std, std_res = 0.02, 0.02 / np.sqrt(2 * L)

    def n(forme, s=std):
        return rng.normal(0.0, s, size=forme).astype(np.float32)

    return {
        "embed": n((a.vocab, d)),
        "couches": {
            "norm_attn": np.ones((L, d), np.float32),
            "q": n((L, d, a.tetes * hd)),
            "k": n((L, d, a.tetes_kv * hd)),
            "v": n((L, d, a.tetes_kv * hd)),
            "o": n((L, a.tetes * hd, d), std_res),
            "norm_mlp": np.ones((L, d), np.float32),
            "gate": n((L, d, a.ffn)),
            "up": n((L, d, a.ffn)),
            "down": n((L, a.ffn, d), std_res),
        },
        "norm_finale": np.ones((d,), np.float32),
        "sortie": n((d, a.vocab)),
    }


def rmsnorm(x, poids, eps):
    x32 = x.astype(jnp.float32)
    x32 = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * poids).astype(x.dtype)


def rope_tables(a: Archi, seq: int):
    inv = 1.0 / (a.rope_theta ** (np.arange(0, a.dim_tete, 2, dtype=np.float32) / a.dim_tete))
    angles = np.outer(np.arange(seq, dtype=np.float32), inv)
    angles = np.concatenate([angles, angles], axis=-1)  # convention "rotate_half" de Hugging Face
    return jnp.asarray(np.cos(angles)), jnp.asarray(np.sin(angles))


def appliquer_rope(x, cos, sin):
    # x : [B, T, H, hd]
    moitie = x.shape[-1] // 2
    tourne = jnp.concatenate([-x[..., moitie:], x[..., :moitie]], axis=-1)
    return (x * cos[None, :, None, :] + tourne * sin[None, :, None, :]).astype(x.dtype)


def couche(a: Archi, h, p, cos, sin, masque):
    B, T, _ = h.shape
    hd = a.dim_tete
    x = rmsnorm(h, p["norm_attn"], a.eps)
    q = (x @ p["q"]).reshape(B, T, a.tetes, hd)
    k = (x @ p["k"]).reshape(B, T, a.tetes_kv, hd)
    v = (x @ p["v"]).reshape(B, T, a.tetes_kv, hd)
    q, k = appliquer_rope(q, cos, sin), appliquer_rope(k, cos, sin)
    rep = a.tetes // a.tetes_kv  # attention à requêtes groupées (GQA)
    k, v = jnp.repeat(k, rep, axis=2), jnp.repeat(v, rep, axis=2)
    scores = jnp.einsum("bthd,bshd->bhts", q, k).astype(jnp.float32) / np.sqrt(hd)
    scores = jnp.where(masque, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    sortie = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(B, T, a.tetes * hd)
    h = h + sortie @ p["o"]
    x = rmsnorm(h, p["norm_mlp"], a.eps)
    return h + (jax.nn.silu(x @ p["gate"]) * (x @ p["up"])) @ p["down"]


def logits(a: Archi, params: dict, tokens, dtype=jnp.bfloat16, remat: bool = True):
    """tokens [B, T] → logits [B, T, vocab] (float32)."""
    T = tokens.shape[1]
    cos, sin = rope_tables(a, T)
    masque = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None]
    p = jax.tree.map(lambda w: w.astype(dtype), params)
    h = p["embed"][tokens]

    def corps(h, pc):
        return couche(a, h, pc, cos, sin, masque), None

    if remat:
        corps = jax.checkpoint(corps)
    h, _ = jax.lax.scan(corps, h, p["couches"])
    h = rmsnorm(h, p["norm_finale"], a.eps)
    return (h @ p["sortie"]).astype(jnp.float32)


def perte(a: Archi, params, x, y, dtype=jnp.bfloat16):
    lg = logits(a, params, x, dtype)
    logp = jax.nn.log_softmax(lg, axis=-1)
    return -jnp.mean(jnp.take_along_axis(logp, y[..., None], axis=-1))
