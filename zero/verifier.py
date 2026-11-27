"""Vérifie que les trois implémentations donnent le même résultat avec les mêmes poids :
JAX (TPU) == MLX (Mac) == Llama de Hugging Face (ce qu'utilisent llama.cpp / Ollama).

    python zero/verifier.py --taille test
Les bibliothèques absentes sont simplement sautées.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import config_zero as C  # noqa: E402
from poids import charger_safetensors, deplier  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taille", default="test")
    args = ap.parse_args()
    plat, meta = charger_safetensors(ICI.parent / "sorties" / "zero" / f"poids_{args.taille}.safetensors")
    a = C.Archi(**json.loads(meta["archi"]))
    params = deplier(plat)
    tokens = np.random.default_rng(0).integers(0, a.vocab, size=(2, min(64, a.seq)))
    res = {}

    try:
        import jax
        import jax.numpy as jnp
        import modele_jax
        res["JAX"] = np.asarray(modele_jax.logits(a, jax.tree.map(jnp.asarray, params), jnp.asarray(tokens),
                                                  dtype=jnp.float32, remat=False))
    except ImportError:
        print("JAX absent, sauté")
    try:
        import mlx.core as mx
        import modele_mlx
        res["MLX"] = np.array(modele_mlx.logits(a, modele_mlx.depuis_numpy(params), mx.array(tokens),
                                                dtype=mx.float32, remat=False))
    except ImportError:
        print("MLX absent, sauté")
    try:
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        from exporter import config_hf, vers_hf
        hf = LlamaForCausalLM(LlamaConfig(**config_hf(a))).eval()
        manquants = hf.load_state_dict({k: torch.from_numpy(v) for k, v in vers_hf(params, a).items()}, strict=False)
        assert not [k for k in manquants.missing_keys if "rotary" not in k], manquants
        with torch.no_grad():
            res["HF Llama"] = hf(torch.from_numpy(tokens)).logits.float().numpy()
    except ImportError:
        print("transformers/torch absents, sauté")

    noms = list(res)
    ok = True
    for i in range(len(noms)):
        for j in range(i + 1, len(noms)):
            ecart = float(np.max(np.abs(res[noms[i]] - res[noms[j]])))
            bon = ecart < 1e-3
            ok &= bon
            print(f"{noms[i]:>8} vs {noms[j]:<8} : écart max {ecart:.2e} {'✅' if bon else '❌'}")
    sys.exit(0 if ok and len(noms) >= 2 else 1)


if __name__ == "__main__":
    main()
