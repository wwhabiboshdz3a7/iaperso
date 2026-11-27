"""Format de poids portable (safetensors) partagé entre MLX (Mac) et JAX (TPU)."""
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def aplatir(params: dict, prefixe: str = "") -> dict:
    plat = {}
    for k, v in params.items():
        nom = f"{prefixe}{k}"
        if isinstance(v, dict):
            plat.update(aplatir(v, nom + "."))
        else:
            plat[nom] = np.asarray(v, dtype=np.float32)
    return plat


def deplier(plat: dict) -> dict:
    params: dict = {}
    for nom, v in plat.items():
        d = params
        *chemin, feuille = nom.split(".")
        for c in chemin:
            d = d.setdefault(c, {})
        d[feuille] = v
    return params


def sauver_safetensors(chemin, plat: dict, archi, pas: int):
    meta = {"archi": json.dumps(archi.dict()), "pas": str(pas)}
    tmp = Path(str(chemin) + ".tmp")
    save_file(plat, str(tmp), metadata=meta)
    tmp.replace(chemin)  # écriture atomique : jamais de fichier à moitié écrit


def charger_safetensors(chemin):
    from safetensors import safe_open

    with safe_open(str(chemin), framework="numpy") as f:
        meta = f.metadata() or {}
    return load_file(str(chemin)), meta
