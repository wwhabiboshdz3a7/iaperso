"""Pré-entraînement de zéro sur Mac (Apple Silicon) avec MLX.

    python zero/train_mac.py --taille test --tokens 2e6       # vérifie que tout marche (1 min)
    python zero/train_mac.py --taille 150M --tokens 3e9       # prototype (quelques jours)
    python zero/train_mac.py --taille 2.4B --tokens 50e9      # l'objectif (très long sur un Mac seul)

Reprise automatique : relance la même commande. Pense à `caffeinate -i` pour éviter la veille.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import config_zero as C  # noqa: E402
import modele_jax  # noqa: E402  (initialisation partagée, numpy uniquement)
import modele_mlx as M  # noqa: E402
from donnees import Corpus  # noqa: E402
from poids import aplatir, charger_safetensors, deplier, sauver_safetensors  # noqa: E402

DOSSIER = ICI.parent / "sorties" / "zero"


def lr_a(pas, total, lr_max):
    warm = min(C.WARMUP, total // 10)
    if pas < warm:
        return lr_max * pas / max(1, warm)
    prog = min(1.0, (pas - warm) / max(1, total - warm))
    mini = lr_max * C.LR_MIN_RATIO
    return mini + 0.5 * (lr_max - mini) * (1 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taille", default="150M", choices=list(C.TAILLES))
    ap.add_argument("--donnees", default=str(ICI.parent / "data" / "zero"))
    ap.add_argument("--tokens", type=float, required=True)
    ap.add_argument("--micro-lot", type=int, default=8)
    ap.add_argument("--tokens-par-pas", type=int, default=C.TOKENS_PAR_STEP // 4)
    ap.add_argument("--lr", type=float, default=C.LR_MAX)
    ap.add_argument("--sauver-tous", type=int, default=200)
    ap.add_argument("--depart", help="poids .safetensors pour démarrer (ex: venant du TPU)")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    a = C.TAILLES[args.taille]
    accum = max(1, args.tokens_par_pas // (args.micro_lot * a.seq))
    tokens_pas = accum * args.micro_lot * a.seq
    total = math.ceil(args.tokens / tokens_pas)
    DOSSIER.mkdir(parents=True, exist_ok=True)
    f_poids = DOSSIER / f"poids_{args.taille}.safetensors"
    f_opt = DOSSIER / f"optim_{args.taille}.safetensors"
    print(f"Modèle {args.taille} : {a.nb_parametres() / 1e9:.2f} G paramètres | {tokens_pas / 1e3:.0f} k tokens/pas "
          f"(micro-lot {args.micro_lot} x accum {accum}) → {total} pas")

    debut = 0
    if f_poids.exists() and f_opt.exists():
        plat, meta = charger_safetensors(f_poids)
        params = M.depuis_numpy(deplier(plat))
        o, _ = charger_safetensors(f_opt)
        m = M.depuis_numpy(deplier({k[2:]: v for k, v in o.items() if k.startswith("m.")}))
        v = M.depuis_numpy(deplier({k[2:]: v for k, v in o.items() if k.startswith("v.")}))
        debut = int(meta.get("pas", 0))
        print(f"↻ reprise au pas {debut}")
    else:
        init = deplier(charger_safetensors(args.depart)[0]) if args.depart else modele_jax.init_parametres(a)
        params = M.depuis_numpy(init)
        m = tree_map(mx.zeros_like, params)
        v = tree_map(mx.zeros_like, params)

    dtype = mx.float32 if args.fp32 else mx.bfloat16
    grad_fn = mx.value_and_grad(lambda p, x, y: M.perte(a, p, x, y, dtype))
    noms_decay = {k for k, _ in tree_flatten(params) if "norm" not in k}

    @mx.compile
    def micro(params, x, y):
        return grad_fn(params, x, y)

    def maj(params, m, v, grads, lr, t):
        """AdamW (b1=0.9, b2=0.95) + écrêtage de la norme globale du gradient, comme sur TPU."""
        gn = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads)))
        echelle = mx.minimum(1.0, C.CLIP / (gn + 1e-6))
        P, G, Mo, V = (dict(tree_flatten(t_)) for t_ in (params, grads, m, v))
        for k in P:
            g = G[k] * echelle
            Mo[k] = 0.9 * Mo[k] + 0.1 * g
            V[k] = 0.95 * V[k] + 0.05 * g * g
            mh, vh = Mo[k] / (1 - 0.9 ** t), V[k] / (1 - 0.95 ** t)
            u = mh / (mx.sqrt(vh) + 1e-8)
            if k in noms_decay:
                u = u + C.WEIGHT_DECAY * P[k]
            P[k] = P[k] - lr * u
        return (tree_unflatten(list(P.items())), tree_unflatten(list(Mo.items())),
                tree_unflatten(list(V.items())), gn)

    corpus = Corpus(args.donnees, a.seq)

    def sauver(pas):
        sauver_safetensors(f_poids, aplatir(tree_map(np.array, params)), a, pas)
        opt = {f"m.{k}": np.array(x) for k, x in tree_flatten(m)} | {f"v.{k}": np.array(x) for k, x in tree_flatten(v)}
        sauver_safetensors(f_opt, opt, a, pas)

    t0, tk = time.time(), 0
    for pas in range(debut, total):
        x, y = corpus.lot(pas, accum * args.micro_lot)
        somme, l_tot = None, 0.0
        for i in range(accum):
            sl = slice(i * args.micro_lot, (i + 1) * args.micro_lot)
            l, g = micro(params, mx.array(x[sl]), mx.array(y[sl]))
            somme = g if somme is None else tree_map(mx.add, somme, g)
            l_tot = l_tot + l
            mx.eval(somme, l_tot)
        somme = tree_map(lambda t: t / accum, somme)
        params, m, v, gn = maj(params, m, v, somme, lr_a(pas, total, args.lr), pas + 1)
        mx.eval(params, m, v)
        tk += tokens_pas
        if pas % 10 == 0 or pas == total - 1:
            dt = time.time() - t0
            vit = tk / dt
            print(f"pas {pas:>7}/{total} | perte {float(l_tot) / accum:.4f} | norme grad {float(gn):.2f} | "
                  f"{vit / 1e3:.1f} k tok/s | reste ~{(total - pas - 1) * tokens_pas / vit / 3600:.1f} h", flush=True)
            t0, tk = time.time(), 0
        if (pas + 1) % args.sauver_tous == 0 or pas == total - 1:
            pv = np.mean([float(M.perte(a, params, mx.array(x), mx.array(y), dtype))
                          for x, y in corpus.lots_validation(min(8, args.micro_lot), 4)])
            print(f"  ✓ validation : perte {pv:.4f} (perplexité {math.exp(pv):.1f}) — sauvegarde", flush=True)
            sauver(pas + 1)
    print(f"\n✅ Terminé. Poids : {f_poids}")


if __name__ == "__main__":
    main()
