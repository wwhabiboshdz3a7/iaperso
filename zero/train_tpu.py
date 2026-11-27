"""Pré-entraînement de zéro sur TPU Google (ou GPU NVIDIA) avec JAX.

Une seule machine :
    python zero/train_tpu.py --taille 2.4B --donnees data/zero --ckpt gs://MON_BUCKET/iaperso
Pod multi-machines (ex. v4-32) : même commande sur toutes les machines, avec --multi-hote
    gcloud compute tpus tpu-vm ssh MON_TPU --worker=all --command="cd iaperso && python zero/train_tpu.py --multi-hote ..."

Le paramètres sont répartis sur toutes les puces (FSDP) : un 2.4B tient sans problème.
Reprise automatique au dernier checkpoint si on relance la même commande.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import config_zero as C  # noqa: E402
import modele_jax as M  # noqa: E402
from donnees import Corpus  # noqa: E402
from poids import aplatir, charger_safetensors, deplier, sauver_safetensors  # noqa: E402


def specs_fsdp(params, n: int):
    """Pour chaque tableau : coupe le plus grand axe divisible par n (jamais l'axe des couches)."""
    def spec(chemin, x):
        empile = chemin[0].key == "couches"
        axes = [i for i in range(x.ndim) if not (empile and i == 0) and x.shape[i] % n == 0]
        if not axes or n == 1:
            return P()
        meilleur = max(axes, key=lambda i: x.shape[i])
        return P(*[("dp" if i == meilleur else None) for i in range(x.ndim)])
    return jax.tree_util.tree_map_with_path(spec, params)


def telecharger_part(source: str, dossier: str, rang: int, n: int):
    """Copie depuis le bucket uniquement les fragments de cette machine (fragment i si i % n == rang)."""
    import subprocess
    sortie = subprocess.run(["gcloud", "storage", "ls", source.rstrip("/") + "/fragment_*.bin"],
                            check=True, capture_output=True, text=True).stdout.split()
    a_copier = [u for u in sorted(sortie)[rang::n] if not (Path(dossier) / Path(u).name).exists()]
    Path(dossier).mkdir(parents=True, exist_ok=True)
    if a_copier:
        print(f"[machine {rang}] téléchargement de {len(a_copier)} fragments...", flush=True)
        subprocess.run(["gcloud", "storage", "cp", *a_copier, dossier], check=True)


def optimiseur(pas_total: int, lr: float):
    chauffe = min(C.WARMUP, pas_total // 10)
    planning = optax.warmup_cosine_decay_schedule(0.0, lr, chauffe, max(pas_total, chauffe + 1), lr * C.LR_MIN_RATIO)
    masque_decay = lambda p: jax.tree_util.tree_map_with_path(  # noqa: E731
        lambda chemin, _: "norm" not in chemin[-1].key, p)
    return optax.chain(
        optax.clip_by_global_norm(C.CLIP),
        optax.adamw(planning, b1=0.9, b2=0.95, eps=1e-8, weight_decay=C.WEIGHT_DECAY, mask=masque_decay),
    ), planning


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taille", default="2.4B", choices=list(C.TAILLES))
    ap.add_argument("--donnees", default=str(ICI.parent / "data" / "zero"))
    ap.add_argument("--gcs-donnees", help="gs://BUCKET/dossier : chaque machine y télécharge sa part du corpus")
    ap.add_argument("--ckpt", default=str(ICI.parent / "sorties" / "zero" / "ckpt_tpu"), help="dossier local ou gs://")
    ap.add_argument("--tokens", type=float, default=50e9, help="tokens à voir au total")
    ap.add_argument("--micro-lot", type=int, default=8, help="séquences par puce et par micro-pas")
    ap.add_argument("--tokens-par-pas", type=int, default=C.TOKENS_PAR_STEP)
    ap.add_argument("--lr", type=float, default=C.LR_MAX)
    ap.add_argument("--sauver-tous", type=int, default=500)
    ap.add_argument("--depart", help="poids .safetensors pour démarrer (ex: entraînés sur le Mac)")
    ap.add_argument("--multi-hote", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="calcul en float32 (CPU / tests)")
    args = ap.parse_args()

    if args.multi_hote:
        jax.distributed.initialize()
    a = C.TAILLES[args.taille]
    n_dev, n_proc, rang = jax.device_count(), jax.process_count(), jax.process_index()
    chef = rang == 0
    mesh = Mesh(np.array(jax.devices()), ("dp",))

    lot_global = args.micro_lot * n_dev
    accum = max(1, args.tokens_par_pas // (lot_global * a.seq))
    tokens_pas = accum * lot_global * a.seq
    pas_total = math.ceil(args.tokens / tokens_pas)
    if chef:
        print(f"Modèle {args.taille} : {a.nb_parametres() / 1e9:.2f} G paramètres | {n_dev} puces, {n_proc} machine(s)")
        print(f"{tokens_pas / 1e6:.2f} M tokens/pas (micro-lot {args.micro_lot} x {n_dev} puces x accum {accum}) "
              f"→ {pas_total} pas pour {args.tokens / 1e9:.0f} G tokens")

    # --- paramètres répartis sur les puces ---
    hote = init = M.init_parametres(a) if not args.depart else deplier(charger_safetensors(args.depart)[0])
    specs = specs_fsdp(init, n_dev)
    shard = jax.tree.map(lambda s: NamedSharding(mesh, s), specs)
    params = jax.tree.map(lambda x, s: jax.make_array_from_callback(x.shape, s, lambda idx, x=x: x[idx]), hote, shard)
    del hote, init

    tx, planning = optimiseur(pas_total, args.lr)
    # l'état d'Adam (moyennes mu, nu) est réparti exactement comme les paramètres ; le reste est répliqué
    def shard_etat(chemin, x):
        cles = [c.key for c in chemin if isinstance(c, jax.tree_util.DictKey)]
        s = shard
        try:
            for c in cles:
                s = s[c]
            if isinstance(s, NamedSharding) and x.ndim:
                return s
        except (KeyError, TypeError):
            pass
        return NamedSharding(mesh, P())
    forme_opt = jax.eval_shape(tx.init, params)
    etat_opt = jax.jit(tx.init, out_shardings=jax.tree_util.tree_map_with_path(shard_etat, forme_opt))(params)
    dtype = jnp.float32 if args.fp32 else jnp.bfloat16
    shard_lot = NamedSharding(mesh, P(None, "dp", None))

    @jax.jit
    def pas_entrainement(params, etat_opt, xs, ys):
        def micro(acc, xy):
            l, g = jax.value_and_grad(lambda p: M.perte(a, p, xy[0], xy[1], dtype))(params)
            return jax.tree.map(jnp.add, acc, (l, g)), None
        zero = (jnp.zeros(()), jax.tree.map(jnp.zeros_like, params))
        (l, g), _ = jax.lax.scan(micro, zero, (xs, ys))
        l, g = l / accum, jax.tree.map(lambda t: t / accum, g)
        maj, etat_opt = tx.update(g, etat_opt, params)
        return optax.apply_updates(params, maj), etat_opt, l, optax.global_norm(g)

    @jax.jit
    def perte_eval(params, x, y):
        return M.perte(a, params, x, y, dtype)

    # --- checkpoints (reprise automatique) ---
    gestion = ocp.CheckpointManager(args.ckpt, options=ocp.CheckpointManagerOptions(max_to_keep=3))
    debut = 0
    if gestion.latest_step() is not None:
        debut = gestion.latest_step()
        abstrait = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding),
                                {"params": params, "opt": etat_opt})
        rest = gestion.restore(debut, args=ocp.args.StandardRestore(abstrait))
        params, etat_opt = rest["params"], rest["opt"]
        if chef:
            print(f"↻ reprise au pas {debut}")

    if args.gcs_donnees:
        telecharger_part(args.gcs_donnees, args.donnees, rang, n_proc)
    corpus = Corpus(args.donnees, a.seq, rang=rang, n_machines=n_proc)

    def lot(pas):
        local = lot_global // n_proc
        x, y = corpus.lot(pas, accum * local, rang=rang)
        x, y = x.reshape(accum, local, a.seq), y.reshape(accum, local, a.seq)
        forme = (accum, lot_global, a.seq)
        return (jax.make_array_from_process_local_data(shard_lot, x, forme),
                jax.make_array_from_process_local_data(shard_lot, y, forme))

    def sauver(pas):
        gestion.save(pas, args=ocp.args.StandardSave({"params": params, "opt": etat_opt}))
        gestion.wait_until_finished()
        complet = multihost_utils.process_allgather(params, tiled=True)
        if chef:
            dossier = Path(ICI.parent / "sorties" / "zero")
            dossier.mkdir(parents=True, exist_ok=True)
            sauver_safetensors(dossier / f"poids_{args.taille}.safetensors", aplatir(complet), a, pas)

    t0, tok0 = time.time(), 0
    for pas in range(debut, pas_total):
        xs, ys = lot(pas)
        params, etat_opt, l, gn = pas_entrainement(params, etat_opt, xs, ys)
        tok0 += tokens_pas
        if chef and (pas % 10 == 0 or pas == pas_total - 1):
            l = float(l)
            dt = time.time() - t0
            vitesse = tok0 / dt
            reste = (pas_total - pas - 1) * tokens_pas / vitesse / 3600
            print(f"pas {pas:>7}/{pas_total} | perte {l:.4f} | norme grad {float(gn):.2f} | "
                  f"lr {float(planning(pas)):.2e} | {vitesse / 1e3:.1f} k tok/s | reste ~{reste:.1f} h", flush=True)
            t0, tok0 = time.time(), 0
        if (pas + 1) % args.sauver_tous == 0 or pas == pas_total - 1:
            local = lot_global // n_proc
            shard_eval = NamedSharding(mesh, P("dp", None))
            pv = np.mean([float(perte_eval(
                params,
                jax.make_array_from_process_local_data(shard_eval, x, (lot_global, a.seq)),
                jax.make_array_from_process_local_data(shard_eval, y, (lot_global, a.seq))))
                for x, y in corpus.lots_validation(local, 4)])
            if chef:
                print(f"  ✓ validation : perte {pv:.4f} (perplexité {math.exp(pv):.1f}) — sauvegarde", flush=True)
            sauver(pas + 1)
    if chef:
        print("\n✅ Pré-entraînement terminé. Poids : sorties/zero/poids_%s.safetensors" % args.taille)


if __name__ == "__main__":
    main()
