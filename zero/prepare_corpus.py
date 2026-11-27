"""Télécharge (en streaming), tokenise et écrit le corpus de pré-entraînement
en fragments binaires de tokens (uint16), prêts à être lus très vite pendant l'entraînement.

    python zero/prepare_corpus.py --tokens 50e9                    # objectif 2.4B (~100 Go sur disque)
    python zero/prepare_corpus.py --tokens 3e9 --sortie data/zero  # prototype 150M
    python zero/prepare_corpus.py --fichiers mon_code/**/*.py      # depuis des fichiers locaux

Reprise : relancer la même commande reprend après le dernier fragment terminé.
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import config_zero as C  # noqa: E402
from tokenizer_train import SORTIE as DOSSIER_TOKENIZER, flux_corpus, flux_fichiers  # noqa: E402

TOKENS_PAR_FRAGMENT = 100_000_000  # 200 Mo par fichier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=float, required=True, help="nombre total de tokens voulus (ex: 50e9)")
    p.add_argument("--sortie", default=str(ICI.parent / "data" / "zero"))
    p.add_argument("--fichiers", nargs="*")
    p.add_argument("--graine", type=int, default=1, help="différente de celle du tokenizer")
    args = p.parse_args()

    tok = Tokenizer.from_file(str(DOSSIER_TOKENIZER / "tokenizer.json"))
    assert tok.get_vocab_size() <= 65535, "uint16 exige un vocabulaire < 65536"
    fin = tok.token_to_id(C.FIN_DOC)
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    etat_f = sortie / "etat.json"
    etat = json.loads(etat_f.read_text()) if etat_f.exists() else {"fragments": 0, "docs_lus": 0}
    deja = etat["fragments"] * TOKENS_PAR_FRAGMENT
    objectif = int(args.tokens)
    if deja >= objectif:
        print("Corpus déjà complet.")
        return

    if args.fichiers:
        chemins = sorted({f for motif in args.fichiers for f in glob.glob(motif, recursive=True)})
        random.Random(args.graine).shuffle(chemins)
        flux = flux_fichiers(chemins)
    else:
        flux = flux_corpus(octets_max=10**15, graine=args.graine)
    # reprise : saute les documents déjà traités (le flux est déterministe pour une graine donnée)
    for _ in range(etat["docs_lus"]):
        next(flux)

    tampon = np.empty(TOKENS_PAR_FRAGMENT, dtype=np.uint16)
    rempli, docs, lot = 0, etat["docs_lus"], []

    def vider_lot():
        nonlocal rempli
        for enc in tok.encode_batch(lot, add_special_tokens=False):
            ids = enc.ids + [fin]
            while ids:
                n = min(len(ids), TOKENS_PAR_FRAGMENT - rempli)
                tampon[rempli:rempli + n] = ids[:n]
                rempli += n
                ids = ids[n:]
                if rempli == TOKENS_PAR_FRAGMENT:
                    ecrire()
        lot.clear()

    def ecrire():
        nonlocal rempli
        i = etat["fragments"]
        tampon.tofile(sortie / f"fragment_{i:05d}.bin")
        etat["fragments"] = i + 1
        etat["docs_lus"] = docs  # approximation sûre : au pire on relit quelques documents
        etat_f.write_text(json.dumps(etat))
        rempli = 0
        print(f"  fragment {i} écrit ({(i + 1) * TOKENS_PAR_FRAGMENT / 1e9:.1f} G tokens)", flush=True)

    for texte in flux:
        lot.append(texte)
        docs += 1
        if len(lot) == 2000:
            vider_lot()
            if etat["fragments"] * TOKENS_PAR_FRAGMENT >= objectif:
                break
    else:
        if lot:
            vider_lot()
        if rempli:  # dernier fragment partiel (fichiers locaux)
            tampon[:rempli].tofile(sortie / f"fragment_{etat['fragments']:05d}.bin")
            etat["fragments"] += 1
            etat["docs_lus"] = docs
            etat_f.write_text(json.dumps(etat))
    print(f"\n✅ {etat['fragments']} fragments dans {sortie}")


if __name__ == "__main__":
    main()
