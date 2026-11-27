"""Construit le dataset d'entraînement au format chat :
    data/processed/train.jsonl  et  data/processed/valid.jsonl
Chaque ligne : {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}]}
Ce format est lu directement par mlx-lm (Mac) et par TRL (cloud).

Usage :
    python scripts/prepare_data.py                 # datasets open source + data/perso/
    python scripts/prepare_data.py --sans-hf       # uniquement tes données perso
    python scripts/prepare_data.py --max-par-dataset 2000   # petit test rapide
"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
import config  # noqa: E402

DOSSIER_PERSO = RACINE / "data" / "perso"
DOSSIER_SORTIE = RACINE / "data" / "processed"
CARACTERES_MAX = config.LONGUEUR_MAX_TOKENS * 3  # ~3,5 caractères/token pour du code, marge de sécurité


def exemple(question: str, reponse: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": config.PROMPT_SYSTEME},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": reponse.strip()},
        ]
    }


def valide(ex: dict) -> bool:
    msgs = ex.get("messages") or []
    if not msgs or msgs[-1].get("role") != "assistant":
        return False
    if any(not (m.get("content") or "").strip() for m in msgs):
        return False
    return sum(len(m["content"]) for m in msgs) <= CARACTERES_MAX


def charger_hf(max_override):
    from datasets import load_dataset

    for d in config.DATASETS:
        q_col, r_col = d["colonnes"]
        limite = max_override if max_override is not None else d["max"]
        print(f"→ Téléchargement {d['nom']} ...")
        try:
            ds = load_dataset(d["nom"], split="train")
        except Exception as e:  # dataset renommé/indisponible : on continue sans lui
            print(f"  ⚠ ignoré ({e.__class__.__name__}: {e})")
            continue
        if q_col not in ds.column_names or r_col not in ds.column_names:
            print(f"  ⚠ colonnes {d['colonnes']} absentes (trouvé {ds.column_names}), ignoré")
            continue
        ds = ds.shuffle(seed=config.GRAINE)
        if limite is not None:
            ds = ds.select(range(min(limite, len(ds))))
        n = 0
        for ligne in ds:
            ex = exemple(ligne[q_col] or "", ligne[r_col] or "")
            if valide(ex):
                n += 1
                yield ex
        print(f"  ✓ {n} exemples gardés")


def charger_perso():
    """Lit data/perso/*.jsonl. Formats acceptés par ligne :
       {"question": "...", "reponse": "..."}
       {"messages": [...]}   (le prompt système est ajouté s'il manque)
    """
    for fichier in sorted(DOSSIER_PERSO.glob("*.jsonl")):
        n = 0
        for i, brut in enumerate(fichier.read_text(encoding="utf-8").splitlines(), 1):
            if not brut.strip():
                continue
            try:
                obj = json.loads(brut)
            except json.JSONDecodeError as e:
                print(f"  ⚠ {fichier.name}:{i} JSON invalide ({e}), ligne ignorée")
                continue
            if "messages" in obj:
                msgs = obj["messages"]
                if msgs and msgs[0].get("role") != "system":
                    msgs = [{"role": "system", "content": config.PROMPT_SYSTEME}] + msgs
                ex = {"messages": msgs}
            else:
                ex = exemple(obj.get("question", ""), obj.get("reponse", ""))
            if valide(ex):
                n += 1
                yield ex
            else:
                print(f"  ⚠ {fichier.name}:{i} exemple vide ou trop long, ignoré")
        print(f"→ {fichier.name} : {n} exemples perso")


def cle(ex: dict) -> str:
    texte = "\x00".join(m["content"] for m in ex["messages"] if m["role"] != "system")
    return hashlib.sha1(texte.encode("utf-8")).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sans-hf", action="store_true", help="n'utilise que data/perso/")
    p.add_argument("--max-par-dataset", type=int, default=None, help="limite chaque dataset HF")
    args = p.parse_args()

    vus, generiques, perso = set(), [], []
    for ex in charger_perso():
        if cle(ex) not in vus:
            vus.add(cle(ex))
            perso.append(ex)
    if not args.sans_hf:
        for ex in charger_hf(args.max_par_dataset):
            if cle(ex) not in vus:
                vus.add(cle(ex))
                generiques.append(ex)

    if not perso and not generiques:
        sys.exit("Aucun exemple. Ajoute des fichiers dans data/perso/ ou retire --sans-hf.")

    rng = random.Random(config.GRAINE)
    # La validation est tirée avant la répétition, pour ne pas évaluer sur des exemples vus en entraînement
    tout = generiques + perso
    rng.shuffle(tout)
    n_valid = max(1, int(len(tout) * config.PART_VALIDATION))
    valid, train = tout[:n_valid], tout[n_valid:]
    ids_valid = {cle(e) for e in valid}
    train += [e for e in perso if cle(e) not in ids_valid] * (config.REPETITION_PERSO - 1)
    rng.shuffle(train)

    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)
    for nom, lignes in (("train", train), ("valid", valid)):
        with open(DOSSIER_SORTIE / f"{nom}.jsonl", "w", encoding="utf-8") as f:
            for ex in lignes:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n✅ {len(train)} exemples train, {len(valid)} valid → {DOSSIER_SORTIE}")
    print(f"   (dont {len(perso)} perso, répétés x{config.REPETITION_PERSO})")


if __name__ == "__main__":
    main()
