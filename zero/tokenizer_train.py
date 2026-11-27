"""Entraîne TON tokenizer (BPE niveau octet, 32 768 tokens) sur un échantillon du corpus.

    python zero/tokenizer_train.py --go 2            # ~2 Go de texte échantillonné (conseillé)
    python zero/tokenizer_train.py --fichiers mes_textes/*.txt   # sur des fichiers locaux

Sortie : sorties/zero/tokenizer/ (tokenizer.json + config pour Hugging Face / llama.cpp)
"""
import argparse
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config_zero as C  # noqa: E402

SORTIE = Path(__file__).resolve().parent.parent / "sorties" / "zero" / "tokenizer"


def flux_corpus(octets_max: int, graine: int = 0):
    """Échantillonne le mélange CORPUS en streaming (rien n'est téléchargé en entier)."""
    from datasets import load_dataset

    sources = []
    for nom, sous, champ, poids in C.CORPUS:
        kw = {"data_dir": sous} if nom.startswith("bigcode/") else {"name": sous}
        ds = load_dataset(nom, split="train", streaming=True, **kw).shuffle(seed=graine, buffer_size=10_000)
        sources.append((iter(ds), champ, poids, f"{nom}/{sous}"))
    rng = random.Random(graine)
    poids = [s[2] for s in sources]
    lus = 0
    while lus < octets_max and sources:
        i = rng.choices(range(len(sources)), weights=poids)[0]
        it, champ, _, nom = sources[i]
        try:
            texte = next(it)[champ]
        except StopIteration:
            print(f"  source épuisée : {nom}")
            sources.pop(i), poids.pop(i)
            continue
        if texte:
            lus += len(texte.encode("utf-8"))
            yield texte


def flux_fichiers(chemins):
    for c in chemins:
        yield Path(c).read_text(encoding="utf-8", errors="ignore")


def construire():
    tok = Tokenizer(models.BPE())
    # Même découpage que GPT-2 : reconnu tel quel par llama.cpp / Ollama
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def sauver(tok: Tokenizer):
    SORTIE.mkdir(parents=True, exist_ok=True)
    tok.save(str(SORTIE / "tokenizer.json"))
    (SORTIE / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": None,
        "eos_token": C.FIN_DOC,
        "pad_token": "<|pad|>",
        "unk_token": None,
        "add_bos_token": False,
        "model_max_length": 1_000_000,
    }, indent=2))
    (SORTIE / "special_tokens_map.json").write_text(json.dumps({"eos_token": C.FIN_DOC, "pad_token": "<|pad|>"}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--go", type=float, default=2.0, help="giga-octets de texte pour apprendre le vocabulaire")
    p.add_argument("--fichiers", nargs="*", help="utilise ces fichiers locaux au lieu du corpus HF")
    p.add_argument("--vocab", type=int, default=C.VOCAB)
    args = p.parse_args()

    tok = construire()
    entraineur = trainers.BpeTrainer(
        vocab_size=args.vocab,
        special_tokens=C.SPECIAUX,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    flux = flux_fichiers(args.fichiers) if args.fichiers else flux_corpus(int(args.go * 1e9))
    tok.train_from_iterator(flux, trainer=entraineur)
    for i, s in enumerate(C.SPECIAUX):
        assert tok.token_to_id(s) == i, f"le jeton spécial {s} devrait avoir l'id {i}"
    sauver(tok)

    exemple = "def bonjour(nom):\n    return f\"Salut {nom} !\"\n"
    ids = tok.encode(exemple).ids
    print(f"\n✅ Tokenizer ({tok.get_vocab_size()} tokens) → {SORTIE}")
    print(f"   exemple : {len(exemple)} caractères → {len(ids)} tokens ; relu = {tok.decode(ids) == exemple}")


if __name__ == "__main__":
    main()
