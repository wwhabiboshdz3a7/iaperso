"""Lecture des fragments de tokens : tirage de fenêtres aléatoires, déterministe par pas
(reprendre au pas N redonne exactement les mêmes données). Commun au Mac et au TPU."""
from pathlib import Path

import numpy as np


class Corpus:
    def __init__(self, dossier: str, seq: int, part_validation: float = 0.005, rang: int = 0, n_machines: int = 1):
        # en multi-machines, chaque machine ne lit (et ne télécharge) qu'un fragment sur n
        fichiers = sorted(Path(dossier).glob("fragment_*.bin"))[rang::n_machines]
        if not fichiers:
            raise SystemExit(f"Aucun fragment dans {dossier}. Lance zero/prepare_corpus.py d'abord.")
        self.morceaux = [np.memmap(f, dtype=np.uint16, mode="r") for f in fichiers]
        self.tailles = np.array([len(m) for m in self.morceaux], dtype=np.int64)
        self.seq = seq
        # les derniers tokens du dernier fragment servent de validation, jamais vus en entraînement
        total = int(self.tailles.sum())
        self.n_valid = max(seq * 64, int(total * part_validation))
        self.n_valid = min(self.n_valid, int(self.tailles[-1]) - 1)
        self.debut_valid = int(self.tailles[-1]) - self.n_valid
        self.total_train = total - self.n_valid

    def _fenetre(self, pos: int) -> np.ndarray:
        """Fenêtre de seq+1 tokens à la position globale `pos` (peut chevaucher deux fragments)."""
        out = np.empty(self.seq + 1, dtype=np.int32)
        rempli = 0
        i = int(np.searchsorted(np.cumsum(self.tailles), pos, side="right"))
        local = pos - int(self.tailles[:i].sum())
        while rempli < self.seq + 1:
            m = self.morceaux[i]
            n = min(self.seq + 1 - rempli, len(m) - local)
            out[rempli:rempli + n] = m[local:local + n]
            rempli += n
            i, local = i + 1, 0
        return out

    def lot(self, pas: int, taille: int, graine: int = 0, rang: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Lot d'entraînement : (entrées, cibles) de forme [taille, seq]. `rang` = numéro de machine (multi-hôte)."""
        rng = np.random.default_rng([graine, pas, rang])  # chaque machine tire des positions différentes
        positions = rng.integers(0, self.total_train - self.seq - 1, size=taille)
        x = np.stack([self._fenetre(int(p)) for p in positions])
        return x[:, :-1], x[:, 1:]

    def lots_validation(self, taille: int, nb: int):
        dernier = self.morceaux[-1]
        pas_ = (self.n_valid - self.seq - 1) // max(1, taille * nb)
        k = 0
        for _ in range(nb):
            x = np.stack([np.asarray(dernier[self.debut_valid + (k + j) * pas_:][: self.seq + 1], dtype=np.int32)
                          for j in range(taille)])
            k += taille
            yield x[:, :-1], x[:, 1:]
