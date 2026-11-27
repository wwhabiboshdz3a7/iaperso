"""Réglages du modèle fait DE ZÉRO (architecture type Llama : RMSNorm, RoPE, SwiGLU, GQA).

Le même format de poids est utilisé sur Mac (MLX) et sur TPU Google (JAX) :
tu peux commencer sur un support et continuer sur l'autre.
"""
from dataclasses import dataclass, asdict

VOCAB = 32768  # taille du tokenizer entraîné par tokenizer_train.py


@dataclass(frozen=True)
class Archi:
    dim: int
    couches: int
    tetes: int
    tetes_kv: int
    ffn: int
    seq: int = 2048
    vocab: int = VOCAB
    rope_theta: float = 100_000.0
    eps: float = 1e-5

    @property
    def dim_tete(self) -> int:
        return self.dim // self.tetes

    def nb_parametres(self) -> int:
        d, hd = self.dim, self.dim_tete
        attn = d * (self.tetes * hd) + 2 * d * (self.tetes_kv * hd) + (self.tetes * hd) * d
        mlp = 3 * d * self.ffn
        return self.couches * (attn + mlp + 2 * d) + 2 * self.vocab * d + d

    def dict(self):
        return asdict(self)


TAILLES = {
    # pour vérifier que tout marche (quelques secondes)
    "test": Archi(dim=64, couches=2, tetes=4, tetes_kv=2, ffn=176, seq=128),
    # prototype : valide toute la chaîne en quelques jours sur le Mac
    "150M": Archi(dim=768, couches=12, tetes=12, tetes_kv=4, ffn=2048),
    "500M": Archi(dim=1280, couches=20, tetes=10, tetes_kv=2, ffn=3456),
    "1B": Archi(dim=2048, couches=18, tetes=16, tetes_kv=4, ffn=5504),
    # l'objectif
    "2.4B": Archi(dim=2560, couches=32, tetes=20, tetes_kv=4, ffn=6912),
}

# Mélange du corpus de pré-entraînement : (dataset HF, config/sous-dossier, champ texte, poids)
# starcoderdata est "gated" : accepte les conditions sur sa page Hugging Face puis `hf auth login`.
CORPUS = [
    ("bigcode/starcoderdata", "python", "content", 0.20),
    ("bigcode/starcoderdata", "javascript", "content", 0.08),
    ("bigcode/starcoderdata", "typescript", "content", 0.06),
    ("bigcode/starcoderdata", "java", "content", 0.05),
    ("bigcode/starcoderdata", "c", "content", 0.03),
    ("bigcode/starcoderdata", "cpp", "content", 0.04),
    ("bigcode/starcoderdata", "go", "content", 0.03),
    ("bigcode/starcoderdata", "rust", "content", 0.03),
    ("bigcode/starcoderdata", "php", "content", 0.02),
    ("bigcode/starcoderdata", "shell", "content", 0.02),
    ("bigcode/starcoderdata", "sql", "content", 0.02),
    ("bigcode/starcoderdata", "html", "content", 0.02),
    ("bigcode/starcoderdata", "css", "content", 0.01),
    ("bigcode/starcoderdata", "markdown", "content", 0.04),
    ("HuggingFaceFW/fineweb-edu", "sample-100BT", "text", 0.27),  # anglais de qualité
    ("HuggingFaceFW/fineweb-2", "fra_Latn", "text", 0.08),         # français
]

# Jetons spéciaux (ids fixes 0..3)
FIN_DOC = "<|endoftext|>"
SPECIAUX = [FIN_DOC, "<|im_start|>", "<|im_end|>", "<|pad|>"]

# Hyperparamètres d'entraînement par défaut
LR_MAX = 3e-4
LR_MIN_RATIO = 0.1
WARMUP = 2000
WEIGHT_DECAY = 0.1
CLIP = 1.0
TOKENS_PAR_STEP = 1_048_576  # ~1M tokens par pas d'optimisation (512 séquences de 2048)


if __name__ == "__main__":
    for nom, a in TAILLES.items():
        n = a.nb_parametres()
        print(f"{nom:>5} : {n / 1e9:5.2f} milliards de paramètres, ~{20 * n / 1e9:.0f} G tokens conseillés (Chinchilla)")
