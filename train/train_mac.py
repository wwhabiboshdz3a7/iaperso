"""Entraînement LoRA sur Mac Apple Silicon (M1/M2/M3/M4) avec MLX.

    pip install -r requirements-mac.txt
    python train/train_mac.py                    # modèle de config.py, pleine précision
    python train/train_mac.py --4bit             # QLoRA : 7B tient dans 16 Go de RAM
    python train/train_mac.py --iters 200        # essai rapide

Ensuite :
    python train/train_mac.py --fusionner        # fusionne LoRA + modèle → sorties/modele_fusionne
"""
import argparse
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
import config  # noqa: E402

ADAPTERS = RACINE / "sorties" / "adapters_mac"
FUSIONNE = RACINE / "sorties" / "modele_fusionne"


def modele(quantifie: bool) -> str:
    if not quantifie:
        return config.MODELE_BASE
    # version 4 bits déjà convertie par la communauté MLX
    return "mlx-community/" + config.MODELE_BASE.split("/")[-1] + "-4bit"


def lancer(cmd):
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--4bit", dest="q4", action="store_true", help="QLoRA (moins de RAM)")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--couches", type=int, default=16, help="nb de couches adaptées par LoRA")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--fusionner", action="store_true", help="fusionne les adapters au lieu d'entraîner")
    args = p.parse_args()

    m = modele(args.q4)
    if args.fusionner:
        cmd = [sys.executable, "-m", "mlx_lm.fuse", "--model", m,
               "--adapter-path", str(ADAPTERS), "--save-path", str(FUSIONNE)]
        if args.q4:
            cmd.append("--dequantize")  # nécessaire pour convertir ensuite en GGUF / Ollama
        lancer(cmd)
        print(f"\n✅ Modèle fusionné dans {FUSIONNE}")
        return

    if not (RACINE / "data" / "processed" / "train.jsonl").exists():
        sys.exit("Lance d'abord : python scripts/prepare_data.py")
    ADAPTERS.mkdir(parents=True, exist_ok=True)
    lancer([
        sys.executable, "-m", "mlx_lm.lora",
        "--model", m,
        "--train",
        "--data", str(RACINE / "data" / "processed"),
        "--fine-tune-type", "lora",
        "--num-layers", str(args.couches),
        "--batch-size", str(args.batch),
        "--iters", str(args.iters),
        "--learning-rate", str(args.lr),
        "--max-seq-length", str(config.LONGUEUR_MAX_TOKENS),
        "--mask-prompt",          # n'apprend que les réponses, pas les questions
        "--grad-checkpoint",      # économise la RAM
        "--steps-per-eval", "200",
        "--save-every", "200",
        "--adapter-path", str(ADAPTERS),
    ])
    print(f"\n✅ Adapters LoRA dans {ADAPTERS}")
    print(f"Test : python -m mlx_lm.generate --model {m} --adapter-path {ADAPTERS} --prompt 'Qui es-tu ?'")


if __name__ == "__main__":
    main()
