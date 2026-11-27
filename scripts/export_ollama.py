"""Convertit le modèle fusionné en GGUF et l'installe dans Ollama,
pour l'utiliser en local (terminal, VS Code avec Continue, Open WebUI...).

Prérequis : Ollama installé (https://ollama.com) et git.
    python scripts/export_ollama.py
    ollama run iaperso
"""
import shutil
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
import config  # noqa: E402

FUSIONNE = RACINE / "sorties" / "modele_fusionne"
LLAMA_CPP = RACINE / "outils" / "llama.cpp"
GGUF = RACINE / "sorties" / f"{config.NOM_IA.lower()}-f16.gguf"
MODELFILE = RACINE / "sorties" / "Modelfile"

TEMPLATE_CHATML = """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""


def lancer(cmd, **kw):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


def main():
    if not (FUSIONNE / "config.json").exists():
        sys.exit("Pas de modèle fusionné. Lance d'abord train_mac.py --fusionner ou train_cloud.py --fusionner.")
    if shutil.which("ollama") is None:
        sys.exit("Ollama n'est pas installé : https://ollama.com/download")

    if not LLAMA_CPP.exists():
        LLAMA_CPP.parent.mkdir(parents=True, exist_ok=True)
        lancer(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", LLAMA_CPP])
    lancer([sys.executable, "-m", "pip", "install", "-q", "-r", LLAMA_CPP / "requirements" / "requirements-convert_hf_to_gguf.txt"])
    lancer([sys.executable, LLAMA_CPP / "convert_hf_to_gguf.py", FUSIONNE, "--outfile", GGUF, "--outtype", "f16"])

    systeme = config.PROMPT_SYSTEME.replace('"', '\\"')
    MODELFILE.write_text(
        f"FROM ./{GGUF.name}\n"
        f'TEMPLATE """{TEMPLATE_CHATML}"""\n'
        f'SYSTEM "{systeme}"\n'
        'PARAMETER stop "<|im_end|>"\n'
        "PARAMETER temperature 0.2\n"
        f"PARAMETER num_ctx {max(4096, config.LONGUEUR_MAX_TOKENS)}\n",
        encoding="utf-8",
    )
    nom = config.NOM_IA.lower()
    # Ollama quantifie en 4 bits à l'import : ~4,7 Go pour un 7B au lieu de ~15 Go
    lancer(["ollama", "create", nom, "-f", MODELFILE.name, "--quantize", "q4_K_M"], cwd=MODELFILE.parent)
    print(f"\n✅ C'est prêt : ollama run {nom}")


if __name__ == "__main__":
    main()
