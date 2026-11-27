"""Transforme TON code (un dossier ou un repo) en exemples d'entraînement,
pour que le modèle apprenne ton style, tes conventions et tes projets.

Pour chaque fichier source, on crée des exemples "continue ce code" :
le début du fichier est donné, le modèle doit écrire la suite.

Usage :
    python scripts/repo_to_dataset.py ~/mes-projets/mon-app  --sortie data/perso/mon_code.jsonl
"""
import argparse
import json
from pathlib import Path

EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".kt", ".swift",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".rb", ".sh", ".sql", ".html", ".css", ".vue", ".svelte",
}
DOSSIERS_IGNORES = {
    ".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".next",
    "target", "vendor", ".idea", ".vscode", "Pods", "DerivedData",
}
MOTS_SENSIBLES = ("api_key", "apikey", "secret", "password", "passwd", "token", "sk-ant-", "BEGIN PRIVATE KEY")


def fichiers(racine: Path):
    for f in sorted(racine.rglob("*")):
        if f.is_file() and f.suffix in EXTENSIONS and not (set(f.relative_to(racine).parts) & DOSSIERS_IGNORES):
            yield f


def decouper(lignes, taille):
    """Coupe le fichier en blocs d'environ `taille` lignes, en coupant de préférence sur une ligne vide."""
    i = 0
    while i < len(lignes):
        fin = min(i + taille, len(lignes))
        for j in range(fin, max(i + taille // 2, i + 1), -1):
            if j < len(lignes) and not lignes[j].strip():
                fin = j
                break
        yield i, fin
        i = fin


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dossier")
    p.add_argument("--sortie", default="data/perso/mon_code.jsonl")
    p.add_argument("--lignes-bloc", type=int, default=60)
    p.add_argument("--max-octets", type=int, default=200_000, help="ignore les fichiers plus gros (générés, minifiés)")
    args = p.parse_args()

    racine = Path(args.dossier).expanduser().resolve()
    sortie = Path(args.sortie)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    n_ex = n_fich = n_sensibles = 0
    with open(sortie, "w", encoding="utf-8") as out:
        for f in fichiers(racine):
            if f.stat().st_size > args.max_octets:
                continue
            try:
                texte = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(m.lower() in texte.lower() for m in MOTS_SENSIBLES):
                n_sensibles += 1  # on n'apprend jamais au modèle un fichier qui pourrait contenir un secret
                continue
            lignes = texte.splitlines()
            if len(lignes) < 10:
                continue
            n_fich += 1
            chemin = f.relative_to(racine)
            langage = f.suffix.lstrip(".")
            for debut, fin in decouper(lignes, args.lignes_bloc):
                if debut == 0:
                    continue  # il faut du contexte avant le bloc à prédire
                contexte = "\n".join(lignes[max(0, debut - 80):debut])
                suite = "\n".join(lignes[debut:fin])
                if len(suite.strip()) < 40:
                    continue
                q = (f"Fichier `{chemin}` de mon projet. Voici le code jusqu'ici :\n\n"
                     f"```{langage}\n{contexte}\n```\n\nÉcris la suite du fichier, dans mon style.")
                r = f"```{langage}\n{suite}\n```"
                out.write(json.dumps({"question": q, "reponse": r}, ensure_ascii=False) + "\n")
                n_ex += 1
    print(f"✅ {n_ex} exemples tirés de {n_fich} fichiers → {sortie}")
    if n_sensibles:
        print(f"   {n_sensibles} fichiers ignorés car ils contiennent peut-être un secret (mot de passe, clé, token)")


if __name__ == "__main__":
    main()
