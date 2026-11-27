"""Réglages centraux du projet IAPerso. Modifie ici, pas dans les scripts."""

# Nom et personnalité de ton IA (utilisé dans le prompt système et le Modelfile Ollama)
NOM_IA = "IAPerso"
PROMPT_SYSTEME = (
    f"Tu es {NOM_IA}, un assistant de programmation expert. "
    "Tu réponds en français sauf si on te parle dans une autre langue. "
    "Tu écris du code propre, testé et commenté quand c'est utile, "
    "tu expliques tes choix brièvement et tu signales les risques (sécurité, perfs, bugs)."
)

# Modèle de base (open source, licence Apache 2.0). Choisis selon ta machine :
#   - "Qwen/Qwen2.5-Coder-1.5B-Instruct" : test rapide, tourne sur n'importe quel Mac M1+ / GPU gratuit Colab
#   - "Qwen/Qwen2.5-Coder-7B-Instruct"   : le bon compromis (Mac 16 Go+ ou GPU 16 Go+)
#   - "Qwen/Qwen2.5-Coder-14B-Instruct"  : plus fort, Mac 32 Go+ ou GPU 24 Go+
MODELE_BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Jeux de données open source utilisés pour la spécialisation code.
# colonnes = (colonne question, colonne réponse) ; max = nb d'exemples gardés (None = tout)
DATASETS = [
    {"nom": "ise-uiuc/Magicoder-OSS-Instruct-75K", "colonnes": ("problem", "solution"), "max": 20000},
    {"nom": "ise-uiuc/Magicoder-Evol-Instruct-110K", "colonnes": ("instruction", "response"), "max": 20000},
    {"nom": "m-a-p/CodeFeedback-Filtered-Instruction", "colonnes": ("query", "answer"), "max": 20000},
]

# Tes données perso (fichiers .jsonl dans data/perso/). Elles sont répétées N fois
# pour peser plus lourd que les données génériques.
REPETITION_PERSO = 5

LONGUEUR_MAX_TOKENS = 2048
PART_VALIDATION = 0.02
GRAINE = 42
