# IAPerso — ton IA de code perso

On ne part pas de zéro (entraîner un modèle de zéro coûte des millions) : on prend un **modèle open source déjà très bon en code** (Qwen2.5-Coder, licence Apache 2.0) et on le **spécialise** avec la méthode LoRA, sur :

1. des jeux de données de code open source (≈ 60 000 exemples question → réponse) ;
2. **tes données à toi** (`data/perso/`) : son identité, tes exemples, et **ton propre code** pour qu'il apprenne ton style.

À la fin, ton modèle tourne **en local** avec Ollama : gratuit, hors ligne, ton code ne sort pas de ta machine.

```
data/perso/*.jsonl ─┐
datasets open source ┼─► prepare_data.py ─► train_mac.py (Mac)   ─► fusion ─► export_ollama.py ─► ollama run iaperso
ton code (repo)  ────┘                      train_cloud.py (GPU)
```

## 0. Régler le projet — `config.py`

- `NOM_IA` et `PROMPT_SYSTEME` : le nom et la personnalité de ton IA.
- `MODELE_BASE` : la taille du modèle selon ta machine.

| Machine | Modèle conseillé | Commande |
|---|---|---|
| Mac M1/M2/M3/M4 8 Go | `Qwen2.5-Coder-1.5B-Instruct` | `train_mac.py` |
| Mac 16 Go | `Qwen2.5-Coder-7B-Instruct` | `train_mac.py --4bit` |
| Mac 32 Go+ | `Qwen2.5-Coder-7B` ou `14B` | `train_mac.py` (7B) / `--4bit` (14B) |
| Colab gratuit (T4 16 Go) | `Qwen2.5-Coder-7B-Instruct` | `train_cloud.py --qlora` |
| GPU A100 / H100 | `7B` ou `14B` | `train_cloud.py` |

## 1. Tes données perso

- `data/perso/identite.jsonl` : déjà rempli avec quelques exemples. Ajoute les tiens, une ligne par exemple :
  `{"question": "...", "reponse": "..."}`
  Plus tu en mets (50 à 500 c'est bien), plus le modèle te ressemble.
- Ton code :
  ```bash
  python scripts/repo_to_dataset.py ~/chemin/vers/mon-projet --sortie data/perso/mon_projet.jsonl
  ```
  Les fichiers qui contiennent peut-être un secret (`password`, `token`, `api_key`...) sont ignorés automatiquement.

## 2. Construire le dataset

```bash
python scripts/prepare_data.py                        # complet (~60k exemples)
python scripts/prepare_data.py --max-par-dataset 2000 # petit essai rapide
```

## 3a. Entraîner sur Mac

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt
python scripts/prepare_data.py --max-par-dataset 2000
python train/train_mac.py --4bit --iters 200     # essai (~15-30 min)
python train/train_mac.py --4bit                 # vrai entraînement (plusieurs heures)
python train/train_mac.py --4bit --fusionner
```
Laisse le Mac branché et empêche la mise en veille : `caffeinate -i python train/train_mac.py --4bit`.

## 3b. Entraîner sur le cloud (GPU NVIDIA)

**Google Colab** (le plus simple, gratuit avec T4) : Exécution → Modifier le type d'exécution → GPU, puis dans des cellules :
```bash
!git clone https://github.com/wwhabiboshdz3a7/iaperso && %cd iaperso
!pip install -q -r requirements-cloud.txt
!python scripts/prepare_data.py
!python train/train_cloud.py --qlora
!python train/train_cloud.py --fusionner
```
Si Colab coupe, relance `train_cloud.py --qlora` : il reprend au dernier checkpoint (à condition de sauvegarder `sorties/` sur Google Drive). Ton repo est privé ? Colab aura besoin d'un token GitHub pour le cloner, ou uploade le dossier à la main.

**Google Cloud** : une VM avec GPU L4 ou A100 (Compute Engine ou Vertex AI Workbench), mêmes commandes.

**Oracle Cloud** : attention, l'offre gratuite *Always Free* n'a **pas de GPU** (processeurs ARM seulement) : trop lent pour entraîner. Il faut une instance GPU payante (A10 / A100). En revanche une VM Always Free peut **héberger** le modèle fini (1.5B ou 7B en 4 bits) avec Ollama.

## 4. Utiliser ton IA

```bash
python scripts/export_ollama.py   # nécessite Ollama : https://ollama.com
ollama run iaperso
```
Dans VS Code : extension **Continue** → fournisseur Ollama → modèle `iaperso`.

## Et la clé API Claude ?

**Pas nécessaire.** Le pipeline marche entièrement avec des données open source et les tiennes. Et ne t'en sers pas pour générer les données d'entraînement : les conditions d'Anthropic interdisent d'utiliser les réponses de Claude pour entraîner un modèle concurrent, et tu risques la suspension du compte.

Ne colle jamais une clé API dans un fichier du repo ni dans un chat.
