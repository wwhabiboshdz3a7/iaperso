# IAPerso « de zéro » : un modèle de code de 2,4 milliards de paramètres

Tout est fait maison : le **tokenizer**, l'**architecture** (type Llama, écrite à la main en JAX et en MLX), l'**entraînement**. Aucun poids existant n'est réutilisé.

| Fichier | Rôle |
|---|---|
| `config_zero.py` | tailles de modèle (test → 2.4B), mélange du corpus, hyperparamètres |
| `tokenizer_train.py` | apprend le vocabulaire (32 768 tokens) |
| `prepare_corpus.py` | télécharge et tokenise le corpus en fragments binaires |
| `modele_jax.py` / `train_tpu.py` | modèle + entraînement TPU Google (ou GPU NVIDIA), multi-machines |
| `modele_mlx.py` / `train_mac.py` | le même modèle + entraînement sur Mac |
| `verifier.py` | prouve que JAX, MLX et Llama Hugging Face donnent le même résultat |
| `exporter.py` | export Hugging Face → GGUF → Ollama |

Les poids sont dans le même format partout : tu peux **commencer sur le Mac et continuer sur TPU** (`--depart`), ou l'inverse.

## Combien de temps ?

Un 2.4B a besoin d'environ **47 milliards de tokens** pour être bien entraîné (règle « Chinchilla » : 20 tokens par paramètre).

| Où | 150M (prototype, 3 G tokens) | 2.4B (47 G tokens) |
|---|---|---|
| Mac M5 Max 128 Go | ~1 à 3 jours | **~1 an non-stop** |
| TPU v4-32 (Google TRC, gratuit) | quelques heures | **~1 à 2 semaines** |

Ces chiffres sont des estimations : le script affiche la vraie vitesse et le temps restant dès les premiers pas.

**Conclusion : le Mac pour le prototype, les TPU Google pour le 2.4B.**

## Étape 1 — sur le Mac : prototype 150M (valide toute la chaîne)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r zero/requirements-mac.txt
hf auth login                      # compte Hugging Face gratuit
# accepte les conditions de https://huggingface.co/datasets/bigcode/starcoderdata

python zero/tokenizer_train.py --go 2               # ~1 h
python zero/prepare_corpus.py --tokens 3e9          # ~6 Go sur disque
caffeinate -i python zero/train_mac.py --taille 150M --tokens 3e9
python zero/verifier.py --taille 150M
python zero/exporter.py --taille 150M --nom iaperso-150m && ollama run iaperso-150m
```
Arrêt avec Ctrl+C quand tu veux : relancer la même commande reprend à la dernière sauvegarde.

## Étape 2 — les TPU gratuits de Google (TPU Research Cloud)

1. **Compte Google Cloud** : crée un projet et active l'API « Cloud TPU ». Les nouveaux comptes ont 300 $ de crédit, ce qui couvre largement le stockage.
2. **Candidature TRC** : https://sites.research.google/trc → décris ton projet (« modèle de code open source entraîné de zéro, code publié sur GitHub »). Google répond par email avec les TPU accordés (type, zone) pour 30 jours, souvent prolongeables. En échange, ils demandent de partager publiquement ce que tu fais (repo, article de blog...).
3. **Bucket de stockage** (même région que tes TPU) :
   ```bash
   gcloud storage buckets create gs://MON_BUCKET --location=REGION_DES_TPU
   ```
4. **Créer le TPU** (prends le type, la zone et la version indiqués dans l'email TRC) :
   ```bash
   gcloud compute tpus tpu-vm create iaperso --zone=ZONE --accelerator-type=v4-32 --version=tpu-ubuntu2204-base
   ```
5. **Installer sur toutes les machines du TPU** :
   ```bash
   gcloud compute tpus tpu-vm ssh iaperso --zone=ZONE --worker=all --command="
     git clone https://github.com/wwhabiboshdz3a7/iaperso && pip install -r iaperso/zero/requirements-tpu.txt"
   ```
6. **Préparer le corpus une fois** (sur la machine 0, rapide car dans le réseau Google), puis l'envoyer au bucket :
   ```bash
   gcloud compute tpus tpu-vm ssh iaperso --zone=ZONE --worker=0
   cd iaperso && hf auth login
   python zero/tokenizer_train.py --go 2
   python zero/prepare_corpus.py --tokens 47e9
   gcloud storage cp -r data/zero sorties/zero/tokenizer gs://MON_BUCKET/
   ```
7. **Lancer l'entraînement** sur toutes les machines. Chacune télécharge seulement sa part du corpus :
   ```bash
   gcloud compute tpus tpu-vm ssh iaperso --zone=ZONE --worker=all --command="cd iaperso &&
     gcloud storage cp -r gs://MON_BUCKET/tokenizer sorties/zero/ &&
     nohup python zero/train_tpu.py --multi-hote --taille 2.4B --tokens 47e9 \
       --gcs-donnees gs://MON_BUCKET/zero --ckpt gs://MON_BUCKET/ckpt > log.txt 2>&1 &"
   ```
   Pour suivre : `ssh --worker=0` puis `tail -f iaperso/log.txt`. Si le TPU est interrompu, relance la même commande : il reprend au dernier checkpoint du bucket.
8. **Récupérer le modèle** : `sorties/zero/poids_2.4B.safetensors` sur la machine 0 → copie-le sur ton Mac, puis `python zero/exporter.py --taille 2.4B`.

**Avant de lancer le 2.4B**, fais un essai de 10 minutes avec `--taille 150M --tokens 1e8` pour valider ton installation TPU.

## Ce qui reste à faire (étape 3)

Après le pré-entraînement, le modèle **complète du code et du texte**, mais ne sait pas encore discuter comme un assistant. La dernière étape, c'est l'**affinage conversationnel** sur des exemples question → réponse (le dossier `data/perso/` du projet sert à ça). Je l'ajouterai quand ton pré-entraînement sera lancé.
