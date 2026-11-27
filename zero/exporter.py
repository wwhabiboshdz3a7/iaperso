"""Exporte ton modèle fait de zéro :
  1. format Hugging Face (architecture Llama standard) → sorties/zero/hf_<taille>/
  2. GGUF (llama.cpp) puis installation dans Ollama.

    python zero/exporter.py --taille 2.4B             # export HF + GGUF + Ollama
    python zero/exporter.py --taille test --hf-seul   # export HF uniquement
"""
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ICI = Path(__file__).resolve().parent
sys.path.insert(0, str(ICI))
import config_zero as C  # noqa: E402
from poids import charger_safetensors, deplier  # noqa: E402
from tokenizer_train import SORTIE as DOSSIER_TOKENIZER  # noqa: E402

RACINE = ICI.parent
LLAMA_CPP = RACINE / "outils" / "llama.cpp"


def vers_hf(params: dict, a: C.Archi) -> dict:
    c = params["couches"]
    hf = {
        "model.embed_tokens.weight": params["embed"],
        "model.norm.weight": params["norm_finale"],
        "lm_head.weight": params["sortie"].T,
    }
    noms = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
            "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
    for i in range(a.couches):
        pre = f"model.layers.{i}."
        hf[pre + "input_layernorm.weight"] = c["norm_attn"][i]
        hf[pre + "post_attention_layernorm.weight"] = c["norm_mlp"][i]
        for k, n in noms.items():
            hf[pre + n + ".weight"] = c[k][i].T  # Hugging Face stocke [sortie, entrée]
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in hf.items()}


def config_hf(a: C.Archi) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": a.dim,
        "intermediate_size": a.ffn,
        "num_hidden_layers": a.couches,
        "num_attention_heads": a.tetes,
        "num_key_value_heads": a.tetes_kv,
        "head_dim": a.dim_tete,
        "hidden_act": "silu",
        "max_position_embeddings": a.seq,
        "rms_norm_eps": a.eps,
        "rope_theta": a.rope_theta,
        "vocab_size": a.vocab,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "bos_token_id": 0,
        "eos_token_id": 0,
        "torch_dtype": "float32",
    }


def exporter_hf(taille: str) -> Path:
    plat, meta = charger_safetensors(RACINE / "sorties" / "zero" / f"poids_{taille}.safetensors")
    a = C.Archi(**json.loads(meta["archi"]))
    sortie = RACINE / "sorties" / "zero" / f"hf_{taille}"
    sortie.mkdir(parents=True, exist_ok=True)
    save_file(vers_hf(deplier(plat), a), str(sortie / "model.safetensors"), metadata={"format": "pt"})
    (sortie / "config.json").write_text(json.dumps(config_hf(a), indent=2))
    for f in DOSSIER_TOKENIZER.glob("*.json"):
        shutil.copy(f, sortie / f.name)
    print(f"✅ Format Hugging Face → {sortie} (pas d'entraînement : {meta.get('pas')})")
    return sortie


def exporter_gguf(dossier_hf: Path, nom: str) -> Path:
    if not LLAMA_CPP.exists():
        LLAMA_CPP.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                        str(LLAMA_CPP / "requirements" / "requirements-convert_hf_to_gguf.txt")], check=True)
    sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))
    spec = importlib.util.spec_from_file_location("convert_hf_to_gguf", LLAMA_CPP / "convert_hf_to_gguf.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    # Notre tokenizer est neuf : llama.cpp ne connaît pas son empreinte. Son pré-découpage est
    # exactement celui de GPT-2, on le déclare donc comme tel.
    for obj in vars(conv).values():
        if isinstance(obj, type) and "get_vocab_base_pre" in vars(obj):
            obj.get_vocab_base_pre = lambda self, tokenizer: "gpt-2"
    gguf = dossier_hf.parent / f"{nom}-f16.gguf"
    sys.argv = ["convert_hf_to_gguf.py", str(dossier_hf), "--outfile", str(gguf), "--outtype", "f16"]
    conv.main()
    return gguf


def installer_ollama(gguf: Path, nom: str, chat: bool):
    if shutil.which("ollama") is None:
        print("Ollama absent (https://ollama.com) : le fichier GGUF est prêt, installe-le plus tard.")
        return
    if chat:  # après l'affinage conversationnel (étape suivante)
        modele = ('TEMPLATE """{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n'
                  '{{ end }}<|im_start|>assistant\n"""\nPARAMETER stop "<|im_end|>"\n')
    else:  # modèle de base : il continue le texte / le code qu'on lui donne
        modele = 'TEMPLATE "{{ .Prompt }}"\n'
    mf = gguf.parent / "Modelfile_zero"
    mf.write_text(f"FROM ./{gguf.name}\n{modele}PARAMETER stop \"{C.FIN_DOC}\"\nPARAMETER temperature 0.3\n")
    subprocess.run(["ollama", "create", nom, "-f", mf.name, "--quantize", "q4_K_M"], cwd=mf.parent, check=True)
    print(f"✅ ollama run {nom}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taille", default="2.4B", choices=list(C.TAILLES))
    ap.add_argument("--nom", default="iaperso-zero")
    ap.add_argument("--hf-seul", action="store_true")
    ap.add_argument("--chat", action="store_true", help="modèle déjà affiné en conversation")
    args = ap.parse_args()
    d = exporter_hf(args.taille)
    if not args.hf_seul:
        installer_ollama(exporter_gguf(d, args.nom), args.nom, args.chat)


if __name__ == "__main__":
    main()
