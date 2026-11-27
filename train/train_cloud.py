"""Entraînement LoRA / QLoRA sur GPU NVIDIA (Google Colab, Google Cloud, Oracle GPU, RunPod...).

    pip install -r requirements-cloud.txt
    python train/train_cloud.py --qlora              # GPU 16 Go (T4 Colab, A10...) avec un 7B
    python train/train_cloud.py                      # GPU 40 Go+ (A100, H100) en bf16
    python train/train_cloud.py --fusionner          # fusionne LoRA + modèle → sorties/modele_fusionne
"""
import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))
import config  # noqa: E402

ADAPTERS = RACINE / "sorties" / "adapters_cloud"
FUSIONNE = RACINE / "sorties" / "modele_fusionne"


def bf16_ok() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def fusionner():
    dtype = torch.bfloat16 if bf16_ok() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(config.MODELE_BASE, torch_dtype=dtype, device_map="auto")
    modele = PeftModel.from_pretrained(base, str(ADAPTERS)).merge_and_unload()
    modele.save_pretrained(FUSIONNE, safe_serialization=True)
    AutoTokenizer.from_pretrained(config.MODELE_BASE).save_pretrained(FUSIONNE)
    print(f"✅ Modèle fusionné dans {FUSIONNE}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qlora", action="store_true", help="charge le modèle en 4 bits (moins de VRAM)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=8, help="accumulation de gradient (batch effectif = batch*accum)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rang", type=int, default=16, help="rang LoRA")
    p.add_argument("--fusionner", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("Pas de GPU NVIDIA détecté. Sur Mac, utilise train/train_mac.py.")
    if args.fusionner:
        return fusionner()

    donnees = RACINE / "data" / "processed"
    ds = load_dataset("json", data_files={"train": str(donnees / "train.jsonl"),
                                          "validation": str(donnees / "valid.jsonl")})

    bf16 = bf16_ok()
    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
    modele = AutoModelForCausalLM.from_pretrained(
        config.MODELE_BASE,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        quantization_config=quant,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.MODELE_BASE)

    lora = LoraConfig(
        r=args.rang, lora_alpha=args.rang * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    reglages = SFTConfig(
        output_dir=str(ADAPTERS),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=config.LONGUEUR_MAX_TOKENS,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=modele,
        args=reglages,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        peft_config=lora,
    )
    # Reprend automatiquement si une session Colab a coupé (un checkpoint existe déjà)
    reprise = any(ADAPTERS.glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=reprise or None)
    trainer.save_model(str(ADAPTERS))
    print(f"\n✅ Adapters LoRA dans {ADAPTERS}. Ensuite : python train/train_cloud.py --fusionner")


if __name__ == "__main__":
    main()
