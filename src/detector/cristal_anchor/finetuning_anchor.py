# %%
import torch

print(f"CUDA disponible : {torch.cuda.is_available()}")
print(f"Nombre de GPUs : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Nom du GPU 0 : {torch.cuda.get_device_name(0)}")
    print(f"VRAM allouée : {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
# %%

from src.config import DATA_FINETUNING, DATA_DIR
import torch
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

from trl import SFTTrainer

dataset = load_dataset("json", data_files=str(DATA_FINETUNING), split="train")

# Paramètres Modèle (Exemple avec Mistral ou Llama-3)
model_name = "mistralai/Mistral-7B-v0.1"
print("step 1: bnb_config")

# 2. Quantisation (QLoRA 4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

# Chargement Modèle & Tokenizer
print("step 2: loading model")
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=compute_dtype)


print("step 4: lora preparation")
# Préparation pour LoRA
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=16,  # Rang de la matrice (plus grand = plus de capacité d'apprentissage, mais plus lourd)
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj",
                    "o_proj", "gate_proj", "up_proj",
                    "down_proj"]  # Cible les couches d'attention
)
print("step 3: loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# model.resize_token_embeddings(len(tokenizer))
# model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

print("step 5: settings training_args")

# 4. Configuration de l'entraînement (TrainingArgs)
training_args = TrainingArguments(
    output_dir=os.path.join(DATA_DIR, "checkpoints"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    lr_scheduler_type="cosine",
    learning_rate=1e-4,
    warmup_ratio=0.03,# LR standard pour QLoRA

    logging_steps=10,
    num_train_epochs=2,  # 3 époques suffisent généralement pour le style
    # Évaluer régulièrement
    eval_steps=200,
    save_steps=200,
    save_strategy='steps',
    bf16= use_bf16,
    fp16=not use_bf16,
    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    group_by_length=True,

    report_to="none",
    run_name="run_human_style_v1",
)
# %%


# 5. Initialisation du Trainer (SFT)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer
)

# %%
# Entraînement
print("step 6: run training")
trainer.train()

# Sauvegarde du modèle final
final_path = os.path.join(DATA_DIR, "final_model")
trainer.model.save_pretrained(final_path)
