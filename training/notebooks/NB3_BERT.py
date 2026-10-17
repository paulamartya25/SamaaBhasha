##############################################################
# SAMAABHASHA — NOTEBOOK 3: BERT ASR ERROR CORRECTION
# Dataset: Wikipedia Hindi — No permissions needed!
# Time: ~2 hours | GPU: T4 required
#
# HOW TO USE IN KAGGLE:
# 1. New Notebook → Enable GPU T4
# 2. Add-ons → Secrets → Add HF_TOKEN
# 3. Paste this ENTIRE code into ONE cell
# 4. Click Run ▶
##############################################################

import subprocess, sys

print("Installing packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "evaluate", "jiwer", "transformers", "datasets",
    "accelerate", "huggingface_hub"], check=True)
print("✅ Packages ready!")

# ALL IMPORTS
import torch
import random
import numpy as np
from dataclasses import dataclass
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from datasets import load_dataset, Dataset
import evaluate
from transformers import BertForMaskedLM
from transformers import BertTokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
print("✅ All imports done!")

# LOGIN
secrets = UserSecretsClient()
login(token=secrets.get_secret("HF_TOKEN"))
print("✅ HuggingFace login done!")

# GPU CHECK
assert torch.cuda.is_available(), "❌ Enable GPU: Session Options → Accelerator → T4"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# CONFIG
MODEL_NAME   = "bert-base-multilingual-cased"
HF_REPO_ID   = "paulamartya25/samaa-bert-corrector"
MAX_LENGTH   = 128
SAMPLE_SIZE  = 30000
print(f"✅ Config set! Repo: {HF_REPO_ID}")

# LOAD DATASET — Hindi Wikipedia
print("\n⏳ Loading Hindi Wikipedia dataset...")
wiki = load_dataset("wikipedia", "20220301.hi",
                     split=f"train[:{SAMPLE_SIZE}]",
                     trust_remote_code=True)
print(f"✅ Loaded {len(wiki)} articles!")

# BUILD TEXT CORPUS
print("⏳ Building text corpus from articles...")
all_texts = []
for item in wiki:
    sentences = item["text"].replace("।", ".").split(".")
    for s in sentences:
        s = s.strip()
        if 10 < len(s) < 300:
            all_texts.append(s)
    if len(all_texts) > 100000:
        break

text_dataset = Dataset.from_dict({"text": all_texts[:100000]})
split        = text_dataset.train_test_split(test_size=0.05, seed=42)
print(f"✅ Corpus ready! Train={len(split['train'])} | Test={len(split['test'])}")

# LOAD TOKENIZER & MODEL
print("\n⏳ Loading mBERT...")
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model     = BertForMaskedLM.from_pretrained(MODEL_NAME)
print(f"✅ mBERT loaded! Params={sum(p.numel() for p in model.parameters()):,}")

# TOKENIZE
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation              = True,
        max_length              = MAX_LENGTH,
        padding                 = "max_length",
        return_special_tokens_mask = True,
    )

print("⏳ Tokenizing...")
train_tok = split["train"].map(tokenize, batched=True,
                                remove_columns=["text"])
eval_tok  = split["test"].map(tokenize,  batched=True,
                               remove_columns=["text"])
print("✅ Tokenization done!")

# DATA COLLATOR — MLM masks 15% of tokens randomly
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
print("✅ MLM data collator ready!")

# TRAINING ARGS
args = TrainingArguments(
    output_dir                  = "./bert-corrector",
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 32,
    num_train_epochs            = 3,
    learning_rate               = 2e-5,
    weight_decay                = 0.01,
    warmup_ratio                = 0.1,
    fp16                        = True,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    logging_steps               = 100,
    load_best_model_at_end      = True,
    push_to_hub                 = True,
    hub_model_id                = HF_REPO_ID,
)
print("✅ Training args set!")

# CREATE TRAINER
trainer = Trainer(
    model         = model,
    args          = args,
    train_dataset = train_tok,
    eval_dataset  = eval_tok,
    data_collator = data_collator,
)
print("✅ Trainer created!")

# TRAIN!
print("\n" + "="*50)
print("🚀 TRAINING STARTED!")
print("   Time: ~2 hours")
print("   Safe to close tab — Kaggle keeps running!")
print("="*50 + "\n")
trainer.train()

# SAVE
trainer.push_to_hub()
tokenizer.push_to_hub(HF_REPO_ID)
print(f"\n✅ Model saved! → https://huggingface.co/{HF_REPO_ID}")

# RESULTS
results = trainer.evaluate()
print(f"\n{'='*50}")
print(f"FINAL EVAL LOSS: {results['eval_loss']:.4f}")
print(f"(Lower = better at predicting masked words)")
print(f"{'='*50}")
