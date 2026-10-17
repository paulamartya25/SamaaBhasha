##############################################################
# SAMAABHASHA — NOTEBOOK 2: MBART TRANSLATION FINE-TUNING
# Dataset: Samanantar (Hindi-English) — 49.6M pairs!
# Time: ~4-5 hours | GPU: T4 required
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
    "evaluate", "sacrebleu", "sentencepiece", "sacremoses",
    "transformers", "datasets", "accelerate", "huggingface_hub"], check=True)
print("✅ Packages ready!")

# ALL IMPORTS
import torch
import numpy as np
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from datasets import load_dataset
import evaluate
from transformers import MBartForConditionalGeneration
from transformers import MBart50TokenizerFast
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
print("✅ All imports done!")

# LOGIN
secrets = UserSecretsClient()
login(token=secrets.get_secret("HF_TOKEN"))
print("✅ HuggingFace login done!")

# GPU CHECK
assert torch.cuda.is_available(), "❌ Enable GPU: Session Options → Accelerator → T4"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# CONFIG
SOURCE_LANG  = "hi"
TARGET_LANG  = "en"
SRC_CODE     = "hi_IN"
TGT_CODE     = "en_XX"
MODEL_NAME   = "facebook/mbart-large-50-many-to-many-mmt"
HF_REPO_ID   = "paulamartya25/samaa-mbart-hindi-english"
TRAIN_SIZE   = 50000
EVAL_SIZE    = 2000
MAX_LENGTH   = 128
print(f"✅ Config: {SRC_CODE} → {TGT_CODE} | Repo: {HF_REPO_ID}")

# LOAD DATASET — Samanantar
print(f"\n⏳ Loading Samanantar Hindi-English dataset ({TRAIN_SIZE} pairs)...")
dataset = load_dataset("ai4bharat/samanantar", SOURCE_LANG,
                        trust_remote_code=True, split="train")
dataset = dataset.shuffle(seed=42)
train_data = dataset.select(range(TRAIN_SIZE))
eval_data  = dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))
print(f"✅ Dataset loaded! Train={len(train_data)} | Eval={len(eval_data)}")
print(f"   Sample: {train_data[0]['src'][:60]} → {train_data[0]['tgt'][:60]}")

# LOAD TOKENIZER & MODEL
print("\n⏳ Loading mBART-50 model (large — takes ~5 mins)...")
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_CODE
tokenizer.tgt_lang = TGT_CODE
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
print(f"✅ Model loaded! Params={sum(p.numel() for p in model.parameters()):,}")

# PREPROCESS
def preprocess(examples):
    model_inputs = tokenizer(
        examples["src"],
        text_target = examples["tgt"],
        max_length  = MAX_LENGTH,
        truncation  = True,
        padding     = False,
    )
    return model_inputs

print("⏳ Tokenizing datasets...")
train_tok = train_data.map(preprocess, batched=True, batch_size=1000,
                            remove_columns=train_data.column_names)
eval_tok  = eval_data.map(preprocess,  batched=True, batch_size=1000,
                            remove_columns=eval_data.column_names)
print("✅ Tokenization done!")

# METRIC — BLEU
sacrebleu = evaluate.load("sacrebleu")
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]
    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": round(result["score"], 4)}

print("✅ BLEU metric loaded!")

# DATA COLLATOR
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
print("✅ Data collator ready!")

# TRAINING ARGS
args = Seq2SeqTrainingArguments(
    output_dir                  = "./mbart-hindi-english",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 4,
    num_train_epochs            = 3,
    learning_rate               = 5e-5,
    warmup_ratio                = 0.1,
    weight_decay                = 0.01,
    fp16                        = True,
    predict_with_generate       = True,
    generation_max_length       = MAX_LENGTH,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    logging_steps               = 100,
    load_best_model_at_end      = True,
    metric_for_best_model       = "bleu",
    greater_is_better           = True,
    push_to_hub                 = True,
    hub_model_id                = HF_REPO_ID,
)
print("✅ Training args set!")

# CREATE TRAINER
trainer = Seq2SeqTrainer(
    model           = model,
    args            = args,
    train_dataset   = train_tok,
    eval_dataset    = eval_tok,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)
print("✅ Trainer created!")

# TRAIN!
print("\n" + "="*50)
print("🚀 TRAINING STARTED!")
print("   Time: ~4-5 hours")
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
print(f"FINAL BLEU : {results['eval_bleu']:.2f}")
print(f"Baseline   : ~28-32 (mBART-50 original Hi→En)")
print(f"{'='*50}")
