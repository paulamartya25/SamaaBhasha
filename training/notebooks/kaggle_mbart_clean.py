# ============================================================
# CELL 1 — Install Dependencies
# Run this FIRST, then restart kernel before running Cell 3+
# ============================================================
!pip install -q transformers datasets accelerate evaluate sacrebleu sentencepiece sacremoses


# ============================================================
# CELL 2 — After install is done
# Kernel → Restart → Then run from Cell 3 onwards
# ============================================================
print("✅ Install done! Restart kernel then run from Cell 3")


# ============================================================
# CELL 3 — HuggingFace Login
# Add HF_TOKEN in Kaggle: Add-ons → Secrets
# ============================================================
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

secrets  = UserSecretsClient()
HF_TOKEN = secrets.get_secret("HF_TOKEN")
login(token=HF_TOKEN, add_to_git_credential=True)
print("✅ Logged in to HuggingFace!")


# ============================================================
# CELL 4 — Check GPU
# ============================================================
import torch
print(f"GPU available : {torch.cuda.is_available()}")
print(f"GPU name      : {torch.cuda.get_device_name(0)}")
print(f"GPU memory    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
assert torch.cuda.is_available(), "❌ GPU not enabled! Go to Session Options → Accelerator → T4 GPU"


# ============================================================
# CELL 5 — Imports
# ============================================================
import numpy as np
from datasets import load_dataset
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate
print("✅ All imports done!")


# ============================================================
# CELL 6 — Language Pair Config
# Change SOURCE_LANG and TARGET_LANG to train different pairs
# Run multiple times with different pairs to cover all Indian langs!
# ============================================================
# mBART-50 language codes
MBART_CODES = {
    "hi": "hi_IN",   # Hindi
    "te": "te_IN",   # Telugu
    "ta": "ta_IN",   # Tamil
    "bn": "bn_IN",   # Bengali
    "mr": "mr_IN",   # Marathi
    "gu": "gu_IN",   # Gujarati
    "kn": "kn_IN",   # Kannada
    "ml": "ml_IN",   # Malayalam
    "or": "or_IN",   # Odia
    "pa": "pa_IN",   # Punjabi
    "en": "en_XX",   # English
    "fr": "fr_XX",   # French
    "de": "de_DE",   # German
}

# ✏️ CHANGE THESE to train different language pairs
SOURCE_LANG  = "hi"       # Source: Hindi
TARGET_LANG  = "en"       # Target: English
TRAIN_SIZE   = 100000     # Use 100K pairs (Samanantar has millions!)
EVAL_SIZE    = 5000
MAX_LENGTH   = 128
MODEL_NAME   = "facebook/mbart-large-50-many-to-many-mmt"
HF_REPO_ID   = f"paulamartya25/samaa-mbart-{SOURCE_LANG}-{TARGET_LANG}"

SRC_CODE = MBART_CODES[SOURCE_LANG]
TGT_CODE = MBART_CODES[TARGET_LANG]

print(f"✅ Config ready!")
print(f"   Language pair : {SRC_CODE} → {TGT_CODE}")
print(f"   Training size : {TRAIN_SIZE:,} pairs")
print(f"   Output repo   : {HF_REPO_ID}")


# ============================================================
# CELL 7 — Load Samanantar Dataset
# World's largest Indian language parallel corpus
# 49.6 MILLION sentence pairs!
# ============================================================
print(f"Loading Samanantar {SOURCE_LANG}↔en dataset...")
print("(This may take 5-10 minutes for large datasets)")

full_dataset = load_dataset(
    "ai4bharat/samanantar",
    SOURCE_LANG,
    trust_remote_code=True,
)

dataset = full_dataset["train"].shuffle(seed=42)
train_data = dataset.select(range(TRAIN_SIZE))
eval_data  = dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

print(f"✅ Dataset loaded!")
print(f"   Total available : {len(dataset):,} pairs")
print(f"   Using train     : {len(train_data):,} pairs")
print(f"   Using eval      : {len(eval_data):,} pairs")
print(f"\nSample pair:")
print(f"   Source ({SOURCE_LANG}): {train_data[0]['src']}")
print(f"   Target (en): {train_data[0]['tgt']}")


# ============================================================
# CELL 8 — Load Tokenizer and Model
# ============================================================
print("Loading mBART-50 tokenizer and model (this takes ~5 mins)...")

tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_CODE
tokenizer.tgt_lang = TGT_CODE

model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

print(f"✅ mBART-50 loaded!")
print(f"   Parameters : {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# CELL 9 — Preprocessing Function
# ============================================================
def preprocess(examples):
    """Tokenize source (Indian lang) and target (English) texts."""
    model_inputs = tokenizer(
        examples["src"],
        text_target = examples["tgt"],
        max_length  = MAX_LENGTH,
        truncation  = True,
        padding     = False,
    )
    return model_inputs

print("Tokenizing train dataset (takes ~10 mins)...")
train_tok = train_data.map(
    preprocess,
    batched      = True,
    batch_size   = 1000,
    remove_columns = train_data.column_names,
)
print("Tokenizing eval dataset...")
eval_tok = eval_data.map(
    preprocess,
    batched      = True,
    batch_size   = 1000,
    remove_columns = eval_data.column_names,
)
print("✅ Tokenization done!")


# ============================================================
# CELL 10 — BLEU Metric Setup
# BLEU = standard metric for translation (higher = better)
# ============================================================
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

    result = sacrebleu.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
    return {"bleu": round(result["score"], 4)}

print("✅ BLEU metric loaded!")


# ============================================================
# CELL 11 — Data Collator
# ============================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model              = model,
    label_pad_token_id = -100,
    pad_to_multiple_of = 8,
)
print("✅ Data collator ready!")


# ============================================================
# CELL 12 — Training Arguments
# ============================================================
training_args = Seq2SeqTrainingArguments(
    output_dir                  = f"./mbart-{SOURCE_LANG}-{TARGET_LANG}",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 4,    # Effective batch = 32
    num_train_epochs            = 3,
    learning_rate               = 5e-5,
    warmup_ratio                = 0.1,
    weight_decay                = 0.01,
    fp16                        = True,
    predict_with_generate       = True,
    generation_max_length       = MAX_LENGTH,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    logging_steps               = 200,
    load_best_model_at_end      = True,
    metric_for_best_model       = "bleu",
    greater_is_better           = True,
    push_to_hub                 = True,
    hub_model_id                = HF_REPO_ID,
)
print("✅ Training arguments set!")


# ============================================================
# CELL 13 — Create Trainer
# ============================================================
trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tok,
    eval_dataset    = eval_tok,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)
print("✅ Trainer created!")


# ============================================================
# CELL 14 — START TRAINING 🚀
# Takes ~5-6 hours on Kaggle T4 GPU
# ============================================================
print(f"🚀 Starting mBART training: {SRC_CODE} → {TGT_CODE}")
print("   Estimated time: 5-6 hours")
print("   You can close the tab — Kaggle runs in background!")
trainer.train()
print("✅ Training complete!")


# ============================================================
# CELL 15 — Save to HuggingFace Hub
# ============================================================
print("Saving model to HuggingFace Hub...")
trainer.push_to_hub()
tokenizer.push_to_hub(HF_REPO_ID)
print(f"✅ Saved! View at: https://huggingface.co/{HF_REPO_ID}")


# ============================================================
# CELL 16 — Final BLEU Score (Research Paper Table)
# ============================================================
results = trainer.evaluate()

print(f"\n{'='*55}")
print(f"  FINAL RESULTS — mBART {SRC_CODE} → {TGT_CODE}")
print(f"{'='*55}")
print(f"  BLEU Score (fine-tuned) : {results['eval_bleu']:.2f}")
print(f"  mBART-50 baseline BLEU  : ~28-32 (Hindi→English)")
print(f"{'='*55}")
print("  → This BLEU score goes in your research paper!")


# ============================================================
# CELL 17 — Test Translation
# ============================================================
from transformers import pipeline as hf_pipeline

print("\nLoading your fine-tuned model for testing...")
translator = hf_pipeline(
    "translation",
    model    = HF_REPO_ID,
    src_lang = SRC_CODE,
    tgt_lang = TGT_CODE,
    device   = 0,   # GPU
)

test_sentences = [
    "नमस्ते, आप कैसे हैं?",
    "मुझे हिंदी में कैप्शन चाहिए।",
    "यह एक बहुत अच्छा प्रोजेक्ट है।",
    "क्या आप मेरी बात समझ रहे हैं?",
]

print("\n🌐 Translation Tests:")
print("=" * 60)
for sent in test_sentences:
    result = translator(sent, max_length=200)
    print(f"  IN  ({SOURCE_LANG}): {sent}")
    print(f"  OUT ({TARGET_LANG}): {result[0]['translation_text']}")
    print()
