# =============================================================================
# SamaaBhasha — Notebook 2: Fine-tune mBART-50 on Indian Language Pairs
# =============================================================================
# Run this on Kaggle with T4 GPU
# Dataset: Samanantar (world's largest Indian language parallel corpus)
#          49.6 MILLION sentence pairs across 11 Indian languages!
# Goal:    Improve translation quality for low-resource Indian language pairs
#          (e.g. Telugu→Odia, Punjabi→Malayalam — barely anyone has done this)
#
# THIS IS YOUR KEY RESEARCH CONTRIBUTION 🔥
# =============================================================================

# ── CELL 1: Install ───────────────────────────────────────────────────────
# !pip install -q transformers datasets accelerate sacrebleu sentencepiece
# !pip install -q sacremoses huggingface_hub

# ── CELL 2: Imports ───────────────────────────────────────────────────────
import os
import torch
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── CELL 3: Language pair config ──────────────────────────────────────────
# mBART language codes for Indian languages
MBART_LANG_CODES = {
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
    "ur": "ur_PK",   # Urdu
    "en": "en_XX",   # English
}

# ⚡ Choose language pair to fine-tune
# Start with hi→en (most data) then experiment with rare pairs
SOURCE_LANG     = "hi"   # Hindi
TARGET_LANG     = "en"   # English
MAX_LENGTH      = 128
MODEL_NAME      = "facebook/mbart-large-50-many-to-many-mmt"
HF_REPO_ID      = "paulamartya25/samaa-mbart-hindi-english"

SRC_CODE = MBART_LANG_CODES[SOURCE_LANG]
TGT_CODE = MBART_LANG_CODES[TARGET_LANG]

print(f"Fine-tuning: {SRC_CODE} → {TGT_CODE}")

# ── CELL 4: Load Samanantar dataset (FREE, 49.6M pairs!) ──────────────────
# Samanantar = largest publicly available parallel corpus for Indian languages
# Available on HuggingFace datasets hub

print("Loading Samanantar dataset...")
dataset = load_dataset(
    "ai4bharat/samanantar",
    f"{SOURCE_LANG}",   # language code
    split="train",
)

# For faster Kaggle training — use a subset (full dataset = 8M+ pairs for Hindi)
# Research tip: Start with 100K, compare with 500K, compare with full
TRAIN_SIZE = 100_000
EVAL_SIZE  = 5_000

dataset = dataset.shuffle(seed=42)
train_data = dataset.select(range(TRAIN_SIZE))
eval_data  = dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

print(f"Train: {len(train_data)} pairs")
print(f"Eval:  {len(eval_data)} pairs")
print(f"Sample: {train_data[0]}")

# ── CELL 5: Load tokenizer + model ───────────────────────────────────────
print("Loading mBART-50 tokenizer & model...")
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
tokenizer.src_lang = SRC_CODE
tokenizer.tgt_lang = TGT_CODE

model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── CELL 6: Preprocessing ────────────────────────────────────────────────
def preprocess(examples):
    """Tokenize source and target sentences."""
    # Samanantar format: examples["src"], examples["tgt"]
    src_texts = examples["src"]
    tgt_texts = examples["tgt"]

    model_inputs = tokenizer(
        src_texts,
        text_target=tgt_texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=False,
    )
    return model_inputs

print("Tokenizing dataset...")
train_tokenized = train_data.map(
    preprocess,
    batched=True,
    batch_size=1000,
    remove_columns=train_data.column_names,
)
eval_tokenized = eval_data.map(
    preprocess,
    batched=True,
    batch_size=1000,
    remove_columns=eval_data.column_names,
)
print("Tokenization done!")

# ── CELL 7: BLEU metric (standard for translation — higher is better) ─────
sacrebleu = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Decode labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]

    result = sacrebleu.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )
    return {"bleu": round(result["score"], 4)}

# ── CELL 8: Data collator ────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

# ── CELL 9: Training arguments ───────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir              = "./mbart-finetuned-indic",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 4,   # effective batch = 32
    learning_rate           = 5e-5,
    num_train_epochs        = 3,
    warmup_ratio            = 0.1,
    weight_decay            = 0.01,
    fp16                    = True,
    predict_with_generate   = True,
    generation_max_length   = 128,
    evaluation_strategy     = "epoch",
    save_strategy           = "epoch",
    logging_steps           = 200,
    load_best_model_at_end  = True,
    metric_for_best_model   = "bleu",
    greater_is_better       = True,
    push_to_hub             = True,
    hub_model_id            = HF_REPO_ID,
)

# ── CELL 10: Train ───────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tokenized,
    eval_dataset    = eval_tokenized,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

print("Starting mBART fine-tuning...")
trainer.train()

# ── CELL 11: Save to HuggingFace Hub ─────────────────────────────────────
trainer.push_to_hub()
tokenizer.push_to_hub(HF_REPO_ID)
print(f"Saved: https://huggingface.co/{HF_REPO_ID}")

# ── CELL 12: Evaluate final BLEU ─────────────────────────────────────────
results = trainer.evaluate()
print(f"\n{'='*50}")
print(f"Final BLEU Score: {results['eval_bleu']:.2f}")
print(f"(mBART-50 baseline BLEU on Hindi→English: ~28-32)")
print(f"Your fine-tuned model should reach: ~35-42")
print(f"{'='*50}")

# ── CELL 13: Test translation ────────────────────────────────────────────
from transformers import pipeline as hf_pipeline

translator = hf_pipeline(
    "translation",
    model=HF_REPO_ID,
    src_lang=SRC_CODE,
    tgt_lang=TGT_CODE,
    max_length=200,
)

test_sentences = [
    "नमस्ते, आप कैसे हैं?",
    "मुझे हिंदी में कैप्शन चाहिए।",
    "यह एक बहुत अच्छा प्रोजेक्ट है।",
]

print("\nTranslation tests:")
for sent in test_sentences:
    result = translator(sent)
    print(f"  IN:  {sent}")
    print(f"  OUT: {result[0]['translation_text']}")
    print()

# ── CELL 14: Research Analysis — Compare language pairs ──────────────────
# This is the KEY research contribution:
# Test how well the fine-tuned model handles RARE pairs
# that have very little training data

RARE_PAIRS = [
    ("te_IN", "ml_IN", "Telugu→Malayalam"),
    ("pa_IN", "or_IN", "Punjabi→Odia"),
    ("bn_IN", "ta_IN", "Bengali→Tamil"),
    ("mr_IN", "kn_IN", "Marathi→Kannada"),
]

print("\nRare language pair evaluation (research contribution):")
print("These pairs have the least existing research — YOUR NOVELTY!")
for src, tgt, label in RARE_PAIRS:
    print(f"  {label}: [Test data needed — collect from Samanantar]")
