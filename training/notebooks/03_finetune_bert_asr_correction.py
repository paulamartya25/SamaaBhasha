# =============================================================================
# SamaaBhasha — Notebook 3: Fine-tune mBERT for ASR Error Correction
# =============================================================================
# Run this on Kaggle with T4 GPU
# Goal: Use BERT to fix errors made by Whisper (ASR post-processing)
#
# HOW IT WORKS:
#   Whisper output (noisy): "the whether is very god today in deli"
#   BERT corrected (clean): "the weather is very good today in Delhi"
#
# WHY THIS IS NOVEL:
#   - Specifically for MULTILINGUAL Indian + World language ASR errors
#   - No existing benchmark for this task
#   - This becomes your PRIMARY research paper contribution
# =============================================================================

# ── CELL 1: Install ───────────────────────────────────────────────────────
# !pip install -q transformers datasets accelerate evaluate
# !pip install -q jiwer torch

# ── CELL 2: Imports ───────────────────────────────────────────────────────
import os
import re
import json
import torch
import random
import numpy as np
from datasets import Dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import evaluate

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── CELL 3: Config ───────────────────────────────────────────────────────
MODEL_NAME  = "bert-base-multilingual-cased"  # mBERT — 104 languages
HF_REPO_ID  = "paulamartya25/samaa-bert-asr-correction"
MAX_LENGTH  = 128

# ── CELL 4: Synthetic ASR Error Generation ────────────────────────────────
# Since no paired (noisy-ASR, clean-text) dataset exists for Indian languages,
# we CREATE one synthetically — this itself is a research contribution!

# Common Whisper errors on Indian speech
WHISPER_ERROR_PATTERNS = {
    # Phonetic confusions common in Indian-accented English
    "weather":  ["whether", "wether"],
    "Delhi":    ["deli", "delhi", "delli"],
    "Mumbai":   ["mumbay", "mombai"],
    "temple":   ["templ", "tempel"],
    "village":  ["vilage", "villge"],
    "language": ["languege", "languge", "langage"],
    "Indian":   ["indyan", "indien"],

    # Whisper hallucinations
    " um ":     [" ", " uh ", " "],
    " uh ":     [" ", " um ", " "],
}

# Word substitution errors (simulates ASR acoustic confusion)
ACOUSTIC_CONFUSIONS = [
    ("their", "there"), ("to", "too"), ("its", "it's"),
    ("then", "than"), ("affect", "effect"), ("accept", "except"),
    ("know", "no"), ("hear", "here"), ("write", "right"),
]

def inject_asr_errors(text: str, error_rate: float = 0.15) -> str:
    """
    Synthetically inject ASR-like errors into clean text.
    Simulates what Whisper produces on Indian-accented speech.
    """
    words = text.split()
    noisy_words = []

    for word in words:
        # Random deletion (ASR sometimes drops words)
        if random.random() < error_rate * 0.3:
            continue

        # Random duplication (repetition errors)
        if random.random() < error_rate * 0.1:
            noisy_words.extend([word, word])
            continue

        # Character-level noise (simulates acoustic confusion)
        if random.random() < error_rate * 0.5:
            chars = list(word)
            if len(chars) > 3:
                # Swap adjacent characters
                i = random.randint(0, len(chars) - 2)
                chars[i], chars[i+1] = chars[i+1], chars[i]
                word = "".join(chars)

        noisy_words.append(word)

    return " ".join(noisy_words)

# ── CELL 5: Load clean text data (IndicCorp — Free) ──────────────────────
from datasets import load_dataset

print("Loading IndicCorp text data for creating training pairs...")

# IndicCorp: large-scale Indian language text corpus
# We use the text to create (noisy, clean) training pairs
indic_data = load_dataset(
    "ai4bharat/IndicNLP-Corpus",
    "hi",        # Hindi
    split="train[:50000]",  # 50K samples
    trust_remote_code=True,
)

print(f"Loaded {len(indic_data)} text samples")

# ── CELL 6: Create noisy-clean dataset ───────────────────────────────────
def create_correction_dataset(examples):
    clean_texts = []
    noisy_texts = []

    for text in examples["text"]:
        if not text or len(text.split()) < 5:
            continue
        clean  = text.strip()
        noisy  = inject_asr_errors(clean, error_rate=0.15)
        clean_texts.append(clean)
        noisy_texts.append(noisy)

    return {"noisy": noisy_texts, "clean": clean_texts}

print("Creating (noisy, clean) training pairs...")
correction_dataset = indic_data.map(
    create_correction_dataset,
    batched=True,
    remove_columns=indic_data.column_names,
    batch_size=1000,
)

print(f"Dataset size: {len(correction_dataset)}")
print(f"\nSample pair:")
print(f"  Noisy: {correction_dataset[0]['noisy']}")
print(f"  Clean: {correction_dataset[0]['clean']}")

# ── CELL 7: Fine-tune mBERT with MLM on clean text ───────────────────────
# Strategy: Fine-tune BERT on CLEAN text from Indian domain
# This teaches BERT the Indian language context patterns
# Then use BERT fill-mask to correct errors at inference

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model     = BertForMaskedLM.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples["clean"],      # Train on CLEAN text
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_special_tokens_mask=True,
    )

print("Tokenizing clean text...")
tokenized = correction_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["noisy", "clean"],
)

# MLM data collator — randomly masks 15% of tokens
# BERT learns to predict the correct tokens from context
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# ── CELL 8: Training ─────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = "./bert-asr-correction",
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 32,
    num_train_epochs        = 3,
    learning_rate           = 2e-5,
    weight_decay            = 0.01,
    warmup_ratio            = 0.1,
    fp16                    = True,
    logging_steps           = 100,
    save_steps              = 500,
    evaluation_strategy     = "epoch",
    push_to_hub             = True,
    hub_model_id            = HF_REPO_ID,
)

# Split for eval
split = correction_dataset.train_test_split(test_size=0.1)
train_tok = split["train"].map(tokenize_function, batched=True, remove_columns=["noisy","clean"])
eval_tok  = split["test"].map(tokenize_function,  batched=True, remove_columns=["noisy","clean"])

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tok,
    eval_dataset    = eval_tok,
    data_collator   = data_collator,
)

print("Training BERT for ASR correction...")
trainer.train()
trainer.push_to_hub()

# ── CELL 9: Inference — BERT fill-mask correction ─────────────────────────
from transformers import pipeline as hf_pipeline

# Load the fill-mask pipeline
fill_mask = hf_pipeline("fill-mask", model=HF_REPO_ID)

def bert_correct_word(sentence: str, word_index: int) -> str:
    """Replace a specific word with [MASK] and let BERT predict."""
    words = sentence.split()
    original_word = words[word_index]
    words[word_index] = "[MASK]"
    masked = " ".join(words)

    predictions = fill_mask(masked)
    # Return top prediction
    return predictions[0]["token_str"]

def correct_asr_output(noisy_text: str, confidence_threshold: float = 0.5) -> str:
    """
    Full ASR correction pipeline using BERT.
    In production: uses Whisper word-level confidence scores
    to decide WHICH words to mask.
    """
    words = noisy_text.split()
    corrected = []

    for i, word in enumerate(words):
        # Simple heuristic: try correcting short words (common errors)
        if len(word) < 4:
            corrected.append(word)
            continue

        # Create masked version
        masked_words = words.copy()
        masked_words[i] = "[MASK]"
        masked_text = " ".join(masked_words)

        try:
            preds = fill_mask(masked_text)
            top_pred = preds[0]
            # Only replace if confidence is high enough
            if top_pred["score"] > confidence_threshold:
                corrected.append(top_pred["token_str"])
            else:
                corrected.append(word)
        except:
            corrected.append(word)

    return " ".join(corrected)

# Test
test_noisy = "the whether is very god today in deli the templ is beautiful"
corrected  = correct_asr_output(test_noisy)
print(f"\nASR Correction Test:")
print(f"  Noisy:     {test_noisy}")
print(f"  Corrected: {corrected}")

# ── CELL 10: Quantitative Evaluation (Research Paper Metric) ─────────────
wer_metric = evaluate.load("wer")

noisy_samples  = [s["noisy"] for s in correction_dataset.select(range(500))]
clean_samples  = [s["clean"] for s in correction_dataset.select(range(500))]
corrected_samples = [correct_asr_output(s) for s in noisy_samples]

wer_before = wer_metric.compute(predictions=noisy_samples,     references=clean_samples)
wer_after  = wer_metric.compute(predictions=corrected_samples, references=clean_samples)

print(f"\n{'='*55}")
print(f"RESEARCH RESULT — BERT ASR Correction Impact")
print(f"{'='*55}")
print(f"WER Before correction: {wer_before*100:.2f}%")
print(f"WER After  correction: {wer_after*100:.2f}%")
print(f"Improvement:           {(wer_before - wer_after)*100:.2f}% absolute")
print(f"{'='*55}")
print("→ This WER improvement table goes in your research paper!")
