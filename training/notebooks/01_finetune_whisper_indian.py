# =============================================================================
# SamaaBhasha — Notebook 1: Fine-tune Whisper on Indian Speech
# =============================================================================
# Run this on Kaggle with T4 GPU (30 hrs/week free)
# Dataset: Kathbath (IIT Madras) + Mozilla Common Voice Indian languages
# Goal:    Improve Whisper's accuracy on Indian-accented speech
#
# INSTRUCTIONS FOR KAGGLE:
# 1. Go to kaggle.com → New Notebook
# 2. Enable GPU: Settings → Accelerator → GPU T4 x2
# 3. Paste this code cell by cell
# 4. After training, save model to HuggingFace Hub
# =============================================================================

# ── CELL 1: Install dependencies ──────────────────────────────────────────
# !pip install -q transformers datasets accelerate evaluate jiwer
# !pip install -q huggingface_hub soundfile librosa

# ── CELL 2: Imports ───────────────────────────────────────────────────────
import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# ── CELL 3: Configuration ─────────────────────────────────────────────────
# Choose language to fine-tune on
# Options: "hi" (Hindi), "te" (Telugu), "ta" (Tamil), "bn" (Bengali),
#          "mr" (Marathi), "gu" (Gujarati), "kn" (Kannada), "ml" (Malayalam)

LANGUAGE       = "hi"          # Start with Hindi — most data available
LANGUAGE_NAME  = "Hindi"
TASK           = "transcribe"
MODEL_NAME     = "openai/whisper-small"   # small = 244M params, fits in T4
HF_REPO_ID     = "paulamartya25/samaa-whisper-hindi"   # your HuggingFace repo

# ── CELL 4: Load Dataset (Mozilla Common Voice — Free) ────────────────────
print("Loading dataset...")

# Mozilla Common Voice — Hindi
# Free, high quality, 1000+ hours across Indian languages
common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    LANGUAGE,
    split="train+validation",
    use_auth_token=True,   # log in via: huggingface-cli login
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    LANGUAGE,
    split="test",
    use_auth_token=True,
)

# Remove unused columns to save memory
cols_to_remove = ["accent", "age", "client_id", "down_votes", "gender",
                  "locale", "path", "segment", "up_votes"]
common_voice = common_voice.remove_columns(
    [c for c in cols_to_remove if c in common_voice["train"].column_names]
)
common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(f"Train samples: {len(common_voice['train'])}")
print(f"Test samples:  {len(common_voice['test'])}")

# ── CELL 5: Load Whisper processor ────────────────────────────────────────
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME, language=LANGUAGE_NAME, task=TASK
)
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=LANGUAGE_NAME, task=TASK
)

# ── CELL 6: Preprocessing function ───────────────────────────────────────
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("Preprocessing dataset (this takes ~10 min)...")
common_voice = common_voice.map(
    prepare_dataset,
    remove_columns=common_voice.column_names["train"],
    num_proc=2,
)
print("Preprocessing done!")

# ── CELL 7: Data collator (handles padding) ───────────────────────────────
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        # Pad input features
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ── CELL 8: WER metric (Word Error Rate — lower is better) ────────────────
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ── CELL 9: Load model ────────────────────────────────────────────────────
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Freeze encoder — only fine-tune decoder (saves GPU memory, faster training)
model.freeze_encoder()
model.model.encoder.gradient_checkpointing = True

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

print(f"Model loaded: {MODEL_NAME}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── CELL 10: Training arguments ───────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir         = "./whisper-finetuned-hindi",
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 1,
    learning_rate      = 1e-5,
    warmup_steps       = 500,
    max_steps          = 4000,       # ~2 hours on T4 — fits in Kaggle session
    gradient_checkpointing = True,
    fp16               = True,       # Mixed precision — faster on T4
    evaluation_strategy = "steps",
    per_device_eval_batch_size = 8,
    predict_with_generate = True,
    generation_max_length = 225,
    save_steps         = 1000,
    eval_steps         = 1000,
    logging_steps      = 100,
    report_to          = ["tensorboard"],
    load_best_model_at_end = True,
    metric_for_best_model = "wer",
    greater_is_better  = False,
    push_to_hub        = True,
    hub_model_id       = HF_REPO_ID,
)

# ── CELL 11: Train! ───────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    args            = training_args,
    model           = model,
    train_dataset   = common_voice["train"],
    eval_dataset    = common_voice["test"],
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    tokenizer       = processor.feature_extractor,
)

print("Starting training...")
trainer.train()

# ── CELL 12: Save + push to HuggingFace Hub ───────────────────────────────
print("Saving model to HuggingFace Hub...")
trainer.push_to_hub()
processor.push_to_hub(HF_REPO_ID)
print(f"Model saved: https://huggingface.co/{HF_REPO_ID}")

# ── CELL 13: Evaluate final WER ───────────────────────────────────────────
results = trainer.evaluate()
print(f"\n{'='*50}")
print(f"Final WER (Word Error Rate): {results['eval_wer']:.2f}%")
print(f"(Baseline Whisper-small WER on Hindi: ~18-25%)")
print(f"{'='*50}")

# ── CELL 14: Quick inference test ─────────────────────────────────────────
from transformers import pipeline as hf_pipeline

pipe = hf_pipeline(
    "automatic-speech-recognition",
    model=HF_REPO_ID,
    chunk_length_s=30,
)

# Test with a sample from test set
sample = common_voice["test"][0]
result = pipe(sample["audio"]["array"])
print(f"\nTest transcription: {result['text']}")
