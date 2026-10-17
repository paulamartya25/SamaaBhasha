##############################################################
# SAMAABHASHA — NOTEBOOK 1: WHISPER FINE-TUNING
# Dataset: Google FLEURS (Hindi) — No permissions needed!
# Time: ~2-3 hours | GPU: T4 required
#
# HOW TO USE IN KAGGLE:
# 1. New Notebook → Enable GPU T4
# 2. Add-ons → Secrets → Add HF_TOKEN
# 3. Paste this ENTIRE code into ONE cell
# 4. Click Run ▶
##############################################################

import subprocess, sys

# Install missing packages
print("Installing packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "evaluate", "jiwer", "transformers", "datasets",
    "accelerate", "huggingface_hub"], check=True)
print("✅ Packages ready!")

# ALL IMPORTS — must be after install
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Audio
import evaluate
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
print("✅ All imports done!")

# LOGIN
secrets = UserSecretsClient()
login(token=secrets.get_secret("HF_TOKEN"))
print("✅ HuggingFace login done!")

# GPU CHECK
assert torch.cuda.is_available(), "❌ Enable GPU: Session Options → Accelerator → T4"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# CONFIG
LANGUAGE      = "hi_in"
LANGUAGE_NAME = "Hindi"
MODEL_NAME    = "openai/whisper-small"
HF_REPO_ID    = "paulamartya25/samaa-whisper-hindi"
print(f"✅ Config set! Repo: {HF_REPO_ID}")

# LOAD DATASET — Google FLEURS Hindi
print("\n⏳ Loading Google FLEURS Hindi dataset...")
raw = DatasetDict()
raw["train"] = load_dataset("google/fleurs", LANGUAGE,
                             split="train", trust_remote_code=True)
raw["test"]  = load_dataset("google/fleurs", LANGUAGE,
                             split="test",  trust_remote_code=True)
raw = raw.cast_column("audio", Audio(sampling_rate=16000))
print(f"✅ Dataset loaded! Train={len(raw['train'])} | Test={len(raw['test'])}")

# LOAD PROCESSOR
print("⏳ Loading Whisper processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME, language=LANGUAGE_NAME, task="transcribe")
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=LANGUAGE_NAME, task="transcribe")
print("✅ Processor loaded!")

# PREPROCESS
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

print("⏳ Preprocessing (5-10 mins)...")
raw = raw.map(prepare_dataset,
              remove_columns=raw.column_names["train"],
              num_proc=2)
print("✅ Preprocessing done!")

# DATA COLLATOR
@dataclass
class SpeechCollator:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = SpeechCollator(processor=processor)
print("✅ Data collator ready!")

# LOAD MODEL
print("⏳ Loading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.freeze_encoder()
model.config.forced_decoder_ids = None
model.config.suppress_tokens    = []
model.config.use_cache          = False
print(f"✅ Model loaded! Params={sum(p.numel() for p in model.parameters()):,}")

# METRIC
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

print("✅ WER metric loaded!")

# TRAINING ARGS
args = Seq2SeqTrainingArguments(
    output_dir                  = "./whisper-hindi",
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 1,
    learning_rate               = 1e-5,
    warmup_steps                = 200,
    max_steps                   = 2000,
    gradient_checkpointing      = True,
    fp16                        = True,
    evaluation_strategy         = "steps",
    per_device_eval_batch_size  = 8,
    predict_with_generate       = True,
    generation_max_length       = 225,
    save_steps                  = 500,
    eval_steps                  = 500,
    logging_steps               = 25,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    push_to_hub                 = True,
    hub_model_id                = HF_REPO_ID,
)
print("✅ Training args set!")

# CREATE TRAINER
trainer = Seq2SeqTrainer(
    args            = args,
    model           = model,
    train_dataset   = raw["train"],
    eval_dataset    = raw["test"],
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    tokenizer       = processor.feature_extractor,
)
print("✅ Trainer created!")

# TRAIN!
print("\n" + "="*50)
print("🚀 TRAINING STARTED!")
print("   Time: ~2-3 hours")
print("   Safe to close tab — Kaggle keeps running!")
print("="*50 + "\n")
trainer.train()

# SAVE
trainer.push_to_hub()
processor.push_to_hub(HF_REPO_ID)
print(f"\n✅ Model saved! → https://huggingface.co/{HF_REPO_ID}")

# RESULTS
results = trainer.evaluate()
print(f"\n{'='*50}")
print(f"FINAL WER  : {results['eval_wer']:.2f}%")
print(f"Baseline   : ~22% (Whisper-small original)")
print(f"Improvement: {max(0, 22 - results['eval_wer']):.2f}%")
print(f"{'='*50}")
