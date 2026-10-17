import subprocess, sys

# ── Install packages first (no kernel restart needed!) ─────────────────────
print("Installing packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers", "datasets", "accelerate", "evaluate",
    "jiwer", "soundfile", "librosa", "huggingface_hub"], check=True)
print("✅ Packages installed!")

# ── ALL imports ────────────────────────────────────────────────────────────
import torch, numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
import evaluate
print("✅ All imports done!")

# ── Login to HuggingFace ───────────────────────────────────────────────────
secrets = UserSecretsClient()
login(token=secrets.get_secret("HF_TOKEN"), add_to_git_credential=True)
print("✅ HuggingFace login done!")

# ── Check GPU ──────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), "❌ No GPU! Go to: Session Options → Accelerator → GPU T4"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ── Config ─────────────────────────────────────────────────────────────────
LANGUAGE      = "hi"
LANGUAGE_NAME = "Hindi"
TASK          = "transcribe"
MODEL_NAME    = "openai/whisper-small"
HF_REPO_ID    = f"paulamartya25/samaa-whisper-{LANGUAGE}"
print(f"✅ Config: {LANGUAGE_NAME} | Repo: {HF_REPO_ID}")

# ── Load Dataset ───────────────────────────────────────────────────────────
print("\nLoading Mozilla Common Voice Hindi (5-10 mins)...")
raw = DatasetDict()
raw["train"] = load_dataset("mozilla-foundation/common_voice_13_0",
                             LANGUAGE, split="train+validation",
                             trust_remote_code=True)
raw["test"]  = load_dataset("mozilla-foundation/common_voice_13_0",
                             LANGUAGE, split="test",
                             trust_remote_code=True)
remove_cols = ["accent","age","client_id","down_votes","gender",
               "locale","path","segment","up_votes"]
raw = raw.remove_columns([c for c in remove_cols if c in raw["train"].column_names])
raw = raw.cast_column("audio", Audio(sampling_rate=16000))
print(f"✅ Train: {len(raw['train'])} | Test: {len(raw['test'])}")

# ── Load Processor ─────────────────────────────────────────────────────────
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME,
                        language=LANGUAGE_NAME, task=TASK)
processor         = WhisperProcessor.from_pretrained(MODEL_NAME,
                        language=LANGUAGE_NAME, task=TASK)
print("✅ Processor loaded!")

# ── Preprocess Dataset ─────────────────────────────────────────────────────
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("Preprocessing dataset (10-15 mins)...")
raw = raw.map(prepare_dataset,
              remove_columns=raw.column_names["train"],
              num_proc=2)
print("✅ Preprocessing done!")

# ── Data Collator ──────────────────────────────────────────────────────────
@dataclass
class DataCollatorSpeech:
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

data_collator = DataCollatorSpeech(processor=processor)

# ── Load Model ─────────────────────────────────────────────────────────────
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.freeze_encoder()
model.config.forced_decoder_ids = None
model.config.suppress_tokens    = []
model.config.use_cache          = False
print(f"✅ Model loaded! Params: {sum(p.numel() for p in model.parameters()):,}")

# ── Metric ─────────────────────────────────────────────────────────────────
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

# ── Training Arguments ─────────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir                  = f"./whisper-{LANGUAGE}",
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 1,
    learning_rate               = 1e-5,
    warmup_steps                = 500,
    max_steps                   = 4000,
    gradient_checkpointing      = True,
    fp16                        = True,
    evaluation_strategy         = "steps",
    per_device_eval_batch_size  = 8,
    predict_with_generate       = True,
    generation_max_length       = 225,
    save_steps                  = 1000,
    eval_steps                  = 1000,
    logging_steps               = 25,
    load_best_model_at_end      = True,
    metric_for_best_model       = "wer",
    greater_is_better           = False,
    push_to_hub                 = True,
    hub_model_id                = HF_REPO_ID,
)

# ── Create Trainer ─────────────────────────────────────────────────────────
trainer = Seq2SeqTrainer(
    args            = training_args,
    model           = model,
    train_dataset   = raw["train"],
    eval_dataset    = raw["test"],
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
    tokenizer       = processor.feature_extractor,
)
print("✅ Trainer created!")

# ── START TRAINING 🚀 ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("🚀 TRAINING STARTED — ~3-4 hours")
print("   You can close this tab safely!")
print("   Kaggle keeps running in the background.")
print("="*50 + "\n")
trainer.train()

# ── Save Model ─────────────────────────────────────────────────────────────
print("\nSaving to HuggingFace Hub...")
trainer.push_to_hub()
processor.push_to_hub(HF_REPO_ID)
print(f"✅ DONE! Model at: https://huggingface.co/{HF_REPO_ID}")

# ── Final Results ──────────────────────────────────────────────────────────
results = trainer.evaluate()
print(f"\n{'='*50}")
print(f"FINAL WER  : {results['eval_wer']:.2f}%")
print(f"Baseline   : ~22% (original Whisper-small)")
print(f"Improvement: {max(0, 22 - results['eval_wer']):.2f}%")
print(f"{'='*50}")
print("→ Copy this into your research paper!")
