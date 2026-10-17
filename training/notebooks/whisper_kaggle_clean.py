# ============================================================
# CELL 1 — Install (Run this first, then restart notebook)
# ============================================================
# !pip install -q transformers datasets accelerate evaluate jiwer soundfile librosa


# ============================================================
# CELL 2 — HuggingFace Login
# ============================================================
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

secrets   = UserSecretsClient()
HF_TOKEN  = secrets.get_secret("HF_TOKEN")
login(token=HF_TOKEN, add_to_git_credential=True)
print("✅ Logged in to HuggingFace!")


# ============================================================
# CELL 3 — Check GPU
# ============================================================
import torch
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 4 — Config (change language here)
# ============================================================
LANGUAGE      = "hi"           # hi=Hindi, te=Telugu, ta=Tamil, bn=Bengali
LANGUAGE_NAME = "Hindi"        # Full name
TASK          = "transcribe"
MODEL_NAME    = "openai/whisper-small"
HF_REPO_ID    = f"paulamartya25/samaa-whisper-{LANGUAGE}"
print(f"✅ Config set: {LANGUAGE_NAME} | Model: {MODEL_NAME}")


# ============================================================
# CELL 5 — Load Dataset
# ============================================================
from datasets import load_dataset, DatasetDict, Audio

print("Loading dataset... (takes 5-10 mins)")
raw = DatasetDict()
raw["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    LANGUAGE, split="train+validation",
    trust_remote_code=True
)
raw["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    LANGUAGE, split="test",
    trust_remote_code=True
)

# Remove unused columns
remove_cols = ["accent","age","client_id","down_votes",
               "gender","locale","path","segment","up_votes"]
raw = raw.remove_columns(
    [c for c in remove_cols if c in raw["train"].column_names]
)
raw = raw.cast_column("audio", Audio(sampling_rate=16000))
print(f"✅ Train: {len(raw['train'])} | Test: {len(raw['test'])}")


# ============================================================
# CELL 6 — Load Whisper Processor
# ============================================================
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer         = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE_NAME, task=TASK)
processor         = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE_NAME, task=TASK)
print("✅ Processor loaded!")


# ============================================================
# CELL 7 — Preprocess Dataset
# ============================================================
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("Preprocessing dataset... (10-15 mins)")
raw = raw.map(
    prepare_dataset,
    remove_columns=raw.column_names["train"],
    num_proc=2
)
print("✅ Preprocessing done!")


# ============================================================
# CELL 8 — Data Collator
# ============================================================
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
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
print("✅ Data collator ready!")


# ============================================================
# CELL 9 — Load Model
# ============================================================
from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.freeze_encoder()
model.config.forced_decoder_ids = None
model.config.suppress_tokens    = []
model.config.use_cache          = False
print(f"✅ Model loaded! Params: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# CELL 10 — WER Metric
# ============================================================
import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

print("✅ Metric loaded!")


# ============================================================
# CELL 11 — Training Arguments
# ============================================================
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir                  = f"./whisper-{LANGUAGE}-finetuned",
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
print("✅ Training args set!")


# ============================================================
# CELL 12 — Create Trainer
# ============================================================
from transformers import Seq2SeqTrainer

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


# ============================================================
# CELL 13 — START TRAINING 🚀
# ============================================================
print("🚀 Starting training... (~3-4 hours)")
print("You can close this tab — Kaggle keeps running!")
trainer.train()


# ============================================================
# CELL 14 — Save to HuggingFace Hub
# ============================================================
trainer.push_to_hub()
processor.push_to_hub(HF_REPO_ID)
print(f"✅ Model saved! → https://huggingface.co/{HF_REPO_ID}")


# ============================================================
# CELL 15 — Final WER Score
# ============================================================
results = trainer.evaluate()
print(f"\n{'='*50}")
print(f"✅ Final WER: {results['eval_wer']:.2f}%")
print(f"   Baseline Whisper-small WER on Hindi: ~22%")
print(f"   Your improvement: {22 - results['eval_wer']:.2f}%")
print(f"{'='*50}")
