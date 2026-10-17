# CELL 2 — PASTE THIS ENTIRE CODE INTO ONE KAGGLE CELL
# Uses Google FLEURS dataset — NO permission needed, works instantly!

# STEP 1: All Imports
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Audio
import evaluate
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

print("✅ Step 1: All imports done!")

# STEP 2: HuggingFace Login
secrets = UserSecretsClient()
HF_TOKEN = secrets.get_secret("HF_TOKEN")
login(token=HF_TOKEN)
print("✅ Step 2: HuggingFace login done!")

# STEP 3: Check GPU
assert torch.cuda.is_available(), "No GPU! Enable it in Session Options"
print(f"✅ Step 3: GPU = {torch.cuda.get_device_name(0)}")

# STEP 4: Config
LANGUAGE      = "hi_in"          # Hindi India (FLEURS format)
LANGUAGE_NAME = "Hindi"
MODEL_NAME    = "openai/whisper-small"
HF_REPO_ID    = "paulamartya25/samaa-whisper-hindi"
print(f"✅ Step 4: Config done. Repo = {HF_REPO_ID}")

# STEP 5: Load Dataset (Google FLEURS — No permission needed!)
print("⏳ Step 5: Loading Google FLEURS Hindi dataset...")
raw = DatasetDict()
raw["train"] = load_dataset(
    "google/fleurs",
    LANGUAGE,
    split="train",
    trust_remote_code=True
)
raw["test"] = load_dataset(
    "google/fleurs",
    LANGUAGE,
    split="test",
    trust_remote_code=True
)
raw = raw.cast_column("audio", Audio(sampling_rate=16000))
print(f"✅ Step 5: Dataset loaded! Train={len(raw['train'])} Test={len(raw['test'])}")

# STEP 6: Load Processor
print("⏳ Step 6: Loading Whisper processor...")
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME, language=LANGUAGE_NAME, task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=LANGUAGE_NAME, task="transcribe"
)
print("✅ Step 6: Processor loaded!")

# STEP 7: Preprocess
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # FLEURS uses "transcription" column
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

print("⏳ Step 7: Preprocessing (5-10 mins)...")
raw = raw.map(
    prepare_dataset,
    remove_columns=raw.column_names["train"],
    num_proc=2
)
print("✅ Step 7: Preprocessing done!")

# STEP 8: Data Collator
@dataclass
class SpeechCollator:
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

data_collator = SpeechCollator(processor=processor)
print("✅ Step 8: Data collator ready!")

# STEP 9: Load Model
print("⏳ Step 9: Loading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.freeze_encoder()
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
print(f"✅ Step 9: Model loaded! Params={sum(p.numel() for p in model.parameters()):,}")

# STEP 10: Metric
wer_metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

print("✅ Step 10: WER metric ready!")

# STEP 11: Training Arguments
args = Seq2SeqTrainingArguments(
    output_dir="./whisper-hindi",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=200,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HF_REPO_ID,
)
print("✅ Step 11: Training args set!")

# STEP 12: Create Trainer
trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=raw["train"],
    eval_dataset=raw["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
print("✅ Step 12: Trainer created!")

# STEP 13: START TRAINING
print("\n========================================")
print("✅ ALL STEPS DONE — STARTING TRAINING!")
print("   Time needed: ~2-3 hours")
print("   You can close the tab safely!")
print("========================================\n")
trainer.train()

# STEP 14: Save to HuggingFace
trainer.push_to_hub()
processor.push_to_hub(HF_REPO_ID)
print(f"\n✅ Model saved! https://huggingface.co/{HF_REPO_ID}")

# STEP 15: Final Results
results = trainer.evaluate()
print(f"\n{'='*50}")
print(f"FINAL WER  : {results['eval_wer']:.2f}%")
print(f"Baseline   : ~22% (original Whisper-small)")
print(f"{'='*50}")
