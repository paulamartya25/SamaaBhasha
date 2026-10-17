# PASTE THIS AS CELL 2 IN KAGGLE
# Make sure Cell 1 (pip install) ran AND kernel restarted first!

# STEP 1: Imports - ALL in one place
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, DatasetDict, Audio
import evaluate

# Import ALL transformers components together
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

print("All imports successful!")

# STEP 2: Login
secrets = UserSecretsClient()
HF_TOKEN = secrets.get_secret("HF_TOKEN")
login(token=HF_TOKEN)
print("HuggingFace login done!")

# STEP 3: Check GPU
print(f"GPU: {torch.cuda.get_device_name(0)}")

# STEP 4: Settings
LANGUAGE = "hi"
LANGUAGE_NAME = "Hindi"
MODEL_NAME = "openai/whisper-small"
HF_REPO_ID = "paulamartya25/samaa-whisper-hindi"
print(f"Settings done. Repo: {HF_REPO_ID}")

# STEP 5: Load dataset
print("Loading dataset...")
raw = DatasetDict()
raw["train"] = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    LANGUAGE,
    split="train+validation",
    trust_remote_code=True
)
raw["test"] = load_dataset(
    "mozilla-foundation/common_voice_13_0",
    LANGUAGE,
    split="test",
    trust_remote_code=True
)
cols_remove = ["accent","age","client_id","down_votes",
               "gender","locale","path","segment","up_votes"]
raw = raw.remove_columns(
    [c for c in cols_remove if c in raw["train"].column_names]
)
raw = raw.cast_column("audio", Audio(sampling_rate=16000))
print(f"Train: {len(raw['train'])} | Test: {len(raw['test'])}")

# STEP 6: Processor
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(
    MODEL_NAME, language=LANGUAGE_NAME, task="transcribe"
)
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, language=LANGUAGE_NAME, task="transcribe"
)
print("Processor loaded!")

# STEP 7: Preprocess
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("Preprocessing... (takes 10-15 mins)")
raw = raw.map(
    prepare_dataset,
    remove_columns=raw.column_names["train"],
    num_proc=2
)
print("Preprocessing done!")

# STEP 8: Data collator
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
print("Data collator ready!")

# STEP 9: Model
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
model.freeze_encoder()
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
print("Model loaded!")

# STEP 10: Metric
wer = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * wer.compute(predictions=pred_str, references=label_str)}

# STEP 11: Training arguments
args = Seq2SeqTrainingArguments(
    output_dir="./whisper-hindi",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=HF_REPO_ID,
)
print("Training args set!")

# STEP 12: Create trainer
trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=raw["train"],
    eval_dataset=raw["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)
print("Trainer created!")

# STEP 13: TRAIN!
print("\n========================================")
print("TRAINING STARTED - takes 3-4 hours")
print("You can close the tab - Kaggle runs it!")
print("========================================\n")
trainer.train()

# STEP 14: Save
trainer.push_to_hub()
processor.push_to_hub(HF_REPO_ID)
print(f"Model saved: https://huggingface.co/{HF_REPO_ID}")

# STEP 15: Results
results = trainer.evaluate()
print(f"\nFINAL WER: {results['eval_wer']:.2f}%")
print("Baseline Whisper-small WER on Hindi: ~22%")
print(f"Your improvement: {max(0, 22 - results['eval_wer']):.2f}%")
