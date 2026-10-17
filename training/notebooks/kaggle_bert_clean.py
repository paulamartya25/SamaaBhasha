# ============================================================
# CELL 1 — Install Dependencies
# Run this cell FIRST, then restart the kernel!
# ============================================================
!pip install -q transformers datasets accelerate evaluate jiwer


# ============================================================
# CELL 2 — Restart Kernel After Install
# After Cell 1 finishes → Kernel → Restart → Run from Cell 3
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
import re
import random
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import evaluate
print("✅ All imports done!")


# ============================================================
# CELL 6 — Config
# ============================================================
MODEL_NAME = "bert-base-multilingual-cased"   # mBERT — 104 languages
HF_REPO_ID = "paulamartya25/samaa-bert-corrector"
MAX_LENGTH = 128
DATASET_SIZE = 50000

print(f"✅ Config ready!")
print(f"   Model      : {MODEL_NAME}")
print(f"   Output repo: {HF_REPO_ID}")


# ============================================================
# CELL 7 — ASR Error Injection Function
# This creates synthetic (noisy → clean) training pairs
# Simulates mistakes Whisper makes on Indian-accented speech
# THIS IS YOUR NOVEL RESEARCH CONTRIBUTION 🔥
# ============================================================
def inject_asr_errors(text: str, error_rate: float = 0.15) -> str:
    """
    Injects realistic ASR-style errors into clean text.
    Simulates what Whisper produces on Indian-accented speech.
    """
    words = text.split()
    if not words:
        return text

    noisy = []
    for word in words:
        r = random.random()

        # Word deletion (Whisper sometimes drops words)
        if r < error_rate * 0.2:
            continue

        # Word repetition (stuttering/repetition error)
        if r < error_rate * 0.3:
            noisy.extend([word, word])
            continue

        # Character swap (phonetic confusion)
        if r < error_rate * 0.7 and len(word) > 3:
            chars = list(word)
            i = random.randint(0, len(chars) - 2)
            chars[i], chars[i+1] = chars[i+1], chars[i]
            word = "".join(chars)

        noisy.append(word)

    return " ".join(noisy) if noisy else text

# Test the function
sample = "The weather in Delhi is very good today the temple is beautiful"
noisy  = inject_asr_errors(sample)
print(f"\n✅ Error injection working!")
print(f"   Clean : {sample}")
print(f"   Noisy : {noisy}")


# ============================================================
# CELL 8 — Load Text Dataset (Wikipedia Hindi)
# Using Wikipedia Hindi — clean, diverse text corpus
# No API key or special access needed!
# ============================================================
print("Loading Hindi Wikipedia dataset...")

wiki_data = load_dataset(
    "wikipedia",
    "20220301.hi",     # Hindi Wikipedia
    split=f"train[:{DATASET_SIZE}]",
    trust_remote_code=True,
)

print(f"✅ Loaded {len(wiki_data)} articles")
print(f"   Sample: {wiki_data[0]['text'][:200]}")


# ============================================================
# CELL 9 — Create (Noisy, Clean) Training Pairs
# ============================================================
clean_texts = []
noisy_texts = []

print("Creating noisy-clean training pairs...")
for item in wiki_data:
    # Split article into sentences
    sentences = item["text"].split("।")   # Hindi sentence separator
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 5 or len(sent) > 200:
            continue
        noisy = inject_asr_errors(sent, error_rate=0.15)
        clean_texts.append(sent)
        noisy_texts.append(noisy)
        if len(clean_texts) >= 80000:
            break
    if len(clean_texts) >= 80000:
        break

print(f"✅ Created {len(clean_texts)} training pairs")
print(f"\nSample pair:")
print(f"   Clean : {clean_texts[0]}")
print(f"   Noisy : {noisy_texts[0]}")

# Create HuggingFace Dataset
correction_dataset = Dataset.from_dict({
    "clean": clean_texts,
    "noisy": noisy_texts,
})


# ============================================================
# CELL 10 — Load mBERT Tokenizer and Model
# ============================================================
print("Loading mBERT tokenizer and model...")

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
model     = BertForMaskedLM.from_pretrained(MODEL_NAME)

print(f"✅ mBERT loaded!")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# CELL 11 — Tokenize Dataset
# Train on CLEAN text — BERT learns correct language patterns
# ============================================================
def tokenize(examples):
    return tokenizer(
        examples["clean"],       # Train on CLEAN text
        truncation              = True,
        max_length              = MAX_LENGTH,
        padding                 = "max_length",
        return_special_tokens_mask = True,
    )

print("Tokenizing dataset...")
split      = correction_dataset.train_test_split(test_size=0.1, seed=42)
train_tok  = split["train"].map(tokenize, batched=True,
                                remove_columns=["clean", "noisy"])
eval_tok   = split["test"].map(tokenize,  batched=True,
                               remove_columns=["clean", "noisy"])
print(f"✅ Tokenized! Train: {len(train_tok)} | Eval: {len(eval_tok)}")


# ============================================================
# CELL 12 — MLM Data Collator
# Randomly masks 15% of tokens → BERT learns to predict them
# This teaches BERT correct word usage from context
# ============================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer       = tokenizer,
    mlm             = True,
    mlm_probability = 0.15,
)
print("✅ MLM data collator ready!")


# ============================================================
# CELL 13 — Training Arguments
# ============================================================
training_args = TrainingArguments(
    output_dir                  = "./bert-asr-corrector",
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 32,
    num_train_epochs            = 3,
    learning_rate               = 2e-5,
    weight_decay                = 0.01,
    warmup_ratio                = 0.1,
    fp16                        = True,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    logging_steps               = 100,
    load_best_model_at_end      = True,
    push_to_hub                 = True,
    hub_model_id                = HF_REPO_ID,
)
print("✅ Training arguments set!")


# ============================================================
# CELL 14 — Create Trainer
# ============================================================
trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_tok,
    eval_dataset  = eval_tok,
    data_collator = data_collator,
)
print("✅ Trainer created!")


# ============================================================
# CELL 15 — START TRAINING 🚀
# Takes ~2 hours on Kaggle T4 GPU
# ============================================================
print("🚀 Starting BERT training...")
print("   Estimated time: 2 hours")
print("   You can close the tab — Kaggle runs in background!")
trainer.train()
print("✅ Training complete!")


# ============================================================
# CELL 16 — Save to HuggingFace Hub
# ============================================================
print("Saving model to HuggingFace Hub...")
trainer.push_to_hub()
tokenizer.push_to_hub(HF_REPO_ID)
print(f"✅ Saved! View at: https://huggingface.co/{HF_REPO_ID}")


# ============================================================
# CELL 17 — Test ASR Correction
# See how BERT corrects noisy speech transcriptions
# ============================================================
from transformers import pipeline as hf_pipeline

print("Loading your fine-tuned BERT for testing...")
fill_mask = hf_pipeline(
    "fill-mask",
    model  = HF_REPO_ID,
    device = 0,   # GPU
)

def correct_word(sentence: str, word_idx: int) -> str:
    """Replace word at index with [MASK] and let BERT predict."""
    words = sentence.split()
    if word_idx >= len(words):
        return sentence
    words[word_idx] = "[MASK]"
    masked = " ".join(words)
    try:
        preds = fill_mask(masked)
        return preds[0]["token_str"]
    except:
        return sentence.split()[word_idx]

# Test on noisy samples
wer_metric = evaluate.load("wer")
test_pairs = list(zip(noisy_texts[:200], clean_texts[:200]))

corrected = []
for noisy, _ in test_pairs:
    words = noisy.split()
    fixed = []
    for i, w in enumerate(words):
        if len(w) > 3 and random.random() < 0.3:
            fixed.append(correct_word(noisy, i))
        else:
            fixed.append(w)
    corrected.append(" ".join(fixed))

wer_noisy     = wer_metric.compute(
    predictions=[p[0] for p in test_pairs],
    references= [p[1] for p in test_pairs]
)
wer_corrected = wer_metric.compute(
    predictions=corrected,
    references= [p[1] for p in test_pairs]
)

print(f"\n{'='*55}")
print(f"  RESULTS — BERT ASR Correction Impact")
print(f"{'='*55}")
print(f"  WER Before correction : {wer_noisy*100:.2f}%")
print(f"  WER After  correction : {wer_corrected*100:.2f}%")
print(f"  Improvement           : {(wer_noisy-wer_corrected)*100:.2f}%")
print(f"{'='*55}")
print("  → This table goes in your research paper!")
