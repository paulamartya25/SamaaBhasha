# ============================================================
# CELL 1 — Install Dependencies
# Run this cell FIRST, then restart the kernel!
# ============================================================
!pip install -q transformers datasets accelerate evaluate rouge_score sentencepiece


# ============================================================
# CELL 2 — Restart Kernel After Install
# After Cell 1 finishes → Kernel → Restart → then run from Cell 3
# ============================================================
print("✅ Install done! Now restart kernel and run from Cell 3")


# ============================================================
# CELL 3 — HuggingFace Login
# Add HF_TOKEN in Kaggle: Add-ons → Secrets → Add Secret
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
    BartTokenizerFast,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate
print("✅ All imports done!")


# ============================================================
# CELL 6 — Config
# ============================================================
MODEL_NAME     = "facebook/bart-large-cnn"
HF_REPO_ID     = "paulamartya25/samaa-bart-captioner"
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 64    # Short captions — ideal for screen overlay
TRAIN_SIZE     = 30000
EVAL_SIZE      = 3000

print(f"✅ Config ready!")
print(f"   Model      : {MODEL_NAME}")
print(f"   Output repo: {HF_REPO_ID}")
print(f"   Train size : {TRAIN_SIZE}")


# ============================================================
# CELL 7 — Load XSum Dataset
# XSum = BBC articles + 1-sentence summaries
# Perfect for caption training (short + informative outputs)
# ============================================================
print("Loading XSum dataset...")

train_ds = load_dataset("xsum", split=f"train[:{TRAIN_SIZE}]")
eval_ds  = load_dataset("xsum", split=f"validation[:{EVAL_SIZE}]")

print(f"✅ Train: {len(train_ds)} | Eval: {len(eval_ds)}")
print(f"\nSample input  : {train_ds[0]['document'][:150]}...")
print(f"Sample caption: {train_ds[0]['summary']}")


# ============================================================
# CELL 8 — Load Tokenizer and Model
# ============================================================
print("Loading BART tokenizer and model...")

tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
model     = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

print(f"✅ Model loaded!")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# CELL 9 — Preprocessing Function
# Simulate spoken text by using first 150 words of articles
# This mirrors what a speaker says in ~45 seconds
# ============================================================
def preprocess(examples):
    # Truncate to simulate spoken sentences (not full articles)
    inputs  = [" ".join(doc.split()[:150]) for doc in examples["document"]]
    targets = examples["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length  = MAX_INPUT_LEN,
        truncation  = True,
        padding     = False,
    )
    labels = tokenizer(
        text_target = targets,
        max_length  = MAX_TARGET_LEN,
        truncation  = True,
        padding     = False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing datasets (takes ~5 mins)...")
train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
eval_tok  = eval_ds.map(preprocess,  batched=True, remove_columns=eval_ds.column_names)
print("✅ Tokenization done!")


# ============================================================
# CELL 10 — ROUGE Metric Setup
# ROUGE = standard metric for summarization (higher = better)
# ROUGE-L measures longest common subsequence
# ============================================================
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    return {k: round(v, 4) for k, v in result.items()}

print("✅ ROUGE metric loaded!")


# ============================================================
# CELL 11 — Data Collator
# ============================================================
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model                = model,
    label_pad_token_id   = -100,
    pad_to_multiple_of   = 8,
)
print("✅ Data collator ready!")


# ============================================================
# CELL 12 — Training Arguments
# ============================================================
training_args = Seq2SeqTrainingArguments(
    output_dir                  = "./bart-captioner",
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 4,    # Effective batch = 16
    num_train_epochs            = 3,
    learning_rate               = 3e-5,
    warmup_ratio                = 0.1,
    weight_decay                = 0.01,
    fp16                        = True,
    predict_with_generate       = True,
    generation_max_length       = MAX_TARGET_LEN,
    generation_num_beams        = 4,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    logging_steps               = 100,
    load_best_model_at_end      = True,
    metric_for_best_model       = "rougeL",
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
# Takes ~2-3 hours on Kaggle T4 GPU
# You can close the tab — Kaggle keeps running!
# ============================================================
print("🚀 Starting BART training...")
print("   Estimated time: 2-3 hours")
print("   You can close the tab safely — Kaggle runs in background")
trainer.train()
print("✅ Training complete!")


# ============================================================
# CELL 15 — Save to HuggingFace Hub
# ============================================================
print("Saving model to HuggingFace Hub...")
trainer.push_to_hub()
tokenizer.push_to_hub(HF_REPO_ID)
print(f"✅ Model saved! View at: https://huggingface.co/{HF_REPO_ID}")


# ============================================================
# CELL 16 — Final ROUGE Scores (Research Paper Table)
# ============================================================
results = trainer.evaluate()

print(f"\n{'='*55}")
print(f"  FINAL RESULTS — BART Caption Summarizer")
print(f"{'='*55}")
print(f"  ROUGE-1 : {results['eval_rouge1']:.4f}")
print(f"  ROUGE-2 : {results['eval_rouge2']:.4f}")
print(f"  ROUGE-L : {results['eval_rougeL']:.4f}")
print(f"{'='*55}")
print("  → Copy these numbers into your research paper!")


# ============================================================
# CELL 17 — Test Caption Generation
# Run this to see how your model generates captions!
# ============================================================
from transformers import pipeline as hf_pipeline

print("Loading your fine-tuned model for testing...")
summarizer = hf_pipeline(
    "summarization",
    model      = HF_REPO_ID,
    device     = 0,   # GPU
)

test_texts = [
    """So basically what I wanted to discuss in this meeting today is
       the overall performance of our project and how we have been doing
       over the last few weeks and I think we should continue with the
       current approach because it seems to be working quite well so far.""",

    """The Eiffel Tower located in Paris France was built between 1887 and
       1889 as the entrance arch for the 1889 World Fair and it was designed
       by Gustave Eiffel and attracts millions of visitors every single year.""",

    """Artificial intelligence is transforming every industry including
       healthcare education finance and transportation and it is expected
       to create new jobs while also automating many existing ones in the
       coming years and decades according to researchers.""",
]

print("\n📝 Caption Generation Tests:")
print("=" * 60)
for i, text in enumerate(test_texts, 1):
    result = summarizer(text, max_length=40, min_length=10, do_sample=False)
    print(f"\nTest {i}:")
    print(f"  Input  : {text[:80].strip()}...")
    print(f"  Caption: {result[0]['summary_text']}")
