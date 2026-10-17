##############################################################
# SAMAABHASHA — NOTEBOOK 4: BART CAPTION SUMMARIZER
# Dataset: XSum (BBC) — No permissions needed!
# Time: ~2-3 hours | GPU: T4 required
#
# HOW TO USE IN KAGGLE:
# 1. New Notebook → Enable GPU T4
# 2. Add-ons → Secrets → Add HF_TOKEN
# 3. Paste this ENTIRE code into ONE cell
# 4. Click Run ▶
##############################################################

import subprocess, sys

print("Installing packages...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "evaluate", "rouge_score", "transformers", "datasets",
    "accelerate", "huggingface_hub"], check=True)
print("✅ Packages ready!")

# ALL IMPORTS
import torch
import numpy as np
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
from datasets import load_dataset
import evaluate
from transformers import BartForConditionalGeneration
from transformers import BartTokenizerFast
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
print("✅ All imports done!")

# LOGIN
secrets = UserSecretsClient()
login(token=secrets.get_secret("HF_TOKEN"))
print("✅ HuggingFace login done!")

# GPU CHECK
assert torch.cuda.is_available(), "❌ Enable GPU: Session Options → Accelerator → T4"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# CONFIG
MODEL_NAME     = "facebook/bart-large-cnn"
HF_REPO_ID     = "paulamartya25/samaa-bart-captioner"
TRAIN_SIZE     = 20000
EVAL_SIZE      = 2000
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 64
print(f"✅ Config set! Repo: {HF_REPO_ID}")

# LOAD DATASET — XSum (BBC short summaries)
print("\n⏳ Loading XSum dataset...")
train_ds = load_dataset("xsum", split=f"train[:{TRAIN_SIZE}]",
                         trust_remote_code=True)
eval_ds  = load_dataset("xsum", split=f"validation[:{EVAL_SIZE}]",
                         trust_remote_code=True)
print(f"✅ Dataset loaded! Train={len(train_ds)} | Eval={len(eval_ds)}")
print(f"   Sample input  : {train_ds[0]['document'][:80]}...")
print(f"   Sample caption: {train_ds[0]['summary']}")

# LOAD TOKENIZER & MODEL
print("\n⏳ Loading BART model...")
tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
model     = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
print(f"✅ BART loaded! Params={sum(p.numel() for p in model.parameters()):,}")

# PREPROCESS
def preprocess(examples):
    # Use first 150 words to simulate spoken text (~45 seconds of speech)
    inputs  = [" ".join(d.split()[:150]) for d in examples["document"]]
    targets = examples["summary"]
    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT_LEN, truncation=True, padding=False)
    labels = tokenizer(
        text_target=targets, max_length=MAX_TARGET_LEN,
        truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("⏳ Tokenizing datasets...")
train_tok = train_ds.map(preprocess, batched=True,
                          remove_columns=train_ds.column_names)
eval_tok  = eval_ds.map(preprocess,  batched=True,
                          remove_columns=eval_ds.column_names)
print("✅ Tokenization done!")

# METRIC — ROUGE
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

# DATA COLLATOR
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
print("✅ Data collator ready!")

# TRAINING ARGS
args = Seq2SeqTrainingArguments(
    output_dir                  = "./bart-captioner",
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 4,
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
print("✅ Training args set!")

# CREATE TRAINER
trainer = Seq2SeqTrainer(
    model           = model,
    args            = args,
    train_dataset   = train_tok,
    eval_dataset    = eval_tok,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
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
tokenizer.push_to_hub(HF_REPO_ID)
print(f"\n✅ Model saved! → https://huggingface.co/{HF_REPO_ID}")

# RESULTS
results = trainer.evaluate()
print(f"\n{'='*50}")
print(f"FINAL ROUGE-1 : {results['eval_rouge1']:.4f}")
print(f"FINAL ROUGE-2 : {results['eval_rouge2']:.4f}")
print(f"FINAL ROUGE-L : {results['eval_rougeL']:.4f}")
print(f"{'='*50}")
print("→ Copy these into your research paper!")

# TEST CAPTION GENERATION
from transformers import pipeline as hf_pipeline
captioner = hf_pipeline("summarization", model=HF_REPO_ID, device=0)
test = ("So basically what I wanted to discuss today is the overall "
        "performance of our quarterly results and I think we should "
        "continue with the current strategy as it seems to be working well.")
caption = captioner(test, max_length=30, min_length=8, do_sample=False)
print(f"\nTest caption: {caption[0]['summary_text']}")
