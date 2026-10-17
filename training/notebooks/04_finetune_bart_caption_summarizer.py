# =============================================================================
# SamaaBhasha — Notebook 4: BART Summarization for Long Captions
# =============================================================================
# Run on Kaggle T4 GPU
# Goal: When someone speaks a very long sentence, BART condenses it into
#       a clean readable caption (abstractive summarization)
#
# Example:
#   Spoken: "So basically what I'm trying to say is that if you look at the
#            data from the perspective of, you know, the overall trend..."
#   Caption: "The data shows an overall positive trend."
# =============================================================================

# ── CELL 1: Install ───────────────────────────────────────────────────────
# !pip install -q transformers datasets accelerate evaluate rouge_score

# ── CELL 2: Imports ───────────────────────────────────────────────────────
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import evaluate

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ── CELL 3: Config ───────────────────────────────────────────────────────
MODEL_NAME     = "facebook/bart-large-cnn"   # Pre-trained on news summarization
HF_REPO_ID     = "paulamartya25/samaa-bart-caption-summarizer"
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 64   # Short captions!

# ── CELL 4: Load Dataset — XSum for English captions ─────────────────────
# XSum: BBC articles + 1-sentence summaries = perfect for caption training
print("Loading XSum dataset...")
dataset = load_dataset("xsum", split="train[:30000]")
eval_ds = load_dataset("xsum", split="validation[:3000]")

print(f"Train: {len(dataset)} samples")
print(f"Eval:  {len(eval_ds)} samples")
print(f"Sample input:    {dataset[0]['document'][:200]}...")
print(f"Sample summary:  {dataset[0]['summary']}")

# ── CELL 5: Load tokenizer + model ───────────────────────────────────────
tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
model     = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── CELL 6: Preprocessing for caption-length output ──────────────────────
def preprocess_function(examples):
    # Truncate input to simulate spoken text (not full articles)
    inputs = [
        " ".join(doc.split()[:150])   # First 150 words = ~45 seconds of speech
        for doc in examples["document"]
    ]
    targets = examples["summary"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing...")
train_tok = dataset.map(preprocess_function, batched=True,
                        remove_columns=dataset.column_names)
eval_tok  = eval_ds.map(preprocess_function,  batched=True,
                        remove_columns=eval_ds.column_names)

# ── CELL 7: ROUGE metric (standard for summarization) ────────────────────
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

# ── CELL 8: Training ─────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,
                                        label_pad_token_id=-100)

training_args = Seq2SeqTrainingArguments(
    output_dir              = "./bart-caption-finetuned",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    gradient_accumulation_steps = 2,
    num_train_epochs        = 3,
    learning_rate           = 3e-5,
    fp16                    = True,
    predict_with_generate   = True,
    generation_max_length   = MAX_TARGET_LEN,
    generation_num_beams    = 4,
    evaluation_strategy     = "epoch",
    save_strategy           = "epoch",
    logging_steps           = 100,
    load_best_model_at_end  = True,
    metric_for_best_model   = "rougeL",
    push_to_hub             = True,
    hub_model_id            = HF_REPO_ID,
)

trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_tok,
    eval_dataset    = eval_tok,
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

print("Fine-tuning BART for caption summarization...")
trainer.train()
trainer.push_to_hub()

# ── CELL 9: Test caption generation ──────────────────────────────────────
from transformers import pipeline as hf_pipeline
summarizer = hf_pipeline("summarization", model=HF_REPO_ID)

# Simulate long spoken text (typical video call speech)
test_inputs = [
    """So basically what I wanted to discuss today is the overall performance
       metrics of our team and you know if you look at the data we have been
       consistently improving quarter over quarter which is really great and
       I think we should continue with the current strategy going forward.""",

    """The Eiffel Tower which is located in Paris France was constructed between
       1887 and 1889 as the entrance arch for the 1889 World Fair and it was
       designed by Gustave Eiffel and is now one of the most recognizable
       structures in the entire world attracting millions of visitors every year.""",
]

print("\nCaption Summarization Tests:")
print("=" * 60)
for text in test_inputs:
    result = summarizer(text, max_length=40, min_length=10, do_sample=False)
    print(f"Input:   {text[:100]}...")
    print(f"Caption: {result[0]['summary_text']}")
    print()

# ── CELL 10: Research metrics ─────────────────────────────────────────────
results = trainer.evaluate()
print(f"\nFinal Results:")
print(f"  ROUGE-1: {results['eval_rouge1']:.4f}")
print(f"  ROUGE-2: {results['eval_rouge2']:.4f}")
print(f"  ROUGE-L: {results['eval_rougeL']:.4f}")
print("\n→ These ROUGE scores go in your research paper!")
