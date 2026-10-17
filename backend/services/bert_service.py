# bert_service.py — BERT-based ASR error correction & context alignment

from transformers import pipeline
import re

# Load once at startup
_fill_mask = None

def get_bert():
    global _fill_mask
    if _fill_mask is None:
        print("[BERT] Loading mBERT fill-mask model...")
        # multilingual BERT — understands 104 languages
        _fill_mask = pipeline(
            "fill-mask",
            model="bert-base-multilingual-cased",
            device=-1,  # CPU
        )
    return _fill_mask

# ── Main correction function ───────────────────────────────────────────────
async def correct_transcription(text: str) -> str:
    """
    Uses BERT to:
    1. Fix common ASR errors (repeated words, fragments)
    2. Clean up punctuation
    3. Handle sentence boundaries
    
    NOTE: For Phase 1, this is a rule-based cleaner.
          Phase 2 will integrate full BERT MLM correction.
    """
    if not text:
        return text

    # Rule-based ASR cleanup
    text = _clean_asr_artifacts(text)

    # BERT context check (Phase 2: uncomment when ready)
    # text = await _bert_correct(text)

    return text

def _clean_asr_artifacts(text: str) -> str:
    """Remove common Whisper ASR artifacts."""
    # Remove repeated words (e.g., "hello hello hello")
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)

    # Remove filler sounds
    fillers = ['um', 'uh', 'hmm', 'ah', 'er', 'like,']
    for f in fillers:
        text = re.sub(rf'\b{f}\b', '', text, flags=re.IGNORECASE)

    # Clean extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]

    return text

# ── Phase 2: Full BERT correction (coming in next sprint) ─────────────────
async def _bert_correct(text: str) -> str:
    """
    Uses BERT fill-mask to verify and correct low-confidence words.
    Will be activated in Phase 2 with confidence scores from Whisper.
    """
    bert = get_bert()
    # TODO: Implement word-level confidence masking and BERT re-scoring
    return text
