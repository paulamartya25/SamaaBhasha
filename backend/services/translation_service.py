# translation_service.py — Multilingual Translation
# Uses NLLB-200 (Meta) for ALL languages — No API key needed!
# ✅ 200 languages including all 22 Indian languages
# ✅ Completely free, self-hosted, open source
# ✅ No Bhashini/AI4Bharat account needed

from transformers import pipeline as hf_pipeline

# ── NLLB-200 Language Codes ────────────────────────────────────────────────
# Indian Languages
INDIAN_LANG_CODES = {
    "hin_Deva", "tel_Telu", "tam_Taml", "ben_Beng", "mar_Deva",
    "guj_Gujr", "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru",
    "urd_Arab", "asm_Beng", "mai_Deva", "san_Deva", "kas_Arab",
    "kok_Deva", "mni_Mtei", "sat_Olck", "snd_Arab", "npi_Deva",
    "bho_Deva", "doi_Deva",
}

# Whisper detected language name → NLLB-200 code
WHISPER_TO_NLLB = {
    "english":   "eng_Latn",
    "hindi":     "hin_Deva",
    "bengali":   "ben_Beng",
    "telugu":    "tel_Telu",
    "tamil":     "tam_Taml",
    "marathi":   "mar_Deva",
    "gujarati":  "guj_Gujr",
    "kannada":   "kan_Knda",
    "malayalam": "mal_Mlym",
    "odia":      "ory_Orya",
    "punjabi":   "pan_Guru",
    "urdu":      "urd_Arab",
    "assamese":  "asm_Beng",
    "french":    "fra_Latn",
    "german":    "deu_Latn",
    "spanish":   "spa_Latn",
    "japanese":  "jpn_Jpan",
    "chinese":   "zho_Hans",
    "arabic":    "arb_Arab",
    "portuguese":"por_Latn",
    "russian":   "rus_Cyrl",
    "korean":    "kor_Hang",
    "italian":   "ita_Latn",
    "dutch":     "nld_Latn",
    "turkish":   "tur_Latn",
}

# ── Load NLLB-200 model once at startup ───────────────────────────────────
# Using distilled 600M version — lighter, faster, fits in Oracle Cloud VM
_nllb_pipeline = None

def get_nllb():
    global _nllb_pipeline
    if _nllb_pipeline is None:
        print("[Translation] Loading NLLB-200 model (first load may take 2-3 min)...")
        _nllb_pipeline = hf_pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            device=-1,   # CPU — works on Oracle Cloud free VM
            max_length=512,
        )
        print("[Translation] NLLB-200 loaded successfully!")
    return _nllb_pipeline

# ── Main translation function ──────────────────────────────────────────────
async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translates text using NLLB-200 (No API key needed).
    Supports 200 languages including all 22 Indian languages.

    Args:
        text:        Input text to translate
        source_lang: Whisper detected language name (e.g. "french", "hindi")
        target_lang: NLLB language code (e.g. "hin_Deva", "eng_Latn")
    """
    if not text or not text.strip():
        return text

    # Map Whisper language name → NLLB code
    src_code = WHISPER_TO_NLLB.get(source_lang.lower(), "eng_Latn")

    # No translation needed if same language
    if src_code == target_lang:
        return text

    try:
        translator = get_nllb()
        result = translator(
            text,
            src_lang=src_code,
            tgt_lang=target_lang,
            max_length=256,
        )
        return result[0]["translation_text"]

    except Exception as e:
        print(f"[Translation] Error: {e}")
        return text   # Return original text if translation fails
