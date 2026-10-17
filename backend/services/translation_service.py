# translation_service.py — Multilingual Translation
# Routes to AI4Bharat IndicTrans2 (Indian langs) or Helsinki mBART (world langs)

import os
import httpx
from transformers import pipeline

# ── Language routing ───────────────────────────────────────────────────────
INDIAN_LANG_CODES = {
    "hin_Deva", "tel_Telu", "tam_Taml", "ben_Beng", "mar_Deva",
    "guj_Gujr", "kan_Knda", "mal_Mlym", "ory_Orya", "pan_Guru",
    "urd_Arab", "asm_Beng", "mai_Deva", "san_Deva", "doi_Deva",
    "kas_Arab", "kok_Deva", "mni_Mtei", "sat_Olck", "snd_Arab",
    "npi_Deva", "bho_Deva",
}

AI4BHARAT_API_URL = "https://api.dhruva.ai4bharat.org/services/inference/translation"
AI4BHARAT_API_KEY  = os.getenv("AI4BHARAT_API_KEY", "")

# mBART pipeline (loaded once at startup for world language translation)
_mbart_pipeline = None

def get_mbart():
    global _mbart_pipeline
    if _mbart_pipeline is None:
        print("[Translation] Loading mBART-50 model...")
        _mbart_pipeline = pipeline(
            "translation",
            model="facebook/mbart-large-50-many-to-many-mmt",
            device=-1,  # CPU (Oracle Cloud free VM has no GPU)
        )
    return _mbart_pipeline

# ── Whisper → IndicTrans2 language code mapping ────────────────────────────
WHISPER_TO_INDIC = {
    "hindi": "hin_Deva", "bengali": "ben_Beng", "telugu": "tel_Telu",
    "tamil": "tam_Taml", "marathi": "mar_Deva", "gujarati": "guj_Gujr",
    "kannada": "kan_Knda", "malayalam": "mal_Mlym", "odia": "ory_Orya",
    "punjabi": "pan_Guru", "urdu": "urd_Arab", "assamese": "asm_Beng",
}

# ── Main translation function ──────────────────────────────────────────────
async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Routes translation:
    - Indian target lang → AI4Bharat IndicTrans2 API (free)
    - World target lang  → mBART-50 (self-hosted, free)
    """
    if not text.strip():
        return text

    # Map whisper language name to code
    src = WHISPER_TO_INDIC.get(source_lang.lower(), source_lang)

    # Same language — no translation needed
    if src == target_lang or source_lang == target_lang:
        return text

    if target_lang in INDIAN_LANG_CODES:
        return await _translate_indic(text, src, target_lang)
    else:
        return await _translate_mbart(text, src, target_lang)

# ── AI4Bharat IndicTrans2 ──────────────────────────────────────────────────
async def _translate_indic(text: str, src: str, tgt: str) -> str:
    """Calls AI4Bharat Dhruva API for Indian language translation."""
    payload = {
        "input": [{"source": text}],
        "config": {
            "serviceId": "ai4bharat/indictrans--gpu-t4",
            "language": {"sourceLanguage": src, "targetLanguage": tgt},
        },
    }
    headers = {"Authorization": f"Bearer {AI4BHARAT_API_KEY}"}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(AI4BHARAT_API_URL, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()["output"][0]["target"]

# ── mBART-50 world language translation ───────────────────────────────────
async def _translate_mbart(text: str, src: str, tgt: str) -> str:
    """Uses local mBART-50 model for world language translation."""
    # mBART language code format: en_XX, fr_XX, de_DE, etc.
    MBART_CODES = {
        "eng_Latn": "en_XX", "fra_Latn": "fr_XX", "deu_Latn": "de_DE",
        "spa_Latn": "es_XX", "jpn_Jpan": "ja_XX", "zho_Hans": "zh_CN",
        "ara_Arab": "ar_AR", "por_Latn": "pt_XX", "rus_Cyrl": "ru_RU",
        "kor_Hang": "ko_KR", "ita_Latn": "it_IT", "nld_Latn": "nl_XX",
    }
    src_code = MBART_CODES.get(src, "en_XX")
    tgt_code = MBART_CODES.get(tgt, "en_XX")

    translator = get_mbart()
    result = translator(
        text,
        src_lang=src_code,
        tgt_lang=tgt_code,
        max_length=512,
    )
    return result[0]["translation_text"]
