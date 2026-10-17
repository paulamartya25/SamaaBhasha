# main.py — SamaaBhasha FastAPI Backend

import base64
import os
import tempfile
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from services.whisper_service    import transcribe_audio
from services.translation_service import translate_text
from services.bert_service       import correct_transcription

app = FastAPI(
    title="SamaaBhasha API",
    description="Real-time multilingual captioning backend",
    version="1.0.0",
)

# ── CORS (allow extension to call this API) ────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ───────────────────────────────────────────────────────────
@app.get("/")
async def health():
    return {"status": "ok", "service": "SamaaBhasha"}

# ── WebSocket endpoint — real-time captioning ──────────────────────────────
@app.websocket("/ws/caption")
async def websocket_caption(websocket: WebSocket, lang: str = "eng_Latn"):
    await websocket.accept()
    print(f"[WS] Client connected — target lang: {lang}")

    try:
        while True:
            # Receive JSON with base64 audio + target language
            data = await websocket.receive_json()
            audio_b64  = data.get("audio")
            target_lang = data.get("targetLang", lang)

            if not audio_b64:
                continue

            # 1. Decode base64 audio → temp file
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_bytes)
                tmp_path = f.name

            try:
                # 2. Whisper STT (via Groq API)
                stt_result = await transcribe_audio(tmp_path)
                original_text   = stt_result["text"]
                detected_lang   = stt_result["language"]

                if not original_text.strip():
                    continue

                # 3. BERT context correction
                corrected_text = await correct_transcription(original_text)

                # 4. Translate to target language
                caption = await translate_text(corrected_text, detected_lang, target_lang)

                # 5. Send caption back to extension
                await websocket.send_json({
                    "caption":       caption,
                    "original":      original_text,
                    "detected_lang": detected_lang,
                })

            finally:
                os.unlink(tmp_path)  # clean up temp file

    except WebSocketDisconnect:
        print("[WS] Client disconnected")

# ── REST endpoint — single translation (optional) ──────────────────────────
@app.post("/translate")
async def translate_endpoint(payload: dict):
    text        = payload.get("text", "")
    source_lang = payload.get("source_lang", "auto")
    target_lang = payload.get("target_lang", "eng_Latn")
    result = await translate_text(text, source_lang, target_lang)
    return {"translation": result}
