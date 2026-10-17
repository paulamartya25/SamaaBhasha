# whisper_service.py — Speech-to-Text via Groq Whisper API (Free Tier)

import os
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

async def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribes audio file using Groq's Whisper API.
    Returns { "text": ..., "language": ... }
    
    Groq free tier: 7,200 seconds of audio / day (~2 hours)
    Model: whisper-large-v3-turbo (fastest)
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment variables")

    async with httpx.AsyncClient(timeout=30) as client:
        with open(audio_path, "rb") as audio_file:
            response = await client.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                files={"file": ("audio.webm", audio_file, "audio/webm")},
                data={
                    "model":            "whisper-large-v3-turbo",
                    "response_format":  "verbose_json",  # includes detected language
                    "temperature":      "0",
                },
            )
            response.raise_for_status()
            result = response.json()

    return {
        "text":     result.get("text", "").strip(),
        "language": result.get("language", "unknown"),
    }
