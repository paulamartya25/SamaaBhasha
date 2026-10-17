# 🌐 SamaaBhasha — समाभाषा
### *Real-Time Multilingual Caption System for Video Calls & Streaming*

> **SamaaBhasha** (Sanskrit: *Universal Language*) is an open-source browser extension that provides real-time captions in 22 Indian languages + 50+ world languages — directly on Google Meet, Zoom, Microsoft Teams, and YouTube.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?logo=fastapi)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-red?logo=googlechrome)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## ✨ Features

- 🎙️ **Real-time Speech-to-Text** using OpenAI Whisper
- 🌍 **Auto language detection** — speaks French? German? Japanese? Detected instantly
- 🇮🇳 **22 Indian Languages** via AI4Bharat IndicTrans2 (Hindi, Telugu, Tamil, Bengali, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi, Urdu + more)
- 🌐 **50+ World Languages** via mBART-50
- 🧠 **BERT context correction** — fixes ASR errors intelligently
- 📝 **BART summarization** — condenses long sentences for clean captions
- 🖥️ **Works everywhere** — Google Meet, Zoom, Teams, YouTube, Twitch, any browser tab
- ⚡ **WebSocket streaming** — ultra-low latency real-time captions
- 💾 **Transcript download** — save full conversation as text

---

## 🏗️ Architecture

```
Browser Extension (Chrome/Firefox)
        ↓  WebSocket
FastAPI Backend (Oracle Cloud Free VM)
        ↓
┌───────────────────────────────┐
│  Groq Whisper  →  STT        │
│  BERT          →  Correction  │
│  mBART-50      →  World langs │
│  IndicTrans2   →  Indian langs│
└───────────────────────────────┘
```

---

## 📁 Project Structure

```
SamaaBhasha/
├── extension/              # Chrome/Firefox Extension (Manifest V3)
│   ├── manifest.json
│   ├── popup/              # Extension popup UI
│   ├── content/            # Content script (caption overlay)
│   ├── background/         # Service worker
│   └── offscreen/          # Audio capture
├── backend/                # FastAPI + WebSocket server
│   ├── main.py
│   ├── routers/
│   │   ├── transcribe.py   # Whisper STT endpoint
│   │   └── translate.py    # Translation endpoint
│   ├── services/
│   │   ├── whisper_service.py
│   │   ├── bert_service.py
│   │   └── translation_service.py
│   ├── requirements.txt
│   └── Dockerfile
├── training/               # Kaggle training notebooks
│   └── notebooks/
│       └── train_mbart_indic.ipynb
└── docs/
    └── architecture.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/paulamartya25/SamaaBhasha.git
cd SamaaBhasha
```

### 2. Setup Backend
```bash
cd backend
pip install -r requirements.txt
# Add your API keys in .env
uvicorn main:app --reload
```

### 3. Load Extension
1. Open Chrome → `chrome://extensions/`
2. Enable **Developer Mode**
3. Click **Load unpacked** → select `extension/` folder

---

## 🌍 Supported Languages

| Indian Languages | World Languages |
|-----------------|-----------------|
| Hindi, Bengali, Telugu, Tamil | French, German, Spanish |
| Marathi, Gujarati, Kannada | Japanese, Chinese, Arabic |
| Malayalam, Odia, Punjabi | Portuguese, Russian, Korean |
| Urdu, Assamese, Maithili | Italian, Dutch, Turkish |
| Sanskrit, Dogri, Kashmiri | + 40 more via mBART-50 |

---

## 🤝 Contributing

Pull requests are welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md).

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👨‍💻 Author

**Paula Martya** — [@paulamartya25](https://github.com/paulamartya25)

> *Built as a research project for Masters in AI/ML*
