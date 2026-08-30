# 🎙️ Luna-TTS: High-Performance Local Zero-Shot Voice Cloning Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)

**Luna-TTS** is a lightweight, 100% local, offline Zero-Shot Voice Cloning API server and Web UI powered by **Qwen3-TTS 1.7B Base Q4_K_M**.

It is designed for real-time AI agents (e.g., Hermes Agent, Telegram bots, local assistants) requiring instant, natural, zero-shot voice synthesis **without GPU VRAM usage or cloud service dependencies**.

---

## 🌟 Key Features

* **⚡ Ultra-Fast CPU Inference**: Driven by GGML `Q4_K_M` quantization (1.16 GB model), achieving ~3-10s voice generation on standard multi-core CPUs.
* **🔒 0 MB GPU VRAM Footprint**: Runs 100% on CPU, leaving your GPU VRAM completely free for local LLMs (e.g., Llama, Qwen).
* **🔊 Pre-Baked Voice Profiles (.npz)**: Extract speaker embeddings and codebook tokens once into compact `.npz` files for instant zero-latency loading.
* **📱 Native Telegram OGG Opus Support**: Exports 24kbps OGG Opus audio (`.ogg`), displaying natively as Telegram Voice Notes (round bubbles, ~14KB per clip, non-looping).
* **✂️ Smart Silence Trimming (1.0s Pause)**: Built-in automated FFmpeg `silenceremove` filter that eliminates hallucinated trailing audio, long drags, and trailing silence, keeping a clean 1.0s natural pause.
* **🎯 Smooth Voice Parameter Tuning**: Calibrated sampling parameters (`temperature=0.3`, `top_k=20`, `top_p=0.85`, `repetition_penalty=1.1`) ensuring silky smooth, non-choppy voice synthesis without cut-offs (up to 2048 tokens).
* **💻 Built-in Glassmorphic Web UI**: Includes an interactive Web UI for online TTS testing, voice profile management, and one-click voice cloning.
* **🔌 OpenAI-Compatible API**: Seamlessly integrates into any platform supporting OpenAI `/v1/audio/speech`.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone git@github.com:ryagentai/lunatts.git
cd lunatts

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Service

```bash
python3 start_server.py
```

The server will start listening at `http://127.0.0.1:8890`.

### 3. Open Web UI

Open your browser and navigate to:
```
http://127.0.0.1:8890/
```

---

## 🌐 API Documentation

### `POST /v1/audio/speech` (OpenAI Compatible)
Synthesize speech from text.

```bash
curl -X POST http://127.0.0.1:8890/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "luna-tts",
    "input": "Hello! This is a test of local zero-shot voice cloning.",
    "voice": "sample2",
    "response_format": "ogg"
  }' \
  --output output.ogg
```

### `POST /v1/audio/clone` (Voice Cloning)
Bake a reference audio file into a `.npz` voice profile and archive original reference audio.

```bash
curl -X POST http://127.0.0.1:8890/v1/audio/clone \
  -F "voice_name=my_voice" \
  -F "file=@/path/to/reference.mp3"
```

### `GET /v1/voices`
List all baked voice profiles and reference audio URLs.

---

## 🛠️ Audio & Quality Specifications

| Attribute | Specification |
| :--- | :--- |
| **Output Formats** | `.ogg` (Opus 24kbps) / `.mp3` (64kbps) / `.wav` |
| **Silence Trimming** | Automated `-af silenceremove=stop_periods=-1:stop_duration=1.0:stop_threshold=-35dB` |
| **Sampling Config** | `temperature=0.3`, `top_k=20`, `top_p=0.85`, `repetition_penalty=1.1` |
| **Max Tokens** | Up to `2048` tokens (~2.8 mins of continuous speech) |

---

## 📁 Project Directory Layout

```
lunatts/
├── lunatts_server.py     # FastAPI Server & Glassmorphic Web UI
├── clone_worker.py       # Q4_K_M Inference Engine & Silence Trimmer
├── bake_voice_clone.py   # Speaker Embedding & ASR Baker
├── start_server.py       # Background Service Manager
├── baked_voices/         # Pre-baked .npz Voice Profile Storage
├── ref_audio/            # Archived Original Audio Files
└── requirements.txt      # Dependency Definitions
```

---

## 📄 License

MIT License. Free for personal and commercial use.
