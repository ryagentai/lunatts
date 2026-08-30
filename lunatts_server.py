#!/usr/bin/env python3
"""
Luna-TTS Official Pre-Baked Voice Inference & Multi-Role Cloning Server (Qwen3-TTS 1.7B Base)
---------------------------------------------------------------------------------------------
Runs a dedicated OpenAI-compatible API server + 9002 PureVision Cyberpunk Web UI listening on port 8890.
100% local, offline Zero-Shot Voice Cloning + Multi-Role Dialogue Synthesis using pre-baked .npz voice profiles.
Uses non-blocking asyncio thread pools so Web UI and health checks never spin or freeze.
"""

import os
import re
import sys
import time
import shutil
import logging
import asyncio
import subprocess
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lunatts_server")

app = FastAPI(title="Luna-TTS Local Voice Engine", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PYTHON_BIN = "/media/ryan/UbuntuDATA/AI_PROJECTS/s2s/venv/bin/python"
WORKER_SCRIPT = "/media/ryan/UbuntuDATA/AI_PROJECTS/lunatts/clone_worker.py"
BAKER_SCRIPT = "/media/ryan/UbuntuDATA/AI_PROJECTS/lunatts/bake_voice_clone.py"
BAKED_VOICES_DIR = "/media/ryan/UbuntuDATA/AI_PROJECTS/lunatts/baked_voices"
REF_AUDIO_DIR = "/media/ryan/UbuntuDATA/AI_PROJECTS/lunatts/ref_audio"

os.makedirs(BAKED_VOICES_DIR, exist_ok=True)
os.makedirs(REF_AUDIO_DIR, exist_ok=True)

class SpeechRequest(BaseModel):
    model: Optional[str] = "luna-tts"
    input: str
    voice: Optional[str] = "sample2"
    ref_audio: Optional[str] = None
    response_format: Optional[str] = "ogg"
    speed: Optional[float] = 1.0

def parse_dialogue_script(text: str, default_voice: str) -> List[Tuple[str, str]]:
    """
    Parses multi-role dialogue script.
    Formats supported:
    1. [voice_name]: text
    2. voice_name: text
    3. [voice_name] text
    """
    baked_voices = set()
    if os.path.exists(BAKED_VOICES_DIR):
        baked_voices = {f.replace(".npz", "") for f in os.listdir(BAKED_VOICES_DIR) if f.endswith(".npz")}

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    segments = []
    
    # Matches [role]: content or role: content or [role] content
    pattern = re.compile(r'^(?:\[([a-zA-Z0-9_\u4e00-\u9fa5]+)\]|([a-zA-Z0-9_\u4e00-\u9fa5]+)[:：])\s*(.*)$')
    
    current_voice = default_voice
    
    for line in lines:
        match = pattern.match(line)
        if match:
            v_name = match.group(1) or match.group(2)
            content = match.group(3)
            if v_name in baked_voices:
                current_voice = v_name
                if content and content.strip():
                    segments.append((current_voice, content.strip()))
                continue
        segments.append((current_voice, line))
        
    # Merge consecutive lines with same voice
    merged = []
    for v, t in segments:
        if merged and merged[-1][0] == v:
            merged[-1] = (v, merged[-1][1] + " " + t)
        else:
            merged.append((v, t))
            
    return merged if merged else [(default_voice, text)]

@app.get("/health")
def health():
    baked_files = [f.replace(".npz", "") for f in os.listdir(BAKED_VOICES_DIR) if f.endswith(".npz")] if os.path.exists(BAKED_VOICES_DIR) else []
    return {
        "status": "ok",
        "service": "Luna-TTS Local Engine",
        "port": 8890,
        "engine_type": "Qwen3-TTS 1.7B Base Q4_K_M Zero-Shot Voice Clone",
        "baked_voices": baked_files,
        "default_voice": "sample2",
        "multi_role_dialogue_supported": True
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "luna-tts", "object": "model", "owned_by": "vui-labs/qwen3-tts"},
            {"id": "qwen3-tts-1.7b-base", "object": "model", "owned_by": "vui-labs/qwen3-tts"}
        ]
    }

@app.get("/v1/voices")
def list_voices():
    voices = []
    if os.path.exists(BAKED_VOICES_DIR):
        for fname in sorted(os.listdir(BAKED_VOICES_DIR)):
            if fname.endswith(".npz"):
                vname = fname[:-4]
                npz_path = os.path.join(BAKED_VOICES_DIR, fname)
                size_kb = round(os.path.getsize(npz_path) / 1024, 1)
                
                ref_file = None
                for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
                    cand = os.path.join(REF_AUDIO_DIR, f"{vname}{ext}")
                    if os.path.exists(cand):
                        ref_file = f"/v1/audio/ref/{vname}"
                        break
                        
                voices.append({
                    "name": vname,
                    "npz_size_kb": size_kb,
                    "has_ref_audio": ref_file is not None,
                    "ref_audio_url": ref_file
                })
    return {"voices": voices}

@app.get("/v1/audio/ref/{voice_name}")
def get_ref_audio(voice_name: str):
    for ext in [".mp3", ".wav", ".m4a", ".flac", ".ogg"]:
        cand = os.path.join(REF_AUDIO_DIR, f"{voice_name}{ext}")
        if os.path.exists(cand):
            mime = "audio/ogg" if ext == ".ogg" else ("audio/mpeg" if ext == ".mp3" else f"audio/{ext[1:]}")
            return FileResponse(cand, media_type=mime)
    raise HTTPException(status_code=404, detail="Reference audio not found")

async def _synthesize_single_segment(text: str, voice_name: str) -> str:
    """Helper to synthesize one audio segment to temporary WAV file."""
    temp_wav = f"/tmp/luna_seg_{time.time_ns()}_{os.urandom(4).hex()}.wav"
    proc = await asyncio.to_thread(
        subprocess.run,
        [PYTHON_BIN, WORKER_SCRIPT, text, temp_wav, voice_name],
        capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0 or not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
        if os.path.exists(temp_wav): os.remove(temp_wav)
        raise RuntimeError(f"Segment synthesis failed for voice '{voice_name}': {proc.stderr}")
    return temp_wav

@app.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest):
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    default_voice = req.voice or "sample2"
    fmt = (req.response_format or "ogg").lower()
    ext = ".ogg" if fmt in {"ogg", "opus"} else ".mp3"
    media_type = "audio/ogg" if ext == ".ogg" else "audio/mpeg"

    segments = parse_dialogue_script(req.input, default_voice)
    logger.info(f"Speech request ({len(segments)} segment(s)): '{req.input[:40]}...' (default_voice={default_voice}, format={fmt})")
    start_t = time.perf_counter()

    temp_audio_out = f"/tmp/luna_out_{time.time_ns()}{ext}"
    created_files = []

    try:
        if len(segments) == 1:
            # Single voice fast-path
            v_name, txt = segments[0]
            proc = await asyncio.to_thread(
                subprocess.run,
                [PYTHON_BIN, WORKER_SCRIPT, txt, temp_audio_out, v_name],
                capture_output=True, text=True, timeout=120
            )
            if proc.returncode != 0 or not os.path.exists(temp_audio_out):
                raise RuntimeError(f"Worker process failed: {proc.stderr}")
        else:
            # Multi-role dialogue synthesis & audio stitching
            seg_wavs = []
            for v_name, txt in segments:
                wav_p = await _synthesize_single_segment(txt, v_name)
                seg_wavs.append(wav_p)
                created_files.append(wav_p)

            # Generate 0.35s turn silence WAV
            silence_wav = f"/tmp/silence_{time.time_ns()}.wav"
            created_files.append(silence_wav)
            await asyncio.to_thread(
                subprocess.run,
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono", "-t", "0.35", silence_wav],
                capture_output=True, check=True
            )

            # Create ffmpeg concat list file
            concat_list = f"/tmp/concat_{time.time_ns()}.txt"
            created_files.append(concat_list)
            with open(concat_list, "w") as f:
                for idx, w_file in enumerate(seg_wavs):
                    f.write(f"file '{w_file}'\n")
                    if idx < len(seg_wavs) - 1:
                        f.write(f"file '{silence_wav}'\n")

            # Stitch all segments with ffmpeg
            if ext == ".ogg":
                ffmpeg_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c:a", "libopus", "-b:a", "24k", "-ar", "48000", temp_audio_out]
            else:
                ffmpeg_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-f", "mp3", "-ac", "1", "-ar", "24000", "-b:a", "64k", temp_audio_out]

            await asyncio.to_thread(
                subprocess.run,
                ffmpeg_cmd, capture_output=True, text=True, check=True
            )

        with open(temp_audio_out, "rb") as f:
            audio_bytes = f.read()

        if os.path.exists(temp_audio_out):
            os.remove(temp_audio_out)

        for tmp_f in created_files:
            if os.path.exists(tmp_f):
                os.remove(tmp_f)

        elapsed = time.perf_counter() - start_t
        logger.info(f"Speech synthesis complete ({len(segments)} segments) in {elapsed:.3f}s ({len(audio_bytes)} bytes)")
        return Response(content=audio_bytes, media_type=media_type)
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        if os.path.exists(temp_audio_out): os.remove(temp_audio_out)
        for tmp_f in created_files:
            if os.path.exists(tmp_f): os.remove(tmp_f)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/clone")
async def clone_voice(
    file: UploadFile = File(...),
    voice_name: str = Form(...),
    ref_text: Optional[str] = Form(None)
):
    voice_name = voice_name.strip().replace(" ", "_")
    if not voice_name:
        raise HTTPException(status_code=400, detail="Voice name cannot be empty")

    ext = os.path.splitext(file.filename)[1].lower() or ".mp3"
    tmp_upload = f"/tmp/upload_{time.time_ns()}{ext}"

    logger.info(f"Baking new voice clone: '{voice_name}' from uploaded file '{file.filename}'")

    try:
        with open(tmp_upload, "wb") as f:
            content = await file.read()
            f.write(content)

        dest_ref = os.path.join(REF_AUDIO_DIR, f"{voice_name}{ext}")
        shutil.copyfile(tmp_upload, dest_ref)

        cmd = [PYTHON_BIN, BAKER_SCRIPT, tmp_upload, voice_name]
        if ref_text and ref_text.strip():
            cmd.append(ref_text.strip())

        # Non-blocking async execution
        proc = await asyncio.to_thread(
            subprocess.run,
            cmd, capture_output=True, text=True, timeout=180
        )

        if proc.returncode != 0:
            logger.error(f"Voice baker process error: {proc.stderr}")
            raise RuntimeError(f"Voice baker failed: {proc.stderr}")

        npz_file = os.path.join(BAKED_VOICES_DIR, f"{voice_name}.npz")
        if not os.path.exists(npz_file):
            raise RuntimeError("Baker completed but NPZ file was not found")

        if os.path.exists(tmp_upload):
            os.remove(tmp_upload)

        return {
            "status": "success",
            "voice_name": voice_name,
            "npz_size_kb": round(os.path.getsize(npz_file) / 1024, 1),
            "ref_audio_saved": dest_ref
        }
    except Exception as e:
        logger.error(f"Voice cloning error: {e}")
        if os.path.exists(tmp_upload):
            os.remove(tmp_upload)
        raise HTTPException(status_code=500, detail=str(e))

# 9002 PureVision Cyberpunk Glassmorphic Design System
HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PureVision Luna-TTS · 极速多角色双音色零样本语音控制台</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #050710;
            --bg-surface: #0b0e20;
            --bg-elev-1: #121630;
            --bg-elev-2: #191e42;
            --bg-hover: #212854;

            --pink-main: #ff7eb6;
            --pink-soft: rgba(255, 126, 182, 0.14);
            --pink-glow: rgba(255, 126, 182, 0.28);

            --blue-main: #70d6ff;
            --blue-soft: rgba(112, 214, 255, 0.14);
            --blue-glow: rgba(112, 214, 255, 0.28);

            --accent-gradient: linear-gradient(135deg, #ff7eb6 0%, #70d6ff 100%);
            --line: rgba(112, 214, 255, 0.10);
            --line-pink: rgba(255, 126, 182, 0.20);
            --line-strong: rgba(112, 214, 255, 0.22);

            --text-1: #ffffff;
            --text-2: #d4dbf5;
            --text-3: #8e97c6;
            --text-4: #575e8a;

            --ok: #5cc47b;
            --ok-soft: rgba(92, 196, 123, 0.12);
            --warn: #ffb703;
            --err: #ff4d6d;

            --r-sm: 6px;
            --r-md: 10px;
            --r-lg: 14px;
            --r-xl: 20px;
            --font: "Manrope", -apple-system, sans-serif;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: var(--font);
            background: var(--bg-base);
            color: var(--text-1);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(255, 126, 182, 0.05) 0%, transparent 45%),
                radial-gradient(circle at 90% 80%, rgba(112, 214, 255, 0.05) 0%, transparent 45%);
            background-attachment: fixed;
        }

        .app-card {
            width: 100%;
            max-width: 920px;
            background: var(--bg-surface);
            border: 1px solid var(--line-strong);
            border-radius: var(--r-xl);
            padding: 2.2rem;
            box-shadow: 0 25px 60px rgba(0, 0, 0, 0.6), 0 0 30px rgba(112, 214, 255, 0.05);
            backdrop-filter: blur(20px);
        }

        .brand-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1.2rem;
            border-bottom: 1px solid var(--line);
        }

        .brand-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-logo-svg {
            display: flex;
            align-items: center;
            justify-content: center;
            background: var(--bg-elev-1);
            border: 1px solid var(--line-pink);
            border-radius: var(--r-md);
            padding: 4px;
            box-shadow: 0 0 15px var(--pink-soft);
        }

        .brand-text h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
        }

        .brand-text p {
            color: var(--text-3);
            font-size: 0.85rem;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--bg-elev-1);
            border: 1px solid var(--line);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            color: var(--text-2);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--ok);
            box-shadow: 0 0 8px var(--ok);
        }

        .nav-tabs {
            display: flex;
            gap: 8px;
            background: var(--bg-elev-1);
            padding: 6px;
            border-radius: var(--r-lg);
            margin-bottom: 2rem;
            border: 1px solid var(--line);
        }

        .tab-btn {
            flex: 1;
            background: none;
            border: none;
            color: var(--text-3);
            font-size: 0.92rem;
            font-weight: 600;
            padding: 10px 16px;
            border-radius: var(--r-md);
            cursor: pointer;
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .tab-btn:hover {
            color: var(--text-1);
            background: rgba(255, 255, 255, 0.04);
        }

        .tab-btn.active {
            color: var(--text-1);
            background: var(--bg-elev-2);
            border: 1px solid var(--line-pink);
            box-shadow: 0 4px 15px var(--pink-soft);
        }

        .tab-panel { display: none; }
        .tab-panel.active { display: block; animation: fadeUp 250ms ease-out; }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .form-group { margin-bottom: 1.4rem; }

        .form-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-2);
            margin-bottom: 0.5rem;
            letter-spacing: 0.01em;
        }

        .preset-btn {
            background: var(--bg-elev-2);
            border: 1px solid var(--line-pink);
            color: var(--pink-main);
            padding: 3px 10px;
            border-radius: var(--r-sm);
            font-size: 0.76rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .preset-btn:hover {
            background: var(--pink-soft);
        }

        .input-field, select, textarea {
            width: 100%;
            background: var(--bg-elev-1);
            border: 1px solid var(--line-strong);
            border-radius: var(--r-md);
            color: var(--text-1);
            padding: 12px 16px;
            font-size: 0.92rem;
            outline: none;
            transition: all 0.2s;
        }

        .input-field:focus, select:focus, textarea:focus {
            border-color: var(--pink-main);
            box-shadow: 0 0 12px var(--pink-glow);
        }

        textarea { height: 130px; resize: vertical; line-height: 1.6; font-family: inherit; }

        .dropzone {
            border: 2px dashed var(--line-strong);
            border-radius: var(--r-md);
            padding: 2rem;
            text-align: center;
            background: var(--bg-elev-1);
            cursor: pointer;
            transition: all 0.2s;
        }

        .dropzone:hover, .dropzone.drag-active {
            border-color: var(--pink-main);
            background: var(--pink-soft);
            box-shadow: 0 0 20px var(--pink-glow);
        }

        .cyber-btn {
            width: 100%;
            background: var(--accent-gradient);
            color: #000;
            border: none;
            border-radius: var(--r-md);
            padding: 14px;
            font-size: 0.98rem;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 4px 20px var(--pink-glow);
            transition: transform 0.15s, opacity 0.2s;
        }

        .cyber-btn:hover { opacity: 0.94; transform: translateY(-1px); }
        .cyber-btn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

        .result-box {
            margin-top: 1.8rem;
            padding: 1.2rem;
            background: var(--bg-elev-1);
            border-radius: var(--r-md);
            border: 1px solid var(--line-pink);
            display: none;
        }

        audio { width: 100%; margin-top: 0.6rem; border-radius: 8px; }

        .voice-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .voice-card {
            background: var(--bg-elev-1);
            border: 1px solid var(--line);
            border-radius: var(--r-md);
            padding: 1.1rem;
            transition: border-color 0.2s;
        }

        .voice-card:hover { border-color: var(--line-pink); }
        .voice-card h4 { color: var(--pink-main); font-size: 1rem; margin-bottom: 4px; }
        .voice-card span { font-size: 0.78rem; color: var(--text-3); }

        .spinner {
            display: inline-block;
            width: 16px; height: 16px;
            border: 2px solid rgba(0,0,0,0.3);
            border-radius: 50%;
            border-top-color: #000;
            animation: spin 0.7s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="app-card">
        <div class="brand-header">
            <div class="brand-title">
                <div class="brand-logo-svg">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500" width="40" height="40">
                      <defs>
                        <linearGradient id="cyberGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                          <stop offset="0%" stop-color="#ff7eb6"/>
                          <stop offset="100%" stop-color="#70d6ff"/>
                        </linearGradient>
                      </defs>
                      <path d="M 75.0 35.0 L 73.3 36.7 L 71.7 36.7 L 65.0 36.7 L 63.3 38.3 L 61.7 38.3 L 56.7 38.3 L 55.0 40.0 L 53.3 40.0 L 51.7 41.7 L 50.0 41.7 L 48.3 41.7 L 45.0 45.0 L 43.3 45.0 L 41.7 46.7 L 40.0 46.7 L 23.3 63.3 L 23.3 65.0 L 20.0 68.3 L 20.0 70.0 L 18.3 71.7 L 18.3 73.3 L 16.7 75.0 L 16.7 78.3 L 15.0 80.0 L 15.0 81.7 L 15.0 83.3 L 13.3 85.0 L 13.3 93.3 L 11.7 95.0 L 11.7 96.7 L 11.7 105.0 L 13.3 106.7 L 13.3 115.0 L 15.0 116.7 L 15.0 118.3 L 16.7 120.0 L 16.7 125.0 L 18.3 126.7 L 18.3 128.3 L 20.0 130.0 L 21.7 130.0 L 21.7 120.0 L 23.3 118.3 L 23.3 116.7 L 25.0 115.0 L 26.7 113.3 L 26.7 111.7 L 30.0 108.3 L 31.7 108.3 L 33.3 106.7 L 35.0 105.0 L 38.3 105.0 L 40.0 103.3 L 281.7 103.3 L 283.3 105.0 L 288.3 110.0 L 288.3 115.0 L 286.7 116.7 L 286.7 118.3 L 285.0 120.0 L 285.0 121.7 L 283.3 123.3 L 283.3 125.0 L 280.0 128.3 L 280.0 130.0 L 278.3 131.7 L 278.3 133.3 L 273.3 138.3 L 271.7 138.3 L 270.0 140.0 L 106.7 140.0 L 105.0 141.7 L 66.7 141.7 L 65.0 140.0 L 63.3 141.7 L 58.3 141.7 L 56.7 143.3 L 53.3 143.3 L 46.7 150.0 L 46.7 151.7 L 45.0 153.3 L 45.0 156.7 L 43.3 158.3 L 43.3 168.3 L 45.0 170.0 L 45.0 175.0 L 48.3 178.3 L 48.3 180.0 L 51.7 183.3 L 51.7 185.0 L 53.3 186.7 L 53.3 188.3 L 55.0 190.0 L 55.0 191.7 L 56.7 193.3 L 58.3 195.0 L 58.3 196.7 L 61.7 200.0 L 61.7 203.3 L 63.3 205.0 L 65.0 206.7 L 65.0 208.3 L 66.7 210.0 L 66.7 211.7 L 68.3 213.3 L 70.0 215.0 L 70.0 216.7 L 71.7 218.3 L 71.7 220.0 L 73.3 221.7 L 73.3 223.3 L 76.7 226.7 L 76.7 228.3 L 78.3 230.0 L 78.3 231.7 L 80.0 233.3 L 80.0 235.0 L 83.3 238.3 L 83.3 240.0 L 85.0 241.7 L 85.0 243.3 L 88.3 246.7 L 88.3 248.3 L 90.0 250.0 L 90.0 251.7 L 93.3 255.0 L 93.3 256.7 L 95.0 258.3 L 95.0 260.0 L 96.7 261.7 L 96.7 263.3 L 98.3 265.0 L 98.3 266.7 L 101.7 270.0 L 101.7 271.7 L 103.3 273.3 L 103.3 275.0 L 106.7 278.3 L 106.7 280.0 L 108.3 281.7 L 108.3 283.3 L 110.0 285.0 L 110.0 286.7 L 113.3 290.0 L 113.3 291.7 L 115.0 293.3 L 115.0 295.0 L 116.7 296.7 L 116.7 298.3 L 118.3 300.0 L 120.0 301.7 L 120.0 303.3 L 121.7 305.0 L 121.7 306.7 L 123.3 308.3 L 123.3 310.0 L 126.7 313.3 L 126.7 315.0 L 128.3 316.7 L 128.3 318.3 L 131.7 321.7 L 131.7 323.3 L 133.3 325.0 L 133.3 326.7 L 135.0 328.3 L 135.0 330.0 L 138.3 333.3 L 138.3 335.0 L 140.0 336.7 L 140.0 338.3 L 141.7 340.0 L 141.7 341.7 L 145.0 345.0 L 145.0 346.7 L 146.7 348.3 L 148.3 350.0 L 148.3 353.3 L 151.7 356.7 L 151.7 358.3 L 153.3 360.0 L 153.3 361.7 L 156.7 365.0 L 156.7 366.7 L 158.3 368.3 L 158.3 370.0 L 160.0 371.7 L 160.0 373.3 L 163.3 376.7 L 163.3 378.3 L 165.0 380.0 L 165.0 381.7 L 166.7 383.3 L 166.7 385.0 L 170.0 388.3 L 170.0 390.0 L 171.7 391.7 L 171.7 393.3 L 175.0 396.7 L 175.0 398.3 L 176.7 400.0 L 176.7 401.7 L 178.3 403.3 L 178.3 405.0 L 181.7 408.3 L 181.7 410.0 L 183.3 411.7 L 183.3 413.3 L 185.0 415.0 L 185.0 416.7 L 186.7 418.3 L 188.3 420.0 L 188.3 421.7 L 190.0 423.3 L 190.0 425.0 L 191.7 426.7 L 193.3 428.3 L 193.3 430.0 L 195.0 431.7 L 195.0 433.3 L 196.7 435.0 L 198.3 436.7 L 198.3 438.3 L 210.0 450.0 L 211.7 450.0 L 213.3 451.7 L 215.0 453.3 L 216.7 453.3 L 218.3 455.0 L 220.0 455.0 L 221.7 456.7 L 223.3 456.7 L 225.0 458.3 L 226.7 458.3 L 228.3 460.0 L 230.0 460.0 L 235.0 460.0 L 236.7 461.7 L 243.3 461.7 L 245.0 463.3 L 246.7 463.3 L 253.3 463.3 L 255.0 461.7 L 261.7 461.7 L 263.3 460.0 L 265.0 460.0 L 270.0 460.0 L 271.7 458.3 L 273.3 458.3 L 275.0 456.7 L 276.7 456.7 L 278.3 455.0 L 280.0 455.0 L 281.7 453.3 L 283.3 453.3 L 285.0 451.7 L 286.7 451.7 L 300.0 438.3 L 300.0 436.7 L 305.0 431.7 L 305.0 430.0 L 303.3 430.0 L 301.7 431.7 L 300.0 431.7 L 298.3 433.3 L 296.7 433.3 L 295.0 435.0 L 293.3 435.0 L 291.7 436.7 L 285.0 436.7 L 283.3 435.0 L 280.0 435.0 L 278.3 433.3 L 276.7 433.3 L 268.3 425.0 L 268.3 423.3 L 266.7 421.7 L 266.7 420.0 L 263.3 416.7 L 263.3 415.0 L 261.7 413.3 L 260.0 411.7 L 260.0 410.0 L 258.3 408.3 L 258.3 406.7 L 256.7 405.0 L 256.7 403.3 L 253.3 400.0 L 253.3 398.3 L 251.7 396.7 L 251.7 395.0 L 250.0 393.3 L 250.0 391.7 L 246.7 388.3 L 246.7 386.7 L 245.0 385.0 L 245.0 383.3 L 243.3 381.7 L 243.3 380.0 L 240.0 376.7 L 240.0 375.0 L 238.3 373.3 L 238.3 371.7 L 235.0 368.3 L 235.0 366.7 L 233.3 365.0 L 233.3 363.3 L 231.7 361.7 L 231.7 360.0 L 228.3 356.7 L 228.3 355.0 L 226.7 353.3 L 226.7 351.7 L 225.0 350.0 L 225.0 348.3 L 223.3 346.7 L 221.7 345.0 L 221.7 343.3 L 220.0 341.7 L 220.0 340.0 L 218.3 338.3 L 216.7 336.7 L 216.7 335.0 L 215.0 333.3 L 215.0 331.7 L 213.3 330.0 L 213.3 328.3 L 210.0 325.0 L 210.0 323.3 L 208.3 321.7 L 208.3 320.0 L 206.7 318.3 L 206.7 316.7 L 203.3 313.3 L 203.3 311.7 L 201.7 310.0 L 201.7 308.3 L 198.3 305.0 L 198.3 303.3 L 196.7 301.7 L 196.7 300.0 L 195.0 298.3 L 195.0 296.7 L 191.7 293.3 L 191.7 291.7 L 190.0 290.0 L 190.0 288.3 L 188.3 286.7 L 188.3 285.0 L 185.0 281.7 L 185.0 280.0 L 183.3 278.3 L 183.3 276.7 L 181.7 275.0 L 181.7 273.3 L 180.0 271.7 L 178.3 270.0 L 178.3 268.3 L 176.7 266.7 L 176.7 265.0 L 175.0 263.3 L 173.3 261.7 L 173.3 260.0 L 171.7 258.3 L 171.7 256.7 L 170.0 255.0 L 170.0 253.3 L 166.7 250.0 L 166.7 248.3 L 165.0 246.7 L 165.0 245.0 L 163.3 243.3 L 163.3 241.7 L 160.0 238.3 L 160.0 236.7 L 158.3 235.0 L 156.7 233.3 L 156.7 230.0 L 153.3 226.7 L 153.3 218.3 L 155.0 216.7 L 155.0 215.0 L 161.7 208.3 L 296.7 208.3 L 298.3 206.7 L 300.0 206.7 L 303.3 206.7 L 305.0 205.0 L 306.7 205.0 L 308.3 203.3 L 310.0 203.3 L 311.7 201.7 L 313.3 201.7 L 323.3 191.7 L 323.3 190.0 L 325.0 188.3 L 325.0 186.7 L 326.7 185.0 L 326.7 183.3 L 330.0 180.0 L 330.0 178.3 L 331.7 176.7 L 331.7 175.0 L 333.3 173.3 L 333.3 171.7 L 335.0 170.0 L 336.7 168.3 L 336.7 166.7 L 338.3 165.0 L 338.3 163.3 L 340.0 161.7 L 340.0 160.0 L 343.3 156.7 L 343.3 155.0 L 345.0 153.3 L 345.0 151.7 L 348.3 148.3 L 348.3 146.7 L 350.0 145.0 L 350.0 143.3 L 351.7 141.7 L 351.7 140.0 L 355.0 136.7 L 355.0 135.0 L 356.7 133.3 L 356.7 131.7 L 358.3 130.0 L 358.3 128.3 L 361.7 125.0 L 361.7 123.3 L 363.3 121.7 L 363.3 120.0 L 366.7 116.7 L 366.7 115.0 L 368.3 113.3 L 368.3 111.7 L 371.7 108.3 L 371.7 106.7 L 373.3 105.0 L 373.3 103.3 L 375.0 101.7 L 375.0 100.0 L 376.7 98.3 L 376.7 96.7 L 380.0 93.3 L 380.0 91.7 L 381.7 90.0 L 381.7 88.3 L 385.0 85.0 L 385.0 83.3 L 386.7 81.7 L 386.7 80.0 L 388.3 78.3 L 388.3 76.7 L 391.7 73.3 L 391.7 71.7 L 393.3 70.0 L 393.3 68.3 L 395.0 66.7 L 395.0 53.3 L 393.3 51.7 L 393.3 50.0 L 391.7 48.3 L 391.7 46.7 L 385.0 40.0 L 383.3 40.0 L 381.7 38.3 L 378.3 38.3 L 376.7 36.7 L 96.7 36.7 L 95.0 35.0 L 76.7 35.0 Z" fill="url(#cyberGradient)"/>
                      <path d="M 420.0 36.7 L 421.7 36.7 L 423.3 38.3 L 425.0 38.3 L 426.7 40.0 L 428.3 40.0 L 433.3 45.0 L 433.3 46.7 L 436.7 50.0 L 436.7 53.3 L 438.3 55.0 L 438.3 63.3 L 436.7 65.0 L 436.7 68.3 L 435.0 70.0 L 435.0 71.7 L 431.7 75.0 L 431.7 76.7 L 430.0 78.3 L 430.0 80.0 L 428.3 81.7 L 428.3 83.3 L 425.0 86.7 L 425.0 88.3 L 423.3 90.0 L 423.3 91.7 L 420.0 95.0 L 420.0 96.7 L 418.3 98.3 L 418.3 100.0 L 416.7 101.7 L 416.7 103.3 L 413.3 106.7 L 413.3 108.3 L 411.7 110.0 L 411.7 111.7 L 410.0 113.3 L 410.0 115.0 L 406.7 118.3 L 406.7 120.0 L 405.0 121.7 L 403.3 123.3 L 403.3 126.7 L 400.0 130.0 L 400.0 131.7 L 398.3 133.3 L 396.7 135.0 L 396.7 136.7 L 395.0 138.3 L 395.0 140.0 L 393.3 141.7 L 393.3 143.3 L 390.0 146.7 L 390.0 148.3 L 388.3 150.0 L 388.3 151.7 L 386.7 153.3 L 386.7 155.0 L 385.0 156.7 L 385.0 158.3 L 383.3 160.0 L 381.7 161.7 L 381.7 163.3 L 378.3 166.7 L 378.3 168.3 L 376.7 170.0 L 376.7 171.7 L 375.0 173.3 L 375.0 175.0 L 371.7 178.3 L 371.7 180.0 L 370.0 181.7 L 370.0 183.3 L 368.3 185.0 L 368.3 186.7 L 366.7 188.3 L 365.0 190.0 L 365.0 191.7 L 363.3 193.3 L 363.3 195.0 L 361.7 196.7 L 360.0 198.3 L 360.0 200.0 L 358.3 201.7 L 358.3 203.3 L 356.7 205.0 L 356.7 206.7 L 353.3 210.0 L 353.3 211.7 L 351.7 213.3 L 351.7 215.0 L 350.0 216.7 L 350.0 218.3 L 348.3 220.0 L 348.3 221.7 L 346.7 223.3 L 345.0 225.0 L 345.0 226.7 L 341.7 230.0 L 341.7 233.3 L 340.0 235.0 L 338.3 236.7 L 338.3 238.3 L 335.0 241.7 L 335.0 243.3 L 333.3 245.0 L 333.3 246.7 L 331.7 248.3 L 331.7 250.0 L 328.3 253.3 L 328.3 255.0 L 326.7 256.7 L 326.7 258.3 L 325.0 260.0 L 325.0 261.7 L 323.3 263.3 L 323.3 265.0 L 316.7 271.7 L 315.0 271.7 L 313.3 273.3 L 311.7 273.3 L 306.7 273.3 L 305.0 271.7 L 301.7 271.7 L 296.7 266.7 L 296.7 265.0 L 295.0 263.3 L 295.0 261.7 L 293.3 260.0 L 293.3 258.3 L 291.7 256.7 L 290.0 255.0 L 290.0 253.3 L 285.0 248.3 L 283.3 248.3 L 281.7 246.7 L 280.0 246.7 L 278.3 245.0 L 225.0 245.0 L 216.7 253.3 L 216.7 255.0 L 216.7 263.3 L 218.3 265.0 L 218.3 266.7 L 221.7 270.0 L 221.7 271.7 L 223.3 273.3 L 223.3 275.0 L 226.7 278.3 L 226.7 280.0 L 228.3 281.7 L 228.3 283.3 L 230.0 285.0 L 230.0 286.7 L 233.3 290.0 L 233.3 291.7 L 235.0 293.3 L 235.0 295.0 L 236.7 296.7 L 236.7 298.3 L 238.3 300.0 L 240.0 301.7 L 240.0 303.3 L 241.7 305.0 L 241.7 306.7 L 243.3 308.3 L 245.0 310.0 L 245.0 313.3 L 248.3 316.7 L 248.3 318.3 L 250.0 320.0 L 251.7 321.7 L 251.7 323.3 L 253.3 325.0 L 253.3 326.7 L 255.0 328.3 L 255.0 330.0 L 258.3 333.3 L 258.3 335.0 L 260.0 336.7 L 260.0 338.3 L 261.7 340.0 L 261.7 341.7 L 265.0 345.0 L 265.0 346.7 L 266.7 348.3 L 266.7 350.0 L 270.0 353.3 L 270.0 355.0 L 271.7 356.7 L 271.7 358.3 L 273.3 360.0 L 273.3 361.7 L 275.0 363.3 L 275.0 365.0 L 276.7 366.7 L 278.3 368.3 L 278.3 370.0 L 280.0 371.7 L 280.0 373.3 L 281.7 375.0 L 283.3 376.7 L 283.3 378.3 L 285.0 380.0 L 285.0 381.7 L 286.7 383.3 L 288.3 385.0 L 288.3 386.7 L 290.0 388.3 L 290.0 390.0 L 291.7 391.7 L 295.0 395.0 L 296.7 395.0 L 300.0 398.3 L 303.3 398.3 L 305.0 400.0 L 313.3 400.0 L 315.0 398.3 L 318.3 398.3 L 321.7 395.0 L 323.3 395.0 L 330.0 388.3 L 330.0 386.7 L 331.7 385.0 L 331.7 383.3 L 333.3 381.7 L 333.3 380.0 L 336.7 376.7 L 336.7 375.0 L 338.3 373.3 L 338.3 371.7 L 340.0 370.0 L 340.0 368.3 L 343.3 365.0 L 343.3 363.3 L 345.0 361.7 L 345.0 360.0 L 346.7 358.3 L 346.7 356.7 L 350.0 353.3 L 350.0 351.7 L 353.3 348.3 L 353.3 346.7 L 355.0 345.0 L 355.0 343.3 L 356.7 341.7 L 356.7 340.0 L 358.3 338.3 L 358.3 336.7 L 361.7 333.3 L 361.7 331.7 L 363.3 330.0 L 363.3 328.3 L 365.0 326.7 L 365.0 325.0 L 366.7 323.3 L 368.3 321.7 L 368.3 320.0 L 370.0 318.3 L 370.0 316.7 L 371.7 315.0 L 373.3 313.3 L 373.3 311.7 L 375.0 310.0 L 375.0 308.3 L 376.7 306.7 L 376.7 305.0 L 380.0 301.7 L 380.0 300.0 L 381.7 298.3 L 381.7 296.7 L 383.3 295.0 L 383.3 293.3 L 386.7 290.0 L 386.7 288.3 L 388.3 286.7 L 388.3 285.0 L 390.0 283.3 L 390.0 281.7 L 391.7 280.0 L 391.7 278.3 L 395.0 275.0 L 395.0 273.3 L 398.3 270.0 L 398.3 268.3 L 400.0 266.7 L 400.0 265.0 L 401.7 263.3 L 401.7 261.7 L 405.0 258.3 L 405.0 256.7 L 406.7 255.0 L 406.7 253.3 L 408.3 251.7 L 408.3 250.0 L 410.0 248.3 L 411.7 246.7 L 411.7 245.0 L 413.3 243.3 L 413.3 241.7 L 415.0 240.0 L 416.7 238.3 L 416.7 236.7 L 418.3 235.0 L 418.3 233.3 L 420.0 231.7 L 420.0 230.0 L 423.3 226.7 L 423.3 225.0 L 425.0 223.3 L 425.0 221.7 L 426.7 220.0 L 426.7 218.3 L 428.3 216.7 L 428.3 215.0 L 430.0 213.3 L 431.7 211.7 L 431.7 210.0 L 435.0 206.7 L 435.0 205.0 L 436.7 203.3 L 436.7 201.7 L 438.3 200.0 L 438.3 198.3 L 441.7 195.0 L 441.7 193.3 L 443.3 191.7 L 443.3 190.0 L 445.0 188.3 L 445.0 186.7 L 446.7 185.0 L 448.3 183.3 L 448.3 181.7 L 451.7 178.3 L 451.7 175.0 L 453.3 173.3 L 455.0 171.7 L 455.0 170.0 L 456.7 168.3 L 456.7 166.7 L 458.3 165.0 L 460.0 163.3 L 460.0 161.7 L 461.7 160.0 L 461.7 158.3 L 463.3 156.7 L 463.3 155.0 L 466.7 151.7 L 466.7 150.0 L 468.3 148.3 L 468.3 146.7 L 470.0 145.0 L 470.0 143.3 L 473.3 140.0 L 473.3 138.3 L 475.0 136.7 L 476.7 135.0 L 476.7 133.3 L 478.3 131.7 L 478.3 130.0 L 480.0 128.3 L 480.0 126.7 L 481.7 125.0 L 481.7 121.7 L 483.3 120.0 L 483.3 118.3 L 483.3 116.7 L 485.0 115.0 L 485.0 106.7 L 486.7 105.0 L 486.7 103.3 L 486.7 95.0 L 485.0 93.3 L 485.0 85.0 L 483.3 83.3 L 483.3 81.7 L 483.3 80.0 L 481.7 78.3 L 481.7 75.0 L 480.0 73.3 L 480.0 71.7 L 478.3 70.0 L 478.3 68.3 L 475.0 65.0 L 475.0 63.3 L 473.3 61.7 L 458.3 46.7 L 456.7 46.7 L 455.0 45.0 L 453.3 45.0 L 451.7 43.3 L 450.0 43.3 L 448.3 41.7 L 446.7 41.7 L 445.0 40.0 L 443.3 40.0 L 441.7 40.0 L 440.0 38.3 L 431.7 38.3 L 430.0 36.7 L 428.3 36.7 L 421.7 36.7 Z" fill="url(#cyberGradient)"/>
                    </svg>
                </div>
                <div class="brand-text">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom: 2px;">
                        <span style="font-size:0.75rem; font-weight:700; color:#C9A96E; letter-spacing:0.15em; text-transform:uppercase;">PUREVISION</span>
                        <span style="font-size:0.7rem; color:var(--text-4); font-weight:300;">| www.pvsdesign.com</span>
                    </div>
                    <h1>Luna-TTS 极速双音色对话控制台</h1>
                    <p>Qwen3-TTS 1.7B Q4_K_M • 支持单人/多角色对话广播剧 • 0 MB 显存</p>
                </div>
            </div>
            <div class="status-pill">
                <div class="status-dot"></div>
                <span>Port 8890 Online</span>
            </div>
        </div>

        <div class="nav-tabs">
            <button class="tab-btn active" onclick="switchTab('tts')">💬 文本/多角色对话合成</button>
            <button class="tab-btn" onclick="switchTab('clone')">🔊 一键克隆声纹</button>
            <button class="tab-btn" onclick="switchTab('manage')">🗂️ 声纹库管理</button>
        </div>

        <!-- TAB 1: TTS -->
        <div id="tab-tts" class="tab-panel active">
            <div class="form-group">
                <label class="form-label">默认音色 (单个角色时使用)</label>
                <select id="tts-voice-select"></select>
            </div>
            <div class="form-group">
                <div class="form-label">
                    <span>输入文本 (支持多角色语法 `[角色名]: 文本`)</span>
                    <button class="preset-btn" onclick="fillDialoguePreset()">🎭 载入双音色对话范例</button>
                </div>
                <textarea id="tts-input-text" placeholder="单人模式直接输入文字，或多角色模式格式：&#10;[sample2]: 主人，今天天气真好。&#10;[sample]: 是啊，我们去海边逛逛吧！">您好，我是 Fina。我已经为您配置好了支持双音色与多角色广播剧对话的 PureVision 极速控制台。</textarea>
            </div>
            <button class="cyber-btn" id="tts-submit-btn" onclick="generateSpeech()">
                <span id="tts-btn-text">🚀 开始极速合成 (自动识别多角色)</span>
            </button>

            <div class="result-box" id="tts-player-box">
                <label class="form-label" style="margin-bottom:0.4rem;">合成结果 (.ogg Opus 原生语音格式)</label>
                <audio id="tts-audio-player" controls></audio>
            </div>
        </div>

        <!-- TAB 2: CLONE -->
        <div id="tab-clone" class="tab-panel">
            <div class="form-group">
                <label class="form-label">音色标识名称 (如 fina_v2, my_voice)</label>
                <input type="text" class="input-field" id="clone-voice-name" placeholder="仅限英文、数字和下划线">
            </div>
            <div class="form-group">
                <label class="form-label">上传参考人声音频 (MP3 / WAV / M4A / FLAC)</label>
                <div class="dropzone" id="dropzone-box"
                     onclick="document.getElementById('clone-file-input').click()"
                     ondragover="handleDragOver(event)"
                     ondragleave="handleDragLeave(event)"
                     ondrop="handleDrop(event)">
                    <span id="dropzone-label">📁 点击或拖入 5~15 秒清晰人声文件</span>
                    <input type="file" id="clone-file-input" accept="audio/*" style="display:none;" onchange="handleFileSelect(this)">
                </div>
            </div>
            <div class="form-group">
                <label class="form-label">参考文本 (可选，留空由 Whisper ASR 自动识别)</label>
                <input type="text" class="input-field" id="clone-ref-text" placeholder="留空自动识别，或者手动输入音频对应的内容">
            </div>
            <button class="cyber-btn" id="clone-submit-btn" onclick="cloneVoice()">
                <span id="clone-btn-text">⚡ 开始提取并固化声纹</span>
            </button>
        </div>

        <!-- TAB 3: MANAGE -->
        <div id="tab-manage" class="tab-panel">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 1rem;">
                <h3 style="font-size:1.1rem; color:var(--text-1);">已固化的声纹 Profile</h3>
                <button class="tab-btn" onclick="loadVoices()" style="max-width:110px;">🔄 刷新列表</button>
            </div>
            <div class="voice-grid" id="voice-card-container"></div>
        </div>
    </div>

    <script>
        let availableVoices = [];

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
            if (tabId === 'manage') loadVoices();
        }

        function fillDialoguePreset() {
            const v1 = availableVoices.length > 0 ? availableVoices[0].name : 'sample2';
            const v2 = availableVoices.length > 1 ? availableVoices[1].name : (v1 === 'sample2' ? 'sample' : 'sample2');
            
            const presetText = `[${v1}]: 主人，今天天气真好，我们去海边逛逛吧？\n[${v2}]: 好啊，这主意听起来不错，我现在去准备一下！\n[${v1}]: 太棒了，那我等你哦！`;
            document.getElementById('tts-input-text').value = presetText;
        }

        function handleFileSelect(input) {
            if (input.files && input.files[0]) {
                document.getElementById('dropzone-label').innerText = '✅ 已选择: ' + input.files[0].name;
            }
        }

        function handleDragOver(e) {
            e.preventDefault();
            e.stopPropagation();
            document.getElementById('dropzone-box').classList.add('drag-active');
        }

        function handleDragLeave(e) {
            e.preventDefault();
            e.stopPropagation();
            document.getElementById('dropzone-box').classList.remove('drag-active');
        }

        function handleDrop(e) {
            e.preventDefault();
            e.stopPropagation();
            const box = document.getElementById('dropzone-box');
            box.classList.remove('drag-active');
            
            const dt = e.dataTransfer;
            const files = dt ? dt.files : null;
            if (files && files.length > 0) {
                const fileInput = document.getElementById('clone-file-input');
                fileInput.files = files;
                handleFileSelect(fileInput);
            }
        }

        async function loadVoices() {
            try {
                const res = await fetch('/v1/voices');
                const data = await res.json();
                availableVoices = data.voices || [];
                
                const select = document.getElementById('tts-voice-select');
                select.innerHTML = '';
                availableVoices.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v.name;
                    opt.innerText = v.name + (v.name === 'sample2' ? ' (Fina 专属音色)' : '');
                    select.appendChild(opt);
                });

                const container = document.getElementById('voice-card-container');
                container.innerHTML = '';
                availableVoices.forEach(v => {
                    const card = document.createElement('div');
                    card.className = 'voice-card';
                    card.innerHTML = `
                        <h4>🎙️ ${v.name}</h4>
                        <span>特征固化包: ${v.npz_size_kb} KB (.npz)</span>
                        ${v.has_ref_audio ? `<audio controls src="${v.ref_audio_url}" style="width:100%; height:32px; margin-top:0.4rem;"></audio>` : '<br><span>原音频文件: 未发现</span>'}
                    `;
                    container.appendChild(card);
                });
            } catch (err) {
                console.error('Failed to load voices:', err);
            }
        }

        async function generateSpeech() {
            const voice = document.getElementById('tts-voice-select').value;
            const input = document.getElementById('tts-input-text').value;
            if (!input.trim()) return alert('请输入需要合成的文本！');

            const btn = document.getElementById('tts-submit-btn');
            const btnText = document.getElementById('tts-btn-text');
            btn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span> 正在 CPU 高速合成多角色对话...';

            try {
                const res = await fetch('/v1/audio/speech', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ voice, input, response_format: 'ogg' })
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || '合成失败');
                }

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const player = document.getElementById('tts-audio-player');
                player.src = url;
                document.getElementById('tts-player-box').style.display = 'block';
                player.play();
            } catch (err) {
                alert('语音合成错误: ' + err.message);
            } finally {
                btn.disabled = false;
                btnText.innerText = '🚀 开始极速合成 (自动识别多角色)';
            }
        }

        async function cloneVoice() {
            const voiceName = document.getElementById('clone-voice-name').value.trim();
            const fileInput = document.getElementById('clone-file-input');
            const refText = document.getElementById('clone-ref-text').value.trim();

            if (!voiceName) return alert('请输入音色标识名称！');
            if (!fileInput.files || !fileInput.files[0]) return alert('请选择参考音频文件！');

            const btn = document.getElementById('clone-submit-btn');
            const btnText = document.getElementById('clone-btn-text');
            btn.disabled = true;
            btnText.innerHTML = '<span class="spinner"></span> 正在提取声纹特征与 Whisper 文本...';

            const formData = new FormData();
            formData.append('voice_name', voiceName);
            formData.append('file', fileInput.files[0]);
            if (refText) formData.append('ref_text', refText);

            try {
                const res = await fetch('/v1/audio/clone', {
                    method: 'POST',
                    body: formData
                });

                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || '克隆失败');

                alert(`🎉 音色 '${data.voice_name}' 提纯与固化成功！\n特征包: ${data.npz_size_kb} KB\n原声归档: ${data.ref_audio_saved}`);
                
                document.getElementById('clone-voice-name').value = '';
                document.getElementById('clone-file-input').value = '';
                document.getElementById('dropzone-label').innerText = '📁 点击或拖入 5~15 秒清晰人声文件';
                loadVoices();
                switchTab('tts');
            } catch (err) {
                alert('音色克隆错误: ' + err.message);
            } finally {
                btn.disabled = false;
                btnText.innerText = '⚡ 开始提取并固化声纹';
            }
        }

        window.onload = loadVoices;
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
def index_page():
    return HTML_CONTENT

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8890, log_level="info")
