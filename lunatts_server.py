#!/usr/bin/env python3
"""
Luna-TTS Official Pre-Baked Voice Inference & Cloning Server (Qwen3-TTS 1.7B Base)
-----------------------------------------------------------------------------------
Runs a dedicated OpenAI-compatible API server + 9002 Cyberpunk Web UI listening on port 8890.
100% local, offline Zero-Shot Voice Cloning using pre-baked .npz voice profiles.
Uses non-blocking asyncio thread pools so Web UI and health checks never spin or freeze.
"""

import os
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
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lunatts_server")

app = FastAPI(title="Luna-TTS Local Voice Engine", version="1.2.0")

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

@app.get("/health")
def health():
    baked_files = [f.replace(".npz", "") for f in os.listdir(BAKED_VOICES_DIR) if f.endswith(".npz")] if os.path.exists(BAKED_VOICES_DIR) else []
    return {
        "status": "ok",
        "service": "Luna-TTS Local Engine",
        "port": 8890,
        "engine_type": "Qwen3-TTS 1.7B Base Q4_K_M Zero-Shot Voice Clone",
        "baked_voices": baked_files,
        "default_voice": "sample2"
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

@app.post("/v1/audio/speech")
async def generate_speech(req: SpeechRequest):
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    voice_name = req.voice or "sample2"
    fmt = (req.response_format or "ogg").lower()
    ext = ".ogg" if fmt in {"ogg", "opus"} else ".mp3"
    media_type = "audio/ogg" if ext == ".ogg" else "audio/mpeg"

    logger.info(f"Speech synthesis request: '{req.input[:30]}...' (voice={voice_name}, format={fmt})")
    start_t = time.perf_counter()

    temp_audio = f"/tmp/luna_out_{time.time_ns()}{ext}"

    try:
        # Non-blocking async execution to keep Uvicorn event loop responsive
        proc = await asyncio.to_thread(
            subprocess.run,
            [PYTHON_BIN, WORKER_SCRIPT, req.input, temp_audio, voice_name],
            capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0:
            logger.error(f"Worker process failed: {proc.stderr}")
            raise RuntimeError(f"Voice clone worker error: {proc.stderr}")

        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
            raise RuntimeError("Worker process failed to produce output audio file")

        with open(temp_audio, "rb") as f:
            audio_bytes = f.read()

        if os.path.exists(temp_audio):
            os.remove(temp_audio)

        elapsed = time.perf_counter() - start_t
        logger.info(f"Speech synthesis complete in {elapsed:.3f}s ({len(audio_bytes)} bytes)")
        return Response(content=audio_bytes, media_type=media_type)
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
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

# 9002 (UbuntuConsole) Fresh Cyberpunk Glassmorphic Design System
HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Luna-TTS 控制台 · 极速零样本语音克隆引擎</title>
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

        .brand-logo {
            width: 38px;
            height: 38px;
            background: var(--accent-gradient);
            border-radius: var(--r-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.2rem;
            box-shadow: 0 0 15px var(--blue-glow);
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
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-2);
            margin-bottom: 0.5rem;
            letter-spacing: 0.01em;
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

        textarea { height: 110px; resize: vertical; line-height: 1.6; }

        .dropzone {
            border: 2px dashed var(--line-strong);
            border-radius: var(--r-md);
            padding: 2rem;
            text-align: center;
            background: var(--bg-elev-1);
            cursor: pointer;
            transition: all 0.2s;
        }

        .dropzone:hover {
            border-color: var(--blue-main);
            background: var(--blue-soft);
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
                <div class="brand-logo">L</div>
                <div class="brand-text">
                    <h1>Luna-TTS 控制台</h1>
                    <p>Qwen3-TTS 1.7B Q4_K_M • 0 MB 显存占用 • 本地 CPU 极速引擎</p>
                </div>
            </div>
            <div class="status-pill">
                <div class="status-dot"></div>
                <span>Port 8890 Online</span>
            </div>
        </div>

        <div class="nav-tabs">
            <button class="tab-btn active" onclick="switchTab('tts')">💬 文本合成试听</button>
            <button class="tab-btn" onclick="switchTab('clone')">🔊 一键克隆声纹</button>
            <button class="tab-btn" onclick="switchTab('manage')">🗂️ 声纹库管理</button>
        </div>

        <!-- TAB 1: TTS -->
        <div id="tab-tts" class="tab-panel active">
            <div class="form-group">
                <label class="form-label">选择目标音色 (Voice Profile)</label>
                <select id="tts-voice-select"></select>
            </div>
            <div class="form-group">
                <label class="form-label">输入要合成的文本 (支持超长文本与智能 1.0s 句尾裁切)</label>
                <textarea id="tts-input-text" placeholder="输入你需要合成的话...">您好，我是 Fina。我已经为您配置好了 9002 风格的 Cyber 极速语音引擎控制台。</textarea>
            </div>
            <button class="cyber-btn" id="tts-submit-btn" onclick="generateSpeech()">
                <span id="tts-btn-text">🚀 开始极速合成</span>
            </button>

            <div class="result-box" id="tts-player-box">
                <label class="form-label">合成结果 (.ogg Opus 原生语音格式)</label>
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
                <div class="dropzone" onclick="document.getElementById('clone-file-input').click()">
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

        function handleFileSelect(input) {
            if (input.files && input.files[0]) {
                document.getElementById('dropzone-label').innerText = '✅ 已选择: ' + input.files[0].name;
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
            btnText.innerHTML = '<span class="spinner"></span> 正在 CPU 高速合成语音...';

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
                btnText.innerText = '🚀 开始极速合成';
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
def index_page():
    return HTML_CONTENT

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8890, log_level="info")
