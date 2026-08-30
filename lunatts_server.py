#!/usr/bin/env python3
"""
Luna-TTS Official Pre-Baked Voice Inference & Cloning Server (Qwen3-TTS 1.7B Base)
-----------------------------------------------------------------------------------
Runs a dedicated OpenAI-compatible API server + Web UI listening on port 8890.
100% local, offline Zero-Shot Voice Cloning using pre-baked .npz voice profiles.
Supports live Web UI for voice preview, text synthesis, and zero-shot voice cloning.
"""

import os
import sys
import time
import shutil
import logging
import subprocess
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lunatts_server")

app = FastAPI(title="Luna-TTS Local Voice Engine", version="1.1.0")

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
    response_format: Optional[str] = "mp3"
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
                
                # Check reference audio
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
            mime = "audio/mpeg" if ext == ".mp3" else f"audio/{ext[1:]}"
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
        proc = subprocess.run(
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

        # Save to ref_audio
        dest_ref = os.path.join(REF_AUDIO_DIR, f"{voice_name}{ext}")
        shutil.copyfile(tmp_upload, dest_ref)

        # Execute voice baker
        cmd = [PYTHON_BIN, BAKER_SCRIPT, tmp_upload, voice_name]
        if ref_text and ref_text.strip():
            cmd.append(ref_text.strip())

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

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

# Modern Glassmorphic Web UI
HTML_CONTENT = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Luna-TTS 极速零样本语音克隆引擎</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-grad: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            --panel-bg: rgba(30, 41, 59, 0.75);
            --panel-border: rgba(255, 255, 255, 0.12);
            --accent-purple: #818cf8;
            --accent-indigo: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.35);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --radius-lg: 16px;
            --radius-md: 10px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', -apple-system, sans-serif; }
        
        body {
            background: var(--bg-grad);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius-lg);
            padding: 2.5rem;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #c084fc, #818cf8, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .tabs {
            display: flex;
            gap: 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
        }

        .tab-btn {
            background: none;
            border: none;
            color: var(--text-muted);
            font-size: 1rem;
            font-weight: 500;
            padding: 0.75rem 1.25rem;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.2s ease;
        }

        .tab-btn.active {
            color: var(--accent-purple);
            border-bottom-color: var(--accent-purple);
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .form-group {
            margin-bottom: 1.5rem;
        }

        label {
            display: block;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-main);
            margin-bottom: 0.5rem;
        }

        input[type="text"], select, textarea {
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius-md);
            color: var(--text-main);
            padding: 0.85rem 1rem;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }

        input[type="text"]:focus, select:focus, textarea:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 10px var(--accent-glow);
        }

        textarea { height: 120px; resize: vertical; }

        .file-dropzone {
            border: 2px dashed var(--panel-border);
            border-radius: var(--radius-md);
            padding: 2rem;
            text-align: center;
            background: rgba(15, 23, 42, 0.4);
            cursor: pointer;
            transition: border-color 0.2s, background 0.2s;
        }

        .file-dropzone:hover {
            border-color: var(--accent-purple);
            background: rgba(99, 102, 241, 0.05);
        }

        .btn {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-purple));
            color: #fff;
            border: none;
            border-radius: var(--radius-md);
            padding: 0.9rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 15px var(--accent-glow);
            transition: transform 0.15s ease, opacity 0.2s;
        }

        .btn:hover { opacity: 0.92; transform: translateY(-1px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

        .player-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(15, 23, 42, 0.7);
            border-radius: var(--radius-md);
            border: 1px solid var(--panel-border);
            display: none;
        }

        audio { width: 100%; margin-top: 0.5rem; }

        .voice-card-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .voice-card {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--panel-border);
            border-radius: var(--radius-md);
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .voice-card h4 { color: var(--accent-purple); font-size: 1rem; }
        .voice-card span { font-size: 0.8rem; color: var(--text-muted); }

        .spinner {
            display: inline-block;
            width: 18px; height: 18px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s ease-in-out infinite;
            margin-right: 8px;
            vertical-align: middle;
        }

        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ Luna-TTS 极速零样本语音引擎</h1>
            <p>Qwen3-TTS 1.7B Base Q4_K_M • 100% 本地 CPU 运行 • 零 GPU 显存占用</p>
        </div>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('tts')">💬 文本合成试听</button>
            <button class="tab-btn" onclick="switchTab('clone')">🔊 一键克隆新音色</button>
            <button class="tab-btn" onclick="switchTab('manage')">🗂️ 音色库管理</button>
        </div>

        <!-- TAB 1: TTS -->
        <div id="tab-tts" class="tab-content active">
            <div class="form-group">
                <label>选择说话人 (Voice Profile)</label>
                <select id="tts-voice-select"></select>
            </div>
            <div class="form-group">
                <label>输入要合成的文本</label>
                <textarea id="tts-input-text" placeholder="输入你想让 Fina 或自定义音色说的话...">您好，我是 Fina。我已经为您成功配置了本地 Q4 极速语音合成系统。</textarea>
            </div>
            <button class="btn" id="tts-submit-btn" onclick="generateSpeech()">
                <span id="tts-btn-text">🚀 开始合成语音</span>
            </button>

            <div class="player-box" id="tts-player-box">
                <label>合成音频结果</label>
                <audio id="tts-audio-player" controls></audio>
            </div>
        </div>

        <!-- TAB 2: CLONE -->
        <div id="tab-clone" class="tab-content">
            <div class="form-group">
                <label>新音色标识名称 (如 fina_v2, ryan_voice)</label>
                <input type="text" id="clone-voice-name" placeholder="仅限英文、数字和下划线">
            </div>
            <div class="form-group">
                <label>上传参考语音文件 (MP3 / WAV / M4A / FLAC)</label>
                <div class="file-dropzone" onclick="document.getElementById('clone-file-input').click()">
                    <span id="file-dropzone-label">📁 点击或拖入音频文件 (建议 5~15 秒清晰人声)</span>
                    <input type="file" id="clone-file-input" accept="audio/*" style="display:none;" onchange="handleFileSelect(this)">
                </div>
            </div>
            <div class="form-group">
                <label>参考文本 (可选，留空则自动运行 Whisper ASR 识别)</label>
                <input type="text" id="clone-ref-text" placeholder="留空自动识别，或者手动输入音频里说话的内容">
            </div>
            <button class="btn" id="clone-submit-btn" onclick="cloneVoice()">
                <span id="clone-btn-text">⚡ 开始提纯并固化声纹</span>
            </button>
        </div>

        <!-- TAB 3: MANAGE -->
        <div id="tab-manage" class="tab-content">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3>已固化的声纹 Profile</h3>
                <button class="tab-btn" onclick="loadVoices()" style="border:none;">🔄 刷新列表</button>
            </div>
            <div class="voice-card-list" id="voice-card-container"></div>
        </div>
    </div>

    <script>
        let availableVoices = [];

        function switchTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            event.target.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');
            if (tabId === 'manage') loadVoices();
        }

        function handleFileSelect(input) {
            if (input.files && input.files[0]) {
                document.getElementById('file-dropzone-label').innerText = '✅ 已选择: ' + input.files[0].name;
            }
        }

        async function loadVoices() {
            try {
                const res = await fetch('/v1/voices');
                const data = await res.json();
                availableVoices = data.voices || [];
                
                // Update select dropdown
                const select = document.getElementById('tts-voice-select');
                select.innerHTML = '';
                availableVoices.forEach(v => {
                    const opt = document.createElement('option');
                    opt.value = v.name;
                    opt.innerText = v.name + (v.name === 'sample2' ? ' (Fina 专属音色)' : '');
                    select.appendChild(opt);
                });

                // Update manage grid
                const container = document.getElementById('voice-card-container');
                container.innerHTML = '';
                availableVoices.forEach(v => {
                    const card = document.createElement('div');
                    card.className = 'voice-card';
                    card.innerHTML = `
                        <h4>🎙️ ${v.name}</h4>
                        <span>特征固化包: ${v.npz_size_kb} KB (.npz)</span>
                        ${v.has_ref_audio ? `<audio controls src="${v.ref_audio_url}" style="width:100%; height:32px; margin-top:0.3rem;"></audio>` : '<span>原音频文件: 未发现</span>'}
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
                    body: JSON.stringify({ voice, input })
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
                btnText.innerText = '🚀 开始合成语音';
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
            btnText.innerHTML = '<span class="spinner"></span> 正在提取声纹与 Whisper 文本...';

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

                alert(`🎉 音色 '${data.voice_name}' 提纯与固化成功！\n特征文件: ${data.npz_size_kb} KB\n原始音频已归档: ${data.ref_audio_saved}`);
                
                // Clear fields & refresh
                document.getElementById('clone-voice-name').value = '';
                document.getElementById('clone-file-input').value = '';
                document.getElementById('file-dropzone-label').innerText = '📁 点击或拖入音频文件 (建议 5~15 秒清晰人声)';
                loadVoices();
                switchTab('tts');
            } catch (err) {
                alert('音色克隆错误: ' + err.message);
            } finally {
                btn.disabled = false;
                btnText.innerText = '⚡ 开始提纯并固化声纹';
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
