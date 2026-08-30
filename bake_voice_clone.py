#!/usr/bin/env python3
"""
Luna-TTS Q4_K_M Voice Baking Utility (把声音克隆一键固化到本地 - Q4 极速版)
------------------------------------------------------------------------
Bakes reference audio into pre-computed voice profile (.npz) using Q4_K_M GGUF model.

Usage:
  python3 bake_voice_clone.py <input_audio_path> <voice_name>
"""

import sys
import os
import time
import subprocess
import numpy as np

sys.path.insert(0, "/media/ryan/UbuntuDATA/AI_PROJECTS/s2s/venv/lib/python3.11/site-packages")
from faster_qwen3_tts import FasterQwen3TTS
from faster_qwen3_tts.ggml_backend import _load_ref_audio_24k
from faster_whisper import WhisperModel

BAKED_VOICES_DIR = "/media/ryan/UbuntuDATA/AI_PROJECTS/lunatts/baked_voices"
REF_AUDIO_DIR = "/media/ryan/UbuntuDATA/AI_PROJECTS/lunatts/ref_audio"

def bake_voice(input_audio_path: str, voice_name: str, ref_text_override: str = None):
    if not os.path.exists(input_audio_path):
        print(f"[ERROR] Audio file does not exist: {input_audio_path}")
        sys.exit(1)

    os.makedirs(BAKED_VOICES_DIR, exist_ok=True)
    os.makedirs(REF_AUDIO_DIR, exist_ok=True)

    # Save original reference audio to ref_audio/
    ext = os.path.splitext(input_audio_path)[1].lower() or ".mp3"
    ref_dest_path = os.path.join(REF_AUDIO_DIR, f"{voice_name}{ext}")
    if os.path.abspath(input_audio_path) != os.path.abspath(ref_dest_path):
        subprocess.run(["cp", "-f", input_audio_path, ref_dest_path], check=False)

    out_npz_path = os.path.join(BAKED_VOICES_DIR, f"{voice_name}.npz")

    print(f"\n=======================================================")
    print(f" 正在固化 Q4 声音样本: {input_audio_path} -> {voice_name}")
    print(f"=======================================================\n")

    # Step 1: Convert audio to 24kHz mono WAV
    tmp_wav = f"/tmp/bake_{voice_name}_24k.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_audio_path, "-ar", "24000", "-ac", "1", tmp_wav],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )

    # Step 2: Auto-transcribe ref_text using Whisper ASR or use override
    if ref_text_override and ref_text_override.strip():
        ref_text = ref_text_override.strip()
        print(f"[1/3] 使用手动提供的参考文本: '{ref_text}'")
    else:
        print("[1/3] 正在使用 Whisper ASR 提取参考语音文本...")
        t0 = time.perf_counter()
        asr = WhisperModel("small", device="cpu", compute_type="int8")
        segs, _ = asr.transcribe(tmp_wav, language="zh")
        ref_text = "".join([s.text for s in segs]).strip()
        print(f"   └─ ASR 识别完成 ({time.perf_counter()-t0:.2f}s): '{ref_text}'")

    # Step 3: Extract speaker embedding and ref_codes
    print("[2/3] 正在通过 Qwen3-TTS Q4_K_M 模型提取声纹特征与 Audio Token...")
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', backend='ggml', quant='Q4_K_M')
    audio_24k = _load_ref_audio_24k(tmp_wav, append_silence=True)
    vref, _ = model._get_or_extract_voice_ref(audio_24k, append_silence=True)
    print(f"   └─ 声纹提取完成 ({time.perf_counter()-t0:.2f}s)")

    # Step 4: Save to compressed NPZ
    print("[3/3] 正在把 Q4 克隆特征固化保存到本地硬盘...")
    np.savez_compressed(
        out_npz_path,
        spk_emb=np.array(vref.ref_spk_emb, dtype=np.float32),
        ref_codes=np.array(vref.ref_codes, dtype=np.int32),
        ref_text=np.array(ref_text, dtype=object)
    )

    if os.path.exists(tmp_wav):
        os.remove(tmp_wav)

    print(f"\n✅ Q4 声音克隆固化成功！")
    print(f" 固化文件位置: {out_npz_path} ({os.path.getsize(out_npz_path)/1024:.1f} KB)\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 bake_voice_clone.py <input_audio_path> <voice_name>")
        sys.exit(1)

    bake_voice(sys.argv[1], sys.argv[2])
