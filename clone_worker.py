#!/usr/bin/env python3
"""
Luna-TTS Q4_K_M Pre-Baked Voice Clone Worker (CPU 极速版)
---------------------------------------------------------
Executes Qwen3-TTS 1.7B Base Q4_K_M quantized voice synthesis using pre-baked .npz voice profiles.
Zero setup overhead, zero audio re-encoding, ultra-fast local CPU inference.
"""

import sys
import os
import io
import time
import subprocess
import numpy as np
import soundfile as sf

sys.path.insert(0, "/media/ryan/UbuntuDATA/AI_PROJECTS/s2s/venv/lib/python3.11/site-packages")
from faster_qwen3_tts import FasterQwen3TTS

BAKED_VOICES_DIR = "/media/ryan/UbuntuDATA/AI_PROJECTS/lunatts/baked_voices"
DEFAULT_VOICE = "sample2"

def main():
    if len(sys.argv) < 3:
        print("Usage: clone_worker.py <text> <output_mp3_path> [voice_name]")
        sys.exit(1)

    text = sys.argv[1]
    out_mp3_path = sys.argv[2]
    voice_name = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else DEFAULT_VOICE

    # Check for baked voice NPZ file
    npz_path = os.path.join(BAKED_VOICES_DIR, f"{voice_name}.npz")
    if not os.path.exists(npz_path):
        npz_path = os.path.join(BAKED_VOICES_DIR, "sample2.npz")
        if not os.path.exists(npz_path):
            npz_path = os.path.join(BAKED_VOICES_DIR, "sample.npz")

    if not os.path.exists(npz_path):
        print(f"[ERROR] No baked voice profile found in {BAKED_VOICES_DIR}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)
    spk_emb = data["spk_emb"]
    ref_codes = data["ref_codes"]
    ref_text = str(data["ref_text"])

    # Load Q4_K_M quantized model for fast CPU execution
    m = FasterQwen3TTS.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base', backend='ggml', quant='Q4_K_M')

    # Dynamically compute max_tokens without artificial 300 token cap (allows full speech completion)
    max_tokens = max(120, min(int(len(text) * 25), 2048))

    res = m.generate_voice_clone(
        text=text,
        language="Chinese",
        ref_spk_emb=spk_emb,
        ref_codes=ref_codes,
        ref_text=ref_text,
        max_new_tokens=max_tokens,
        temperature=0.3,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.1,
        append_silence=True
    )

    audio_data = res[0][0] if isinstance(res[0], list) else res[0]
    sample_rate = res[1]

    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buf.getvalue()

    out_ext = os.path.splitext(out_mp3_path)[1].lower()
    silence_filter = "silenceremove=stop_periods=-1:stop_duration=1.0:stop_threshold=-35dB"
    if out_ext in {".ogg", ".opus"}:
        cmd = ["ffmpeg", "-y", "-i", "pipe:0", "-af", silence_filter, "-c:a", "libopus", "-b:a", "24k", "-ar", "48000", out_mp3_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", "pipe:0", "-af", silence_filter, "-f", "mp3", "-ac", "1", "-ar", "24000", "-b:a", "64k", out_mp3_path]

    subprocess.run(
        cmd,
        input=wav_bytes, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
    )
    print(f"SUCCESS: Generated Q4 pre-baked voice audio -> {out_mp3_path}")

if __name__ == "__main__":
    main()
