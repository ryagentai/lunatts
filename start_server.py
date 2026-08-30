#!/usr/bin/env python3
import subprocess
import time
import os
import sys

log_file = "/media/ryan/UbuntuDATA/AI_PROJECTS/lunatts/server.log"
with open(log_file, "w") as f:
    f.write("=== Luna-TTS Server Log ===\n")

proc = subprocess.Popen(
    ["/media/ryan/UbuntuDATA/AI_PROJECTS/s2s/venv/bin/python", "/media/ryan/UbuntuDATA/AI_PROJECTS/lunatts/lunatts_server.py"],
    stdout=open(log_file, "a"),
    stderr=subprocess.STDOUT,
    start_new_session=True
)

print(f"Server started with PID {proc.pid}")
