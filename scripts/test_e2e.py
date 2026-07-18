#!/usr/bin/env python3
"""End-to-end test: text turn + audio turn against a running speech server.

Usage: python scripts/test_e2e.py [--server http://localhost:8000] [--wav sample/female.wav]
"""
import argparse
import asyncio
import base64
import json
import time

import numpy as np
import soundfile as sf
import websockets

MIC_SR = 16000


async def collect_response(ws, timeout=90):
    """Read messages until chat_done; return (events, transcription, text, audio_samples)."""
    events = []
    transcription = None
    text = None
    audio_samples = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
        msg = json.loads(raw)
        events.append(msg["type"])
        if msg["type"] == "transcription":
            transcription = msg["data"]
        elif msg["type"] == "endpoint":
            print(f"  endpoint: {msg['data']} ({msg['source']})")
        elif msg["type"] == "audio_chunk" and msg.get("pcm"):
            audio_samples += len(base64.b64decode(msg["pcm"])) // 4
        elif msg["type"] == "error":
            raise RuntimeError(f"server error: {msg['data']}")
        elif msg["type"] == "chat_done":
            text = msg["data"]
            return events, transcription, text, audio_samples, msg
    raise TimeoutError("no chat_done within timeout")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="http://localhost:8000")
    p.add_argument("--wav", default="sample/female.wav")
    args = p.parse_args()

    url = args.server.replace("http", "ws", 1).rstrip("/") + "/ws"
    print(f"connecting to {url}")
    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        print(f"ready: llm={ready['llm_model']}")

        # --- test 1: text turn ---
        print("\n[test 1] text turn")
        t0 = time.monotonic()
        await ws.send(json.dumps({"type": "text",
                                  "data": "Reply with one short sentence: what are you?"}))
        _, _, text, samples, done = await collect_response(ws)
        dt = time.monotonic() - t0
        print(f"  reply: {text!r}")
        print(f"  audio: {samples} samples ({samples / 24000:.2f}s)")
        print(f"  total {dt:.2f}s, first token {done['t_first_token_ms']}ms, "
              f"first audio {done['t_first_audio_ms']}ms")
        assert text and samples > 0, "no reply or no audio"

        # --- test 2: audio turn (stream a wav like a mic would) ---
        print(f"\n[test 2] audio turn ({args.wav})")
        audio, sr = sf.read(args.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != MIC_SR:
            n = int(len(audio) * MIC_SR / sr)
            audio = np.interp(np.linspace(0, len(audio) - 1, n),
                              np.arange(len(audio)), audio).astype(np.float32)
        print(f"  sending {len(audio) / MIC_SR:.2f}s of audio")
        chunk = 2048  # 128 ms
        for i in range(0, len(audio), chunk):
            await ws.send(json.dumps({
                "type": "audio",
                "data": base64.b64encode(audio[i:i + chunk].tobytes()).decode()}))
            await asyncio.sleep(0.01)  # faster than realtime but paced
        silence = np.zeros(chunk, dtype=np.float32)
        for _ in range(24):  # ~3 s of silence: covers the hard-end path too
            await ws.send(json.dumps({
                "type": "audio",
                "data": base64.b64encode(silence.tobytes()).decode()}))
            await asyncio.sleep(0.01)

        t0 = time.monotonic()
        _, transcription, text, samples, done = await collect_response(ws)
        print(f"  transcription: {transcription!r}")
        print(f"  reply: {text!r}")
        print(f"  audio: {samples} samples ({samples / 24000:.2f}s)")
        print(f"  first token {done['t_first_token_ms']}ms, "
              f"first audio {done['t_first_audio_ms']}ms (after speech end)")
        assert transcription and text and samples > 0

        print("\nall tests passed")


if __name__ == "__main__":
    asyncio.run(main())
