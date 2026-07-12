#!/usr/bin/env python3
"""Realtime speech client.

Streams mic audio (16 kHz float32) to the server continuously; the server does
VAD/endpointing. Plays back streamed TTS audio (24 kHz float32) through a
persistent output stream for gapless playback.

Barge-in: while the assistant is playing, mic audio is not forwarded (echo
control), but the local level is monitored. Sustained sound stops playback,
sends an interrupt, and forwards the buffered onset of your speech.

Also accepts typed input: plain text sends a text turn, "i"/"stop" interrupts,
"/v <voice>" switches voice, "/q" quits.
"""
import argparse
import asyncio
import base64
import json
import logging
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import websockets

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("speech-client")

MIC_SR = 16000
TTS_SR = 24000
MIC_BLOCK = 512          # 32 ms at 16 kHz
BLOCKS_PER_MSG = 4       # 128 ms per websocket message


class Player:
    """Persistent output stream pulling from a chunk queue (gapless playback)."""

    def __init__(self, sr=TTS_SR, device=None):
        self._lock = threading.Lock()
        self._chunks = deque()
        self._pos = 0
        self._buffered = 0
        self.stream = sd.OutputStream(
            samplerate=sr, channels=1, dtype="float32",
            blocksize=1024, device=device, callback=self._callback)

    def _callback(self, outdata, frames, _time, status):
        if status:
            logger.debug(f"output status: {status}")
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                cur = self._chunks[0]
                take = min(frames - filled, len(cur) - self._pos)
                outdata[filled:filled + take, 0] = cur[self._pos:self._pos + take]
                self._pos += take
                filled += take
                self._buffered -= take
                if self._pos >= len(cur):
                    self._chunks.popleft()
                    self._pos = 0
        if filled < frames:
            outdata[filled:, 0] = 0

    def write(self, pcm: np.ndarray):
        with self._lock:
            self._chunks.append(pcm)
            self._buffered += len(pcm)

    def clear(self):
        with self._lock:
            self._chunks.clear()
            self._pos = 0
            self._buffered = 0

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._buffered > 0

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


class SpeechClient:
    def __init__(self, server_url, barge_rms=0.06, input_device=None,
                 output_device=None, no_mic=False):
        url = server_url.rstrip("/")
        for prefix, repl in (("https", "wss"), ("http", "ws")):
            if url.startswith(prefix + "://"):
                url = repl + url[len(prefix):]
                break
        self.url = url + "/ws"
        self.barge_rms = barge_rms
        self.input_device = input_device
        self.output_device = output_device
        self.no_mic = no_mic
        self.ws = None
        self.running = False
        self.player = Player(device=output_device)
        self.mic_q = None
        self.loop = None
        self.interrupted_utts = set()
        self.assistant_line_open = False

    # --- mic capture -> asyncio queue ---

    def _mic_callback(self, indata, _frames, _time, status):
        if status:
            logger.debug(f"input status: {status}")
        block = indata[:, 0].copy()
        self.loop.call_soon_threadsafe(self.mic_q.put_nowait, block)

    async def send(self, msg: dict):
        await self.ws.send(json.dumps(msg))

    async def send_audio(self, pcm: np.ndarray):
        await self.send({
            "type": "audio",
            "data": base64.b64encode(pcm.astype(np.float32).tobytes()).decode(),
        })

    async def mic_sender(self):
        pre_roll = deque(maxlen=int(1.0 * MIC_SR / MIC_BLOCK))  # ~1s
        loud_run = 0
        batch = []
        while self.running:
            block = await self.mic_q.get()

            if self.player.playing:
                # Echo control: don't forward mic while assistant audio plays,
                # but watch for the user talking over it.
                batch.clear()
                pre_roll.append(block)
                rms = float(np.sqrt(np.mean(block ** 2)))
                loud_run = loud_run + 1 if rms > self.barge_rms else 0
                if loud_run >= 5:  # ~160 ms of sustained sound
                    self._print("\n[barge-in]")
                    self.player.clear()
                    await self.send({"type": "interrupt"})
                    onset = np.concatenate(list(pre_roll))
                    pre_roll.clear()
                    loud_run = 0
                    await self.send_audio(onset)
                continue

            if pre_roll:
                pre_roll.clear()  # stale audio from playback period; drop it
                loud_run = 0

            batch.append(block)
            if len(batch) >= BLOCKS_PER_MSG:
                await self.send_audio(np.concatenate(batch))
                batch = []

    # --- server messages ---

    def _print(self, text, end="\n"):
        if self.assistant_line_open and end == "\n":
            sys.stdout.write("\n")
            self.assistant_line_open = False
        sys.stdout.write(text + end)
        sys.stdout.flush()

    async def receiver(self):
        async for raw in self.ws:
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "ready":
                self._print(f"[connected] llm={msg['llm_model']} "
                            f"voices={','.join(msg['voices'][:4])}...")
            elif mtype == "transcription":
                self._print(f"you: {msg['data']}")
            elif mtype == "chat_chunk":
                if msg.get("utt") in self.interrupted_utts:
                    continue
                prefix = "" if self.assistant_line_open else "assistant: "
                sys.stdout.write(prefix + msg["data"] + " ")
                sys.stdout.flush()
                self.assistant_line_open = True
            elif mtype == "audio_chunk":
                if msg.get("utt") in self.interrupted_utts or not msg.get("pcm"):
                    continue
                pcm = np.frombuffer(base64.b64decode(msg["pcm"]), dtype=np.float32)
                self.player.write(pcm.copy())
            elif mtype == "chat_done":
                self._print(f"[first token {msg.get('t_first_token_ms')}ms, "
                            f"first audio {msg.get('t_first_audio_ms')}ms]")
            elif mtype == "interrupted":
                self.interrupted_utts.add(msg.get("utt"))
                self.player.clear()
                self._print("[interrupted]")
            elif mtype == "vad":
                logger.debug(f"vad: {msg['data']}")
            elif mtype == "voice_changed":
                self._print(f"[voice: {msg['data']}]")
            elif mtype == "error":
                self._print(f"[server error] {msg['data']}")

    # --- typed input ---

    async def repl(self):
        loop = asyncio.get_running_loop()
        while self.running:
            try:
                line = await loop.run_in_executor(None, input)
            except (EOFError, KeyboardInterrupt):
                break
            line = line.strip()
            if not line:
                continue
            if line.lower() in ("i", "stop", "interrupt"):
                self.player.clear()
                await self.send({"type": "interrupt"})
            elif line.startswith("/v "):
                await self.send({"type": "set_voice", "data": line[3:].strip()})
            elif line in ("/q", "/quit", "exit"):
                self.running = False
                break
            else:
                self._print(f"you (text): {line}")
                await self.send({"type": "text", "data": line})

    async def run(self):
        self.loop = asyncio.get_running_loop()
        self.mic_q = asyncio.Queue()
        self.running = True

        self._print(f"connecting to {self.url} ...")
        async with websockets.connect(self.url, max_size=None) as ws:
            self.ws = ws
            self.player.start()

            mic_stream = None
            if not self.no_mic:
                mic_stream = sd.InputStream(
                    samplerate=MIC_SR, channels=1, dtype="float32",
                    blocksize=MIC_BLOCK, device=self.input_device,
                    callback=self._mic_callback)
                mic_stream.start()
                self._print("[mic live] talk, or type a message "
                            "(i=interrupt, /v <voice>, /q=quit)")
            else:
                self._print("[text mode] type a message (/q to quit)")

            tasks = [asyncio.create_task(self.receiver()),
                     asyncio.create_task(self.repl())]
            if mic_stream:
                tasks.append(asyncio.create_task(self.mic_sender()))
            try:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception():
                        raise t.exception()
            finally:
                self.running = False
                if mic_stream:
                    mic_stream.stop()
                    mic_stream.close()
                self.player.stop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", default="http://localhost:8000",
                   help="server URL (default http://localhost:8000, e.g. via SSH tunnel)")
    p.add_argument("--barge-rms", type=float, default=0.06,
                   help="mic RMS threshold to interrupt playback (default 0.06)")
    p.add_argument("--input-device", type=int, default=None)
    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--no-mic", action="store_true", help="text input only")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    client = SpeechClient(args.server, barge_rms=args.barge_rms,
                          input_device=args.input_device,
                          output_device=args.output_device,
                          no_mic=args.no_mic)
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
