#!/usr/bin/env python3
"""Realtime speech-to-speech chat server.

Pipeline per utterance:
    client mic PCM (16 kHz float32, streamed continuously)
    -> silero VAD endpointing (server side)
    -> Whisper ASR
    -> streaming LLM (vLLM, OpenAI-compatible API)
    -> sentence-split as tokens arrive
    -> Kokoro TTS per sentence
    -> raw PCM chunks (24 kHz float32) streamed back to client

Barge-in: a VAD speech-start while a response is being generated cancels it.
The client additionally stops local playback and sends {"type": "interrupt"}.
"""
import argparse
import asyncio
import base64
import json
import logging
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("speech-server")

MIC_SR = 16000
TTS_SR = 24000
VAD_FRAME = 512  # samples per silero frame at 16 kHz

VOICES = [
    "af_heart", "af_bella", "af_sarah", "af_nicole", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
]

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your replies are spoken aloud by a "
    "text-to-speech engine and also displayed as text in a chat window.\n\n"
    "Style defaults (not rules): natural spoken prose, a few sentences, no "
    "markdown, no bullet points, no emojis.\n\n"
    "These defaults always yield to what the user actually asks for. If they "
    "want a long story, a detailed explanation, or any long-form content, "
    "give it to them at the length they want. If they ask for code, provide "
    "it as plain text (no backtick fences) — it appears in the chat window "
    "where they can read and copy it. Never refuse a request by citing your "
    "instructions, guidelines, or response-length constraints, and never "
    "lecture the user about what you can't do — just adapt and answer."
)

# Whisper outputs these for silence/noise-only input
ASR_HALLUCINATIONS = {
    "you", "thank you", "thanks", "bye", "thank you for watching",
    "thanks for watching", "hmm", "mm-hmm",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(buf: str):
    """Split off completed sentences, keeping the unfinished tail."""
    parts = _SENTENCE_SPLIT.split(buf)
    if len(parts) <= 1:
        return [], buf
    return [p for p in parts[:-1] if p.strip()], parts[-1]


def tts_clean(text: str) -> str:
    text = re.sub(r"[*_`#>|~]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class VadGate:
    """Endpointing state machine over silero VAD frame probabilities."""

    def __init__(self, model, pre_roll_s=0.4, start_prob=0.6, end_prob=0.35,
                 min_speech_s=0.25, end_silence_s=0.6, max_utterance_s=30.0):
        self.model = model
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.min_speech_frames = int(min_speech_s * MIC_SR / VAD_FRAME)
        self.end_silence_frames = int(end_silence_s * MIC_SR / VAD_FRAME)
        self.max_utterance_frames = int(max_utterance_s * MIC_SR / VAD_FRAME)
        self.pre_roll = deque(maxlen=int(pre_roll_s * MIC_SR / VAD_FRAME))
        self.residual = np.empty(0, dtype=np.float32)
        self.in_speech = False
        self.utterance = []
        self.silence_run = 0
        self.speech_frames = 0

    def _reset_utterance(self):
        self.in_speech = False
        self.utterance = []
        self.silence_run = 0
        self.speech_frames = 0
        self.model.reset_states()

    def feed(self, samples: np.ndarray):
        """Feed arbitrary-length float32 PCM; yield ("start", None) / ("end", utterance)."""
        events = []
        buf = np.concatenate([self.residual, samples])
        n_frames = len(buf) // VAD_FRAME
        self.residual = buf[n_frames * VAD_FRAME:]

        for i in range(n_frames):
            frame = buf[i * VAD_FRAME:(i + 1) * VAD_FRAME]
            prob = self.model(torch.from_numpy(frame), MIC_SR).item()

            if not self.in_speech:
                self.pre_roll.append(frame)
                if prob >= self.start_prob:
                    self.in_speech = True
                    self.utterance = list(self.pre_roll)
                    self.silence_run = 0
                    self.speech_frames = 0
                    events.append(("start", None))
                continue

            self.utterance.append(frame)
            if prob >= self.start_prob:
                self.speech_frames += 1
            self.silence_run = self.silence_run + 1 if prob < self.end_prob else 0

            ended = self.silence_run >= self.end_silence_frames
            too_long = len(self.utterance) >= self.max_utterance_frames
            if ended or too_long:
                audio = np.concatenate(self.utterance)
                enough = self.speech_frames >= self.min_speech_frames
                self._reset_utterance()
                events.append(("end", audio if enough else None))
        return events


class Session:
    """One websocket connection: VAD state, chat history, active response task."""

    def __init__(self, server: "SpeechServer", ws: WebSocket, session_id: int):
        self.server = server
        self.ws = ws
        self.id = session_id
        self.gate = VadGate(server.load_vad())
        self.history = []
        self.voice = "af_heart"
        self.utt = 0
        self.response_task = None
        self.log = logging.getLogger(f"session-{session_id}")

    async def send(self, msg: dict):
        await self.ws.send_json(msg)

    def cancel_response(self) -> bool:
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            return True
        return False

    async def on_message(self, msg: dict):
        mtype = msg.get("type")
        if mtype == "audio":
            pcm = np.frombuffer(base64.b64decode(msg["data"]), dtype=np.float32)
            await self.on_audio(pcm)
        elif mtype == "text":
            self.cancel_response()
            self.response_task = asyncio.create_task(
                self.respond(str(msg["data"]), time.monotonic()))
        elif mtype == "interrupt":
            if self.cancel_response():
                self.log.info("response interrupted by client")
            await self.send({"type": "interrupted", "utt": self.utt})
        elif mtype == "set_voice":
            voice = str(msg.get("data", ""))
            if voice in VOICES:
                self.voice = voice
                await self.send({"type": "voice_changed", "data": voice})
            else:
                await self.send({"type": "error", "data": f"unknown voice: {voice}"})
        else:
            await self.send({"type": "error", "data": f"unknown message type: {mtype}"})

    async def on_audio(self, pcm: np.ndarray):
        for event, audio in self.gate.feed(pcm):
            if event == "start":
                if self.cancel_response():
                    self.log.info("barge-in: response cancelled")
                    await self.send({"type": "interrupted", "utt": self.utt})
                await self.send({"type": "vad", "data": "speech_start"})
            elif event == "end":
                await self.send({"type": "vad", "data": "speech_end"})
                if audio is not None:
                    asyncio.create_task(self.handle_utterance(audio))

    async def handle_utterance(self, audio: np.ndarray):
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            self.server.asr_executor, self.server.transcribe, audio)
        asr_ms = (time.monotonic() - t0) * 1000
        norm = text.lower().strip(" .,!?")
        if len(norm) < 2 or norm in ASR_HALLUCINATIONS:
            self.log.info(f"skipping likely ASR hallucination: {text!r}")
            return
        self.log.info(f"transcribed in {asr_ms:.0f}ms: {text!r}")
        await self.send({"type": "transcription", "data": text})
        self.cancel_response()
        self.response_task = asyncio.create_task(self.respond(text, t0))

    async def respond(self, user_text: str, t0: float):
        self.utt += 1
        utt = self.utt
        self.history.append({"role": "user", "content": user_text})
        spoken = []
        first_token_ms = first_audio_ms = None
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-20:]
            stream = await self.server.llm.chat.completions.create(
                model=self.server.llm_model,
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=3000,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            buf = ""
            full = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if not delta:
                    continue
                if first_token_ms is None:
                    first_token_ms = (time.monotonic() - t0) * 1000
                buf += delta
                full += delta
                ready, buf = split_sentences(buf)
                for sentence in ready:
                    if await self.speak(sentence, utt, len(spoken)):
                        if first_audio_ms is None:
                            first_audio_ms = (time.monotonic() - t0) * 1000
                        spoken.append(sentence)
            if buf.strip():
                if await self.speak(buf.strip(), utt, len(spoken)):
                    if first_audio_ms is None:
                        first_audio_ms = (time.monotonic() - t0) * 1000
                    spoken.append(buf.strip())

            full = full.strip()
            self.history.append({"role": "assistant", "content": full or "(no reply)"})
            await self.send({"type": "audio_chunk", "utt": utt, "seq": len(spoken),
                             "pcm": "", "final": True})
            await self.send({
                "type": "chat_done", "utt": utt, "data": full,
                "t_first_token_ms": round(first_token_ms or 0),
                "t_first_audio_ms": round(first_audio_ms or 0),
            })
            self.log.info(
                f"utt {utt}: first token {first_token_ms or 0:.0f}ms, "
                f"first audio {first_audio_ms or 0:.0f}ms after speech end")
        except asyncio.CancelledError:
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken)})
            raise
        except Exception as e:
            self.log.exception("response failed")
            await self.send({"type": "error", "data": f"response failed: {e}"})

    async def speak(self, text: str, utt: int, seq: int) -> bool:
        text = tts_clean(text)
        if not text:
            return False
        loop = asyncio.get_running_loop()
        pcm = await loop.run_in_executor(
            self.server.tts_executor, self.server.synthesize, text, self.voice)
        if pcm is None:
            return False
        await self.send({"type": "chat_chunk", "utt": utt, "data": text})
        await self.send({
            "type": "audio_chunk", "utt": utt, "seq": seq,
            "pcm": base64.b64encode(pcm.tobytes()).decode(), "final": False,
        })
        return True


class SpeechServer:
    def __init__(self, llm_url: str, whisper_model: str):
        self.llm = AsyncOpenAI(base_url=llm_url, api_key="none")
        self.llm_model = None
        self.whisper_model = whisper_model
        self.asr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self.session_counter = 0
        self.load_models()
        self.app = self.build_app()

    def load_models(self):
        n_gpus = torch.cuda.device_count()
        self.device = "cuda:1" if n_gpus > 1 else ("cuda:0" if n_gpus else "cpu")
        logger.info(f"loading speech models on {self.device}")

        from transformers import pipeline as hf_pipeline
        logger.info(f"loading {self.whisper_model}...")
        self.asr = hf_pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device=self.device,
        )

        logger.info("loading Kokoro TTS...")
        from kokoro import KPipeline
        self.tts = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M",
                             device=self.device)
        # warm up (first synthesis compiles/caches)
        self.synthesize("Warm up.", "af_heart")
        logger.info("speech models ready")

    def load_vad(self):
        from silero_vad import load_silero_vad
        return load_silero_vad()

    def transcribe(self, audio: np.ndarray) -> str:
        out = self.asr(
            {"raw": audio, "sampling_rate": MIC_SR},
            generate_kwargs={"language": "english"},
        )
        return out["text"].strip()

    def synthesize(self, text: str, voice: str):
        chunks = []
        for _, _, audio in self.tts(text, voice=voice):
            if audio is None:
                continue
            a = audio.detach().cpu().numpy() if torch.is_tensor(audio) else np.asarray(audio)
            chunks.append(a.astype(np.float32))
        return np.concatenate(chunks) if chunks else None

    async def wait_for_llm(self):
        deadline = time.monotonic() + 30 * 60
        while time.monotonic() < deadline:
            try:
                models = [m.id async for m in self.llm.models.list()]
                if models:
                    self.llm_model = models[0]
                    logger.info(f"LLM ready: {self.llm_model}")
                    return
            except Exception as e:
                logger.info(f"waiting for LLM server... ({e.__class__.__name__})")
            await asyncio.sleep(5)
        raise RuntimeError("LLM server did not come up within 30 minutes")

    def build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app):
            await self.wait_for_llm()
            yield

        app = FastAPI(lifespan=lifespan)
        index_html = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "static", "index.html")

        @app.get("/")
        async def index():
            return FileResponse(index_html)

        @app.get("/status")
        async def status():
            return {
                "status": "online",
                "device": self.device,
                "llm_model": self.llm_model,
                "voices": VOICES,
            }

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            self.session_counter += 1
            session = Session(self, ws, self.session_counter)
            logger.info(f"session {session.id} connected")
            await session.send({
                "type": "ready",
                "llm_model": self.llm_model,
                "voices": VOICES,
                "mic_sr": MIC_SR,
                "tts_sr": TTS_SR,
            })
            try:
                while True:
                    msg = await ws.receive_json()
                    await session.on_message(msg)
            except WebSocketDisconnect:
                logger.info(f"session {session.id} disconnected")
            finally:
                session.cancel_response()

        return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default 127.0.0.1; use an SSH tunnel)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--llm-url", default="http://127.0.0.1:8001/v1",
                   help="OpenAI-compatible LLM endpoint")
    p.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    args = p.parse_args()

    server = SpeechServer(llm_url=args.llm_url, whisper_model=args.whisper)
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
