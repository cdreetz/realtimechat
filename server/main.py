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
import io
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

TURN_CLASSIFIER_PROMPT = (
    "You judge whether a speaker has finished their conversational turn. "
    "Given a voice transcript, answer DONE if it is a complete utterance the "
    "assistant should respond to now, or WAIT if the speaker trailed off "
    "mid-thought and will likely continue. Answer with exactly one word: "
    "DONE or WAIT."
)

# words a turn essentially never ends on -> speaker is mid-thought
MIDTHOUGHT_ENDINGS = {
    "and", "but", "or", "so", "because", "if", "when", "while", "then",
    "um", "uh", "like", "the", "a", "an", "to", "with", "of", "for", "in",
    "on", "at", "is", "are", "was", "were", "i", "we", "they", "it's",
    "that", "my", "your", "his", "her", "their", "very", "really",
}


def turn_heuristic(text: str):
    """Fast local verdict: "done", "wait", or None (ambiguous -> ask the LLM)."""
    t = text.strip().lower()
    if not t:
        return "wait"
    if t.endswith("?"):
        return "done"
    if t.endswith(("...", "…", ",", "-", "–", ":")):
        return "wait"
    words = t.rstrip(".!").split()
    if words and words[-1] in MIDTHOUGHT_ENDINGS:
        return "wait"
    return None

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


def audio_to_wav_b64(audio: np.ndarray) -> str:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, MIC_SR, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


class VadGate:
    """Endpointing state machine over silero VAD frame probabilities.

    Emits ("start", None) at speech onset, ("pause", audio_so_far) after a
    short silence (a candidate end-of-turn for semantic evaluation),
    ("resume", None) if speech continues after a pause, and ("end", audio)
    after a long silence or the utterance length cap (audio is None if there
    was not enough speech to bother with).
    """

    def __init__(self, model, pre_roll_s=0.4, start_prob=0.6, end_prob=0.35,
                 min_speech_s=0.25, pause_silence_s=0.35, end_silence_s=2.4,
                 max_utterance_s=45.0):
        self.model = model
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.min_speech_frames = int(min_speech_s * MIC_SR / VAD_FRAME)
        self.pause_frames = int(pause_silence_s * MIC_SR / VAD_FRAME)
        self.end_silence_frames = int(end_silence_s * MIC_SR / VAD_FRAME)
        self.max_utterance_frames = int(max_utterance_s * MIC_SR / VAD_FRAME)
        self.pre_roll = deque(maxlen=int(pre_roll_s * MIC_SR / VAD_FRAME))
        self.residual = np.empty(0, dtype=np.float32)
        self.in_speech = False
        self.utterance = []
        self.silence_run = 0
        self.speech_frames = 0
        self.pause_emitted = False

    def force_reset(self):
        """Drop the current utterance (used when a pause is judged end-of-turn)."""
        self.in_speech = False
        self.utterance = []
        self.silence_run = 0
        self.speech_frames = 0
        self.pause_emitted = False
        self.model.reset_states()

    def feed(self, samples: np.ndarray):
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
                    self.pause_emitted = False
                    events.append(("start", None))
                continue

            self.utterance.append(frame)
            if prob >= self.start_prob:
                self.speech_frames += 1
            if prob < self.end_prob:
                self.silence_run += 1
            else:
                if self.pause_emitted:
                    self.pause_emitted = False
                    events.append(("resume", None))
                self.silence_run = 0

            if (not self.pause_emitted
                    and self.silence_run == self.pause_frames
                    and self.speech_frames >= self.min_speech_frames):
                self.pause_emitted = True
                events.append(("pause", np.concatenate(self.utterance)))

            ended = self.silence_run >= self.end_silence_frames
            too_long = len(self.utterance) >= self.max_utterance_frames
            if ended or too_long:
                audio = np.concatenate(self.utterance)
                enough = self.speech_frames >= self.min_speech_frames
                self.force_reset()
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
        self.eval_task = None
        self.gate_gen = 0          # bumped on start/resume; stale evals discard
        self.pause_transcript = None  # (gen, text) reusable at hard end
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

    def cancel_eval(self):
        if self.eval_task and not self.eval_task.done():
            self.eval_task.cancel()

    async def on_audio(self, pcm: np.ndarray):
        for event, audio in self.gate.feed(pcm):
            if event == "start":
                self.gate_gen += 1
                self.cancel_eval()
                if self.cancel_response():
                    self.log.info("barge-in: response cancelled")
                    await self.send({"type": "interrupted", "utt": self.utt})
                await self.send({"type": "vad", "data": "speech_start"})
            elif event == "resume":
                self.gate_gen += 1
                self.cancel_eval()
                await self.send({"type": "vad", "data": "resume"})
            elif event == "pause":
                await self.send({"type": "vad", "data": "pause"})
                self.eval_task = asyncio.create_task(
                    self.evaluate_pause(audio, self.gate_gen))
            elif event == "end":
                self.cancel_eval()
                await self.send({"type": "vad", "data": "speech_end"})
                if audio is not None:
                    asyncio.create_task(self.finalize_turn(audio))

    async def evaluate_pause(self, audio: np.ndarray, gen: int):
        """Short pause: transcribe and judge whether the turn is complete."""
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            self.server.asr_executor, self.server.transcribe, audio)
        if gen != self.gate_gen:
            return  # user resumed while we transcribed
        norm = text.lower().strip(" .,!?")
        if len(norm) < 2 or norm in ASR_HALLUCINATIONS:
            return  # noise; let the hard end handle it
        verdict = turn_heuristic(text)
        source = "heuristic"
        if verdict is None:
            verdict = await self.classify_turn(text)
            source = "llm"
        if gen != self.gate_gen:
            return
        ms = (time.monotonic() - t0) * 1000
        self.log.info(f"endpoint {verdict} ({source}, {ms:.0f}ms): {text!r}")
        await self.send({"type": "endpoint", "data": verdict,
                         "source": source, "transcript": text})
        if verdict == "done":
            self.gate.force_reset()
            self.gate_gen += 1
            await self.start_turn(text, t0, audio)
        else:
            self.pause_transcript = (gen, text)

    async def classify_turn(self, text: str) -> str:
        if not self.server.classifier_model:
            return "done"
        try:
            kwargs = {}
            if self.server.wants_no_thinking(self.server.classifier_model):
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            r = await self.server.classifier_llm.chat.completions.create(
                model=self.server.classifier_model,
                messages=[{"role": "system", "content": TURN_CLASSIFIER_PROMPT},
                          {"role": "user", "content": text}],
                max_tokens=3,
                temperature=0.0,
                **kwargs,
            )
            out = (r.choices[0].message.content or "").lower()
            return "wait" if "wait" in out else "done"
        except Exception as e:
            self.log.warning(f"turn classifier failed ({e}); assuming done")
            return "done"

    async def finalize_turn(self, audio: np.ndarray):
        """Hard end (long silence / length cap): respond even if mid-thought."""
        t0 = time.monotonic()
        if self.pause_transcript and self.pause_transcript[0] == self.gate_gen:
            text = self.pause_transcript[1]
        else:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                self.server.asr_executor, self.server.transcribe, audio)
        self.pause_transcript = None
        norm = text.lower().strip(" .,!?")
        if len(norm) < 2 or norm in ASR_HALLUCINATIONS:
            self.log.info(f"skipping likely ASR hallucination: {text!r}")
            return
        await self.start_turn(text, t0, audio)

    async def start_turn(self, text: str, t0: float, audio: np.ndarray = None):
        self.pause_transcript = None
        self.log.info(f"turn: {text!r}")
        await self.send({"type": "transcription", "data": text})
        self.cancel_response()
        self.response_task = asyncio.create_task(self.respond(text, t0, audio))

    async def respond(self, user_text: str, t0: float, audio: np.ndarray = None):
        self.utt += 1
        utt = self.utt
        self.history.append({"role": "user", "content": user_text})
        spoken = []
        first_token_ms = first_audio_ms = None
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-20:]
            kwargs = dict(
                model=self.server.llm_model,
                stream=True,
                temperature=0.7,
                max_tokens=None if self.server.reasoning_effort else 3000,
            )
            if self.server.reasoning_effort:
                kwargs["reasoning_effort"] = self.server.reasoning_effort
            if self.server.wants_no_thinking(self.server.llm_model):
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

            # Audio-native models hear the actual utterance instead of the
            # transcript (current turn only; history stays text).
            if audio is not None and self.server.audio_input:
                audio_messages = messages[:-1] + [{
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {
                        "data": audio_to_wav_b64(audio), "format": "wav"}}],
                }]
                try:
                    stream = await self.server.llm.chat.completions.create(
                        messages=audio_messages, **kwargs)
                except Exception as e:
                    self.log.warning(
                        f"audio input rejected ({e.__class__.__name__}: {e}); "
                        "falling back to transcript")
                    stream = await self.server.llm.chat.completions.create(
                        messages=messages, **kwargs)
            else:
                stream = await self.server.llm.chat.completions.create(
                    messages=messages, **kwargs)
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
    def __init__(self, llm_url: str, whisper_model: str, llm_model: str = None,
                 llm_api_key: str = None, classifier_url: str = None,
                 reasoning_effort: str = None, audio_input: bool = False):
        self.llm = AsyncOpenAI(base_url=llm_url, api_key=llm_api_key or "none")
        self.llm_model = llm_model
        self.reasoning_effort = reasoning_effort
        self.audio_input = audio_input
        self.classifier_llm = (AsyncOpenAI(base_url=classifier_url, api_key="none")
                               if classifier_url else self.llm)
        self.classifier_model = None
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

    def wants_no_thinking(self, model: str) -> bool:
        """Local Qwen hybrid-thinking models need enable_thinking=False."""
        return model is not None and "qwen" in model.lower()

    async def wait_for_llm(self):
        if self.llm_model is None:
            deadline = time.monotonic() + 30 * 60
            while time.monotonic() < deadline:
                try:
                    models = [m.id async for m in self.llm.models.list()]
                    if models:
                        self.llm_model = models[0]
                        break
                except Exception as e:
                    logger.info(f"waiting for LLM server... ({e.__class__.__name__})")
                await asyncio.sleep(5)
            if self.llm_model is None:
                raise RuntimeError("LLM server did not come up within 30 minutes")
        logger.info(f"LLM: {self.llm_model} (audio_input={self.audio_input}, "
                    f"reasoning_effort={self.reasoning_effort})")

        # Turn classifier prefers a fast local model; degrade to heuristics-only.
        if self.classifier_llm is self.llm:
            self.classifier_model = self.llm_model
        else:
            try:
                models = [m.id async for m in self.classifier_llm.models.list()]
                self.classifier_model = models[0] if models else None
            except Exception:
                self.classifier_model = None
        if self.classifier_model:
            logger.info(f"turn classifier: {self.classifier_model}")
        else:
            logger.warning("no turn classifier LLM reachable; heuristics only")

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
                session.cancel_eval()

        return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default 127.0.0.1; use an SSH tunnel)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--llm-url", default="http://127.0.0.1:8001/v1",
                   help="OpenAI-compatible LLM endpoint")
    p.add_argument("--llm-model", default=None,
                   help="model id (default: first model the endpoint lists)")
    p.add_argument("--llm-api-key-env", default=None, metavar="ENV_VAR",
                   help="name of an env var holding the LLM API key")
    p.add_argument("--reasoning-effort", default=None,
                   choices=["low", "medium", "high"],
                   help="pass reasoning_effort to the LLM (reasoning models only)")
    p.add_argument("--audio-input", action="store_true",
                   help="send user turns as audio to an audio-native LLM "
                        "(falls back to the transcript if rejected)")
    p.add_argument("--classifier-url", default="http://127.0.0.1:8001/v1",
                   help="fast local LLM for end-of-turn classification "
                        "(heuristics-only if unreachable)")
    p.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    args = p.parse_args()

    api_key = os.environ.get(args.llm_api_key_env) if args.llm_api_key_env else None
    if args.llm_api_key_env and not api_key:
        p.error(f"--llm-api-key-env: ${args.llm_api_key_env} is not set")
    classifier_url = None if args.classifier_url == args.llm_url else args.classifier_url

    server = SpeechServer(
        llm_url=args.llm_url, whisper_model=args.whisper,
        llm_model=args.llm_model, llm_api_key=api_key,
        classifier_url=classifier_url, reasoning_effort=args.reasoning_effort,
        audio_input=args.audio_input)
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
