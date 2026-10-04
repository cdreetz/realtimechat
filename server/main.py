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
import uuid
from pathlib import Path
import hashlib
import platform
import importlib.metadata
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from openai import AsyncOpenAI

from tools import TOOLS, run_tool
from recording import Recording

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
    "The user is speaking out loud and their speech reaches you as an "
    "automatic transcription, so expect transcription artifacts: misspelled "
    "or wrong-sounding words, homophones (e.g. 'right' for 'write', 'to' "
    "for 'two'), missing punctuation, split or merged words, and occasional "
    "garbled fragments. Interpret what the user most plausibly meant given "
    "the conversation context instead of taking odd wording literally, and "
    "don't point out or correct their 'spelling' — they never typed "
    "anything. If a transcription is too garbled to infer the intent, "
    "briefly ask them to repeat it.\n\n"
    "Style defaults (not rules): natural spoken prose, a few sentences, no "
    "markdown, no bullet points, no emojis.\n\n"
    "These defaults always yield to what the user actually asks for. If they "
    "want a long story, a detailed explanation, or any long-form content, "
    "give it to them at the length they want. If they ask for code, provide "
    "it as plain text (no backtick fences) — it appears in the chat window "
    "where they can read and copy it. Never refuse a request by citing your "
    "instructions, guidelines, or response-length constraints, and never "
    "lecture the user about what you can't do — just adapt and answer.\n\n"
    "You may have tools available — some built in (like web search), some "
    "provided by the user's application. Use a tool whenever the answer "
    "depends on current or verifiable facts, or when the user asks you to "
    "act on their environment. Before your first tool call in a turn, say "
    "one very short phrase like 'Let me look that up.' or 'Sure.' so the "
    "user hears something while it runs. After using tools, answer "
    "conversationally — summarize, never read URLs or raw results aloud.\n\n"
    "Never claim an action happened unless you actually called the tool "
    "and saw its result in this conversation — saying 'done' or 'fixed' "
    "without having made the tool call is a serious failure; announce, "
    "then immediately call the tool in the same turn. Older parts of long "
    "conversations get dropped, so if you are unsure what you did or what "
    "state the user's environment is in, check with a tool (list files, "
    "read the file, query state) instead of guessing or asserting from "
    "memory. If the user says you are wrong about a past action, check "
    "before responding."
)

# Client tools such as sandbox commands can intentionally run for up to 120s.
CLIENT_TOOL_TIMEOUT_S = 135

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
                 min_speech_s=0.25, early_silence_s=0.15, pause_silence_s=0.35,
                 end_silence_s=2.4, max_utterance_s=45.0):
        self.model = model
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.min_speech_frames = int(min_speech_s * MIC_SR / VAD_FRAME)
        self.early_frames = int(early_silence_s * MIC_SR / VAD_FRAME)
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
        self.processed_samples = 0
        self.frame_observations = []

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
        self.frame_observations = []
        buf = np.concatenate([self.residual, samples])
        n_frames = len(buf) // VAD_FRAME
        self.residual = buf[n_frames * VAD_FRAME:]

        for i in range(n_frames):
            frame = buf[i * VAD_FRAME:(i + 1) * VAD_FRAME]
            prob = self.model(torch.from_numpy(frame), MIC_SR).item()
            self.processed_samples += VAD_FRAME
            self.frame_observations.append([self.processed_samples - VAD_FRAME, prob])

            if not self.in_speech:
                self.pre_roll.append(frame)
                if prob >= self.start_prob:
                    self.in_speech = True
                    self.utterance = list(self.pre_roll)
                    self.silence_run = 0
                    self.speech_frames = 0
                    self.pause_emitted = False
                    events.append(("start", None, self.processed_samples))
                continue

            self.utterance.append(frame)
            if prob >= self.start_prob:
                self.speech_frames += 1
            if prob < self.end_prob:
                self.silence_run += 1
            else:
                if self.pause_emitted:
                    self.pause_emitted = False
                    events.append(("resume", None, self.processed_samples))
                self.silence_run = 0

            # head start for ASR: transcription can begin 200ms before the
            # pause event fires, so the transcript is ready at the pause
            if (self.silence_run == self.early_frames
                    and self.speech_frames >= self.min_speech_frames):
                events.append(("early", np.concatenate(self.utterance), self.processed_samples))

            if (not self.pause_emitted
                    and self.silence_run == self.pause_frames
                    and self.speech_frames >= self.min_speech_frames):
                self.pause_emitted = True
                events.append(("pause", np.concatenate(self.utterance), self.processed_samples))

            ended = self.silence_run >= self.end_silence_frames
            too_long = len(self.utterance) >= self.max_utterance_frames
            if ended or too_long:
                audio = np.concatenate(self.utterance)
                enough = self.speech_frames >= self.min_speech_frames
                self.force_reset()
                events.append(("end", audio if enough else None, self.processed_samples))
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
        self.early = None          # (gen, asr_task) started at the early event
        self.gate_gen = 0          # bumped on start/resume; stale evals discard
        self.pause_transcript = None  # (gen, text) reusable at hard end
        self.client_tools = []        # tool schemas registered by the client
        self.client_tool_names = set()
        self.client_instructions = None
        self.pending_tools = {}       # call_id -> Future awaiting client result
        self.log = logging.getLogger(f"session-{session_id}")
        self.connection_id = uuid.uuid4().hex
        self.speech_id = None
        self.input_samples = 0
        self.recorder = None
        self.recording_error_sent = False
        self.background_tasks = set()

    def record(self, kind, **data):
        if self.recorder:
            return self.recorder.event(kind, connection_id=self.connection_id, **data)

    def record_audio(self, track, audio, rate, **data):
        if self.recorder:
            return self.recorder.audio(track, audio.astype("<f4", copy=False).tobytes(),
                                       rate, connection_id=self.connection_id, **data)

    async def set_recording(self, enabled, client=None):
        if self.recorder and (not enabled or self.recorder.error):
            recorder, self.recorder = self.recorder, None
            await asyncio.to_thread(recorder.close)
        if enabled and not self.recorder:
            self.recorder = Recording(self.server.recordings_dir, {
                "connection_id": self.connection_id, "client": client or {},
                "llm_model": self.server.llm_model, "asr_model": self.server.whisper_model,
                "classifier_model": self.server.classifier_model, "tts_model": "Kokoro-82M",
                "voice": self.voice, "mic_sample_rate": MIC_SR, "tts_sample_rate": TTS_SR,
                "mic_format": "browser-processed mono float32; includes silence",
                "input_sample_origin": self.input_samples,
                "runtime": {"python": platform.python_version(),
                            **{name: importlib.metadata.version(name) for name in
                               ("numpy", "fastapi", "uvicorn")}},
                "vad": {k: v for k, v in vars(self.gate).items()
                        if isinstance(v, (int, float, bool))},
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                  for name in ("main.py", "tools.py", "recording.py")},
                "capture_scope": "from recording start; earlier context may appear in model requests",
            })
            self.recording_error_sent = False
            self.record("session.context", history=self.history, client_tools=self.client_tools,
                        client_instructions=self.client_instructions, gate_gen=self.gate_gen)
        await self.send({"type": "recording_status", "enabled": bool(self.recorder),
                         "recording_id": self.recorder.id if self.recorder else None})

    async def check_recording(self):
        if self.recorder and self.recorder.error and not self.recording_error_sent:
            self.recording_error_sent = True
            await self.ws.send_json({"type": "recording_status", "enabled": False,
                                    "recording_id": self.recorder.id,
                                    "error": self.recorder.error})

    async def send(self, msg: dict):
        await self.ws.send_json(msg)
        if msg.get("type") == "audio_chunk" and msg.get("pcm"):
            self.record("server.message", message={k: v for k, v in msg.items() if k != "pcm"})
        else:
            self.record("server.message", message=msg)
        await self.check_recording()

    def cancel_response(self) -> bool:
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            return True
        return False

    async def on_message(self, msg: dict):
        received_ns = time.monotonic_ns()
        mtype = msg.get("type")
        await self.check_recording()
        if mtype == "set_recording":
            await self.set_recording(msg.get("enabled") is True, msg.get("client"))
            return
        if mtype == "client_event":
            # Client observations/labels are claims, not model truth or instructions.
            if msg.get("event") == "feedback" and msg.get("label") not in {"accepted", "incorrect", "revised", "unsure"}:
                await self.send({"type": "error", "data": "invalid feedback label"})
                return
            saved = None
            if self.recorder and msg.get("recording_id") == self.recorder.id:
                saved = self.record("client." + str(msg.get("event", "observation")), payload=msg)
            if msg.get("event") == "feedback":
                if saved and not await asyncio.to_thread(self.recorder.flush):
                    saved = None
                await self.send({"type": "feedback_saved", "event_id": saved["event_id"] if saved else None,
                                 "client_event_id": msg.get("client_event_id")})
            elif msg.get("event") == "clock" and saved:
                await self.send({"type": "clock_sync", "client_event_id": msg.get("client_event_id"),
                                 "client_sent_ms": msg.get("monotonic_ms"),
                                 "server_received_ns": received_ns, "server_sent_ns": time.monotonic_ns()})
            return
        if mtype == "audio":
            pcm = np.frombuffer(base64.b64decode(msg["data"]), dtype="<f4")
            self.record_audio("mic", pcm, MIC_SR, input_start_sample=self.input_samples,
                              client={k: v for k, v in msg.items() if k not in {"data", "type"}})
            self.input_samples += len(pcm)
            await self.on_audio(pcm)
            return
        self.record("client.message", message=msg,
                    late_tool_result=(mtype == "tool_result" and msg.get("call_id") not in self.pending_tools))
        if mtype == "text":
            self.cancel_response()
            self.response_task = asyncio.create_task(
                self.respond(str(msg["data"]), time.monotonic()))
        elif mtype == "interrupt":
            if self.cancel_response():
                self.log.info("response interrupted by client")
            await self.send({"type": "interrupted", "utt": self.utt})
        elif mtype == "register_tools":
            tools = [t for t in (msg.get("tools") or [])
                     if t.get("type") == "function"
                     and t.get("function", {}).get("name")]
            self.client_tools = tools
            self.client_tool_names = {t["function"]["name"] for t in tools}
            instructions = msg.get("instructions")
            self.client_instructions = str(instructions) if instructions else None
            self.log.info(f"client registered tools: {sorted(self.client_tool_names)}")
            await self.send({"type": "tools_registered",
                             "names": sorted(self.client_tool_names)})
        elif mtype == "tool_result":
            fut = self.pending_tools.pop(msg.get("call_id"), None)
            if fut and not fut.done():
                client_ms = msg.get("client_duration_ms")
                try:
                    client_ms = max(0.0, float(client_ms))
                except (TypeError, ValueError):
                    client_ms = None
                fut.set_result({
                    "result": str(msg.get("result", "")),
                    "client_duration_ms": client_ms,
                })
        elif mtype == "set_voice":
            voice = str(msg.get("data", ""))
            if voice in VOICES:
                self.voice = voice
                await self.send({"type": "voice_changed", "data": voice})
            else:
                await self.send({"type": "error", "data": f"unknown voice: {voice}"})
        else:
            await self.send({"type": "error", "data": f"unknown message type: {mtype}"})

    def trimmed_history(self):
        """Full history until it nears the context budget; then drop oldest
        messages (never starting mid tool exchange — orphaned tool results
        break chat templates). Returns (messages, dropped_count) so trimming
        is surfaced, not silent."""
        def cost(m):  # rough tokens: chars/4 plus per-message overhead
            return (len(str(m.get("content") or ""))
                    + len(str(m.get("tool_calls") or ""))) // 4 + 8

        h = list(self.history)
        total = sum(map(cost, h))
        dropped = 0
        while len(h) > 2 and total > self.server.history_budget:
            total -= cost(h.pop(0))
            dropped += 1
        while h and (h[0]["role"] == "tool"
                     or (h[0]["role"] == "assistant" and h[0].get("tool_calls"))):
            total -= cost(h.pop(0))
            dropped += 1
        return h, dropped

    def cancel_eval(self):
        if self.eval_task and not self.eval_task.done():
            self.eval_task.cancel()
        if self.early and not self.early[1].done():
            self.early[1].cancel()
        self.early = None

    async def transcribe_async(self, audio: np.ndarray, source="final", end_sample=None) -> str:
        started = time.monotonic()
        asr_id = uuid.uuid4().hex
        speech_id, gen = self.speech_id, self.gate_gen
        audio_ref = self.record_audio("asr_input", audio, MIC_SR, asr_id=asr_id,
                                     input_end_sample=end_sample, source=source)
        self.record("asr.started", asr_id=asr_id, speech_id=speech_id, gate_gen=gen,
                    source=source, audio_ref=audio_ref)
        loop = asyncio.get_running_loop()
        metric = {
            "session": self.id,
            "audio_ms": round(len(audio) / MIC_SR * 1000),
        }
        recorder = self.recorder
        def transcribe_recorded():
            # Keep outputs from inference that finishes after its consumer cancels.
            out = self.server.transcribe_raw(audio)
            if recorder:
                recorder.event("asr.inference_result", asr_id=asr_id, output=out,
                               connection_id=self.connection_id,
                               speech_id=speech_id, gate_gen=gen)
            return out["text"].strip()
        try:
            text = await loop.run_in_executor(
                self.server.asr_executor, transcribe_recorded)
            metric["status"] = "completed"
            self.record("asr.completed", asr_id=asr_id, speech_id=speech_id,
                        gate_gen=gen, source=source, text=text)
            return text
        except asyncio.CancelledError:
            self.record("asr.cancelled", asr_id=asr_id, speech_id=speech_id,
                        reason="consumer cancelled; inference thread may finish")
            raise
        except Exception as exc:
            metric["status"] = "error"
            metric["error"] = f"{exc.__class__.__name__}: {exc}"
            self.record("asr.error", asr_id=asr_id, error=metric["error"])
            raise
        finally:
            metric["duration_ms"] = round((time.monotonic() - started) * 1000)
            self.log.info("asr_metric %s", json.dumps(metric, separators=(",", ":")))

    async def on_audio(self, pcm: np.ndarray):
        events = self.gate.feed(pcm)
        self.record("vad.frames", frame_samples=VAD_FRAME, sample_rate=MIC_SR,
                    frames=self.gate.frame_observations)
        for event, audio, end_sample in events:
            if event == "start":
                self.speech_id = uuid.uuid4().hex
            self.record("vad.event", event=event, speech_id=self.speech_id,
                        input_end_sample=end_sample,
                        input_start_sample=end_sample - len(audio) if audio is not None else None,
                        gate_gen=self.gate_gen)
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
            elif event == "early":
                self.early = (self.gate_gen,
                              asyncio.create_task(self.transcribe_async(audio, "early", end_sample)))
            elif event == "pause":
                await self.send({"type": "vad", "data": "pause"})
                self.eval_task = asyncio.create_task(
                    self.evaluate_pause(audio, self.gate_gen))
            elif event == "end":
                self.cancel_eval()
                await self.send({"type": "vad", "data": "speech_end"})
                if audio is not None:
                    task = asyncio.create_task(self.finalize_turn(audio))
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)

    async def evaluate_pause(self, audio: np.ndarray, gen: int):
        """Short pause: judge whether the turn is complete, with two overlaps:
        the transcript usually comes from the early-ASR head start, and while
        the LLM verdict is pending the response is generated speculatively
        (held, not sent) so a DONE verdict costs no extra latency."""
        t0 = time.monotonic()
        early = self.early
        try:
            if early and early[0] == gen and not early[1].cancelled():
                text = await early[1]
            else:
                text = await self.transcribe_async(audio, "pause")
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.log.exception("pause transcription failed")
            await self.send({"type": "error", "data":
                             f"transcription failed: {exc}"})
            return
        self.early = None
        if gen != self.gate_gen:
            return  # user resumed while we transcribed
        norm = text.lower().strip(" .,!?")
        if len(norm) < 2 or norm in ASR_HALLUCINATIONS:
            return  # noise; let the hard end handle it
        verdict = turn_heuristic(text)
        source = "heuristic"
        spec = hold = None
        if verdict is None:
            hold = asyncio.Event()
            self.cancel_response()
            spec = asyncio.create_task(self.respond(text, t0, audio, hold=hold))
            self.response_task = spec
            try:
                verdict = await self.classify_turn(text)
            except asyncio.CancelledError:
                spec.cancel()
                raise
            source = "llm+spec"
        if gen != self.gate_gen:
            if spec:
                spec.cancel()
            return
        ms = (time.monotonic() - t0) * 1000
        self.log.info(f"endpoint {verdict} ({source}, {ms:.0f}ms): {text!r}")
        await self.send({"type": "endpoint", "data": verdict,
                         "source": source, "transcript": text,
                         "speech_id": self.speech_id, "gate_gen": gen})
        if verdict == "done":
            self.gate.force_reset()
            self.gate_gen += 1
            self.pause_transcript = None
            self.log.info(f"turn: {text!r}")
            await self.send({"type": "transcription", "data": text, "speech_id": self.speech_id})
            if spec:
                hold.set()
            else:
                self.cancel_response()
                self.response_task = asyncio.create_task(
                    self.respond(text, t0, audio))
        else:
            if spec:
                spec.cancel()
            self.pause_transcript = (gen, text)

    async def classify_turn(self, text: str) -> str:
        classifier_id = uuid.uuid4().hex
        self.record("endpoint.classifier_started", classifier_id=classifier_id,
                    model=self.server.classifier_model, prompt=TURN_CLASSIFIER_PROMPT,
                    text=text, speech_id=self.speech_id, gate_gen=self.gate_gen,
                    max_tokens=3, temperature=0.0)
        if not self.server.classifier_model:
            self.record("endpoint.classifier_result", classifier_id=classifier_id,
                        verdict="done", reason="classifier unavailable")
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
            self.record("endpoint.classifier_result", classifier_id=classifier_id, output=out)
            return "wait" if "wait" in out else "done"
        except asyncio.CancelledError:
            self.record("endpoint.classifier_cancelled", classifier_id=classifier_id)
            raise
        except Exception as e:
            self.record("endpoint.classifier_error", classifier_id=classifier_id,
                        error=str(e), fallback="done")
            self.log.warning(f"turn classifier failed ({e}); assuming done")
            return "done"

    async def finalize_turn(self, audio: np.ndarray):
        """Hard end (long silence / length cap): respond even if mid-thought."""
        t0 = time.monotonic()
        if self.pause_transcript and self.pause_transcript[0] == self.gate_gen:
            text = self.pause_transcript[1]
        else:
            try:
                text = await self.transcribe_async(audio)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self.log.exception("final transcription failed")
                await self.send({"type": "error", "data":
                                 f"transcription failed: {exc}"})
                return
        self.pause_transcript = None
        norm = text.lower().strip(" .,!?")
        if len(norm) < 2 or norm in ASR_HALLUCINATIONS:
            self.log.info(f"skipping likely ASR hallucination: {text!r}")
            return
        await self.start_turn(text, t0, audio)

    async def start_turn(self, text: str, t0: float, audio: np.ndarray = None):
        self.pause_transcript = None
        self.log.info(f"turn: {text!r}")
        await self.send({"type": "transcription", "data": text, "speech_id": self.speech_id})
        self.cancel_response()
        self.response_task = asyncio.create_task(self.respond(text, t0, audio))

    async def respond(self, user_text: str, t0: float, audio: np.ndarray = None,
                      hold: asyncio.Event = None):
        """Generate and stream a response. With `hold`, generation runs
        speculatively: nothing is sent to the client or committed to history
        until the event is set (cancellation before that leaves no trace)."""
        self.utt += 1
        utt = self.utt
        run_id = uuid.uuid4().hex
        model_audio_ref = (self.record_audio("model_input", audio, MIC_SR, run_id=run_id,
                                             role="source utterance",
                                             native_audio=self.server.audio_input,
                                             transform="WAV PCM16 via soundfile if native_audio=true")
                           if audio is not None else None)
        self.record("response.started", run_id=run_id, utt=utt, text=user_text,
                    speculative=hold is not None, speech_id=self.speech_id,
                    gate_gen=self.gate_gen, input_audio_ref=model_audio_ref,
                    input_source="speech" if audio is not None else "typed_text")
        user_msg = {"role": "user", "content": user_text}
        released = hold is None
        if released:
            self.history.append(user_msg)
            self.record("response.released", run_id=run_id, utt=utt, reason="no hold")

        async def release():
            nonlocal released
            if not released:
                await hold.wait()
                self.history.append(user_msg)
                released = True
                self.record("response.released", run_id=run_id, utt=utt, reason="endpoint done")

        spoken = []
        stream = None
        first_token_ms = first_action_ms = first_audio_ms = None
        metric = {
            "session": self.id,
            "utt": utt,
            "status": "running",
            "audio_input": bool(audio is not None and self.server.audio_input),
            "history_messages": 0,
            "input_ms": 0,
            "first_action_ms": None,
            "first_token_ms": None,
            "first_audio_ms": None,
            "model_ms": 0,
            "tool_ms": 0,
            "tts_ms": 0,
            "total_ms": 0,
            "model_rounds": [],
            "tools": [],
        }

        async def publish_metric(status):
            metric.update({
                "status": status,
                "first_action_ms": round(first_action_ms) if first_action_ms else None,
                "first_token_ms": round(first_token_ms) if first_token_ms else None,
                "first_audio_ms": round(first_audio_ms) if first_audio_ms else None,
                "model_ms": round(metric["model_ms"]),
                "tool_ms": round(metric["tool_ms"]),
                "tts_ms": round(metric["tts_ms"]),
                "total_ms": round((time.monotonic() - t0) * 1000),
            })
            self.server.record_turn_metric(metric)
            self.record("response.finished", run_id=run_id, utt=utt, released=released,
                        metric=metric)
            self.log.info("turn_metric %s", json.dumps(metric, separators=(",", ":")))
            try:
                await self.send({"type": "turn_metrics", **metric})
            except Exception:
                pass

        try:
            system = SYSTEM_PROMPT
            if self.client_instructions:
                system += "\n\n" + self.client_instructions
            hist, dropped = self.trimmed_history()
            if dropped:
                system += (f"\n\nNote: this conversation is long — the earliest "
                           f"{dropped} messages are no longer visible to you. "
                           f"Anything you don't see, verify with tools.")
                self.log.warning(f"history trimmed: dropped {dropped} of "
                                 f"{len(self.history)} messages")
                await self.send({"type": "history_trimmed", "dropped": dropped})
            messages = ([{"role": "system", "content": system}] + hist
                        + ([user_msg] if hold is not None else []))
            metric["history_messages"] = len(hist)
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

            kwargs["tools"] = TOOLS + self.client_tools

            # Audio-native models hear the actual utterance instead of the
            # transcript (current turn only; history stays text).
            work = list(messages)
            if audio is not None and self.server.audio_input:
                work[-1] = {
                    "role": "user",
                    "content": [{"type": "input_audio", "input_audio": {
                        "data": audio_to_wav_b64(audio), "format": "wav"}}],
                }

            buf = ""
            full = ""

            async def flush_speak(text):
                nonlocal first_audio_ms
                await release()
                started = time.monotonic()
                did_speak = await self.speak(text, utt, len(spoken))
                metric["tts_ms"] += (time.monotonic() - started) * 1000
                if did_speak:
                    if first_audio_ms is None:
                        first_audio_ms = (time.monotonic() - t0) * 1000
                    spoken.append(text)

            for round_i in range(5):
                round_started = time.monotonic()
                if round_i == 0:
                    metric["input_ms"] = round((round_started - t0) * 1000)
                round_tts_started = metric["tts_ms"]
                def record_request(fallback=False):
                    # Replace embedded audio with an exact source + documented transform.
                    snapshot = [{**m, "content": [{"type": "input_audio", "source": model_audio_ref}]}
                                if isinstance(m.get("content"), list) and any(
                                    p.get("type") == "input_audio" for p in m["content"])
                                else m for m in work]
                    self.record("model.request", run_id=run_id, utt=utt, round=round_i,
                                messages=snapshot, parameters=kwargs, transcript_fallback=fallback)
                record_request()
                try:
                    stream = await self.server.llm.chat.completions.create(
                        messages=work, **kwargs)
                except Exception as e:
                    if round_i == 0 and audio is not None and self.server.audio_input:
                        self.log.warning(
                            f"audio input rejected ({e.__class__.__name__}); "
                            "falling back to transcript")
                        work = list(messages)
                        self.record("model.audio_rejected", run_id=run_id, error=str(e))
                        record_request(fallback=True)
                        stream = await self.server.llm.chat.completions.create(
                            messages=work, **kwargs)
                    else:
                        raise

                headers_ms = (time.monotonic() - round_started) * 1000
                round_first_action_ms = None
                tool_calls = {}
                round_content = ""
                async for chunk in stream:
                    self.record("model.delta", run_id=run_id, utt=utt, round=round_i,
                                chunk=chunk.model_dump(mode="json", exclude_none=True))
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    has_action = bool(delta and (delta.content or delta.tool_calls))
                    if has_action and round_first_action_ms is None:
                        now = time.monotonic()
                        round_first_action_ms = (now - round_started) * 1000
                        if first_action_ms is None:
                            first_action_ms = (now - t0) * 1000
                    if delta and delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index if tc.index is not None else 0
                            ent = tool_calls.setdefault(
                                idx, {"id": None, "name": "", "args": ""})
                            if tc.id:
                                ent["id"] = tc.id
                            if tc.function and tc.function.name:
                                ent["name"] = tc.function.name
                            if tc.function and tc.function.arguments:
                                ent["args"] += tc.function.arguments
                    content = delta.content if delta else None
                    if not content:
                        continue
                    if first_token_ms is None:
                        first_token_ms = (time.monotonic() - t0) * 1000
                    buf += content
                    full += content
                    round_content += content
                    ready, buf = split_sentences(buf)
                    for sentence in ready:
                        await flush_speak(sentence)

                round_ms = (time.monotonic() - round_started) * 1000
                # TTS is intentionally generated while the model stream is
                # open. Keep it out of model_ms so regressions are attributable.
                round_model_ms = max(
                    0, round_ms - (metric["tts_ms"] - round_tts_started))
                metric["model_ms"] += round_model_ms
                metric["model_rounds"].append({
                    "round": round_i + 1,
                    "headers_ms": round(headers_ms),
                    "first_action_ms": (round(round_first_action_ms)
                                        if round_first_action_ms else None),
                    "model_ms": round(round_model_ms),
                    "wall_ms": round(round_ms),
                    "tool_names": [tool_calls[i]["name"]
                                   for i in sorted(tool_calls)],
                })

                if not tool_calls:
                    break

                calls = [tool_calls[i] for i in sorted(tool_calls)]
                for i, c in enumerate(calls):
                    c["id"] = c["id"] or f"call_{utt}_{round_i}_{i}"
                    self.record("tool.proposed", run_id=run_id, utt=utt, round=round_i,
                                call=c, held=not released)
                await release()  # tools have visible effects; never run held
                # Speak any buffered pre-tool phrase ("Let me check…") so the
                # user hears something while tools run.
                if buf.strip():
                    await flush_speak(buf.strip())
                    buf = ""
                assistant_msg = {
                    "role": "assistant",
                    "content": round_content or "",
                    "tool_calls": [{
                        "id": c["id"], "type": "function",
                        "function": {"name": c["name"],
                                     "arguments": c["args"] or "{}"},
                    } for c in calls],
                }
                work.append(assistant_msg)
                # Tool exchanges go into history too, or the model forgets
                # what it did (and invents ids / claims phantom actions).
                self.history.append(assistant_msg)
                for c in calls:
                    self.log.info(f"tool call: {c['name']}({c['args'][:120]})")
                    tool_started = time.monotonic()
                    client_ms = None
                    self.record("tool.started", run_id=run_id, utt=utt, call=c)
                    try:
                        if c["name"] in self.client_tool_names:
                            result, client_ms = await self.call_client_tool(c, utt)
                        else:
                            await self.send({"type": "tool_call", "utt": utt,
                                             "action_id": c["id"],
                                             "name": c["name"], "args": c["args"][:200]})
                            result = await run_tool(c["name"], c["args"])
                    except asyncio.CancelledError:
                        self.record("tool.cancelled", run_id=run_id, call_id=c["id"],
                                    effect_status="unknown; client or process may still run")
                        raise
                    except Exception as exc:
                        self.record("tool.error", run_id=run_id, call_id=c["id"], error=str(exc))
                        raise
                    self.record("tool.completed", run_id=run_id, utt=utt, call_id=c["id"],
                                result=result, client_duration_ms=client_ms,
                                correctness="unlabeled")
                    tool_ms = (time.monotonic() - tool_started) * 1000
                    metric["tool_ms"] += tool_ms
                    tool_metric = {"name": c["name"], "duration_ms": round(tool_ms)}
                    if client_ms is not None:
                        tool_metric["client_ms"] = round(client_ms)
                    metric["tools"].append(tool_metric)
                    self.log.info("tool_metric %s", json.dumps({
                        "session": self.id, "utt": utt, **tool_metric,
                    }, separators=(",", ":")))
                    await self.send({"type": "tool_result", "utt": utt,
                                     "action_id": c["id"],
                                     "name": c["name"],
                                     "preview": result[:200]})
                    work.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": result})
                    self.history.append({"role": "tool", "tool_call_id": c["id"],
                                         "content": result[:2000]})

            if buf.strip():
                await flush_speak(buf.strip())
            await release()

            full = full.strip()
            self.history.append({"role": "assistant", "content": full or "(no reply)"})
            await self.send({"type": "audio_chunk", "utt": utt, "seq": len(spoken),
                             "pcm": "", "final": True})
            await self.send({
                "type": "chat_done", "utt": utt, "data": full,
                "t_first_token_ms": round(first_token_ms or 0),
                "t_first_audio_ms": round(first_audio_ms or 0),
            })
            await publish_metric("completed")
            self.log.info(
                f"utt {utt}: first token {first_token_ms or 0:.0f}ms, "
                f"first audio {first_audio_ms or 0:.0f}ms after speech end")
        except asyncio.CancelledError:
            if spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken)})
            await publish_metric("cancelled")
            raise
        except Exception as e:
            self.log.exception("response failed")
            metric["error"] = f"{e.__class__.__name__}: {e}"
            await publish_metric("error")
            await self.send({"type": "error", "data": f"response failed: {e}"})
        finally:
            if stream is not None:
                await stream.close()

    async def call_client_tool(self, call: dict, utt: int) -> tuple[str, float | None]:
        """Forward a tool call to the client and await its result."""
        fut = asyncio.get_running_loop().create_future()
        self.pending_tools[call["id"]] = fut
        try:
            await self.send({"type": "tool_call", "utt": utt,
                             "call_id": call["id"], "name": call["name"],
                             "args": call["args"] or "{}"})
            payload = await asyncio.wait_for(fut, timeout=CLIENT_TOOL_TIMEOUT_S)
            if isinstance(payload, dict):
                return payload["result"], payload.get("client_duration_ms")
            return str(payload), None
        except asyncio.TimeoutError:
            return "error: the application did not respond to the tool call", None
        finally:
            self.pending_tools.pop(call["id"], None)

    async def speak(self, text: str, utt: int, seq: int) -> bool:
        text = tts_clean(text)
        if not text:
            return False
        loop = asyncio.get_running_loop()
        recorder, voice = self.recorder, self.voice
        self.record("tts.started", utt=utt, seq=seq, text=text, voice=voice)
        def synthesize_recorded():
            pcm = self.server.synthesize(text, voice)
            if recorder and pcm is not None:
                recorder.audio("tts", pcm.astype("<f4", copy=False).tobytes(), TTS_SR,
                               utt=utt, seq=seq, text=text, voice=voice, stage="generated",
                               connection_id=self.connection_id)
            return pcm
        try:
            pcm = await loop.run_in_executor(self.server.tts_executor, synthesize_recorded)
        except asyncio.CancelledError:
            self.record("tts.cancelled", utt=utt, seq=seq)
            raise
        except Exception as exc:
            self.record("tts.error", utt=utt, seq=seq, error=str(exc))
            raise
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
                 reasoning_effort: str = None, audio_input: bool = False,
                 recordings_dir: str = "data/recordings"):
        self.recordings_dir = str(Path(recordings_dir).expanduser().resolve())
        self.llm = AsyncOpenAI(base_url=llm_url, api_key=llm_api_key or "none")
        self.llm_model = llm_model
        self.reasoning_effort = reasoning_effort
        self.audio_input = audio_input
        self.classifier_llm = (AsyncOpenAI(base_url=classifier_url, api_key="none")
                               if classifier_url else self.llm)
        self.classifier_model = None
        self.history_budget = 80000
        self.whisper_model = whisper_model
        self.asr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self.session_counter = 0
        self.turn_metrics = deque(maxlen=200)
        self.load_models()
        self.app = self.build_app()

    def record_turn_metric(self, metric: dict):
        # Do not retain transcripts or tool arguments: timing data should be
        # useful without turning the metrics endpoint into conversation logs.
        self.turn_metrics.append(json.loads(json.dumps(metric)))

    def metrics_snapshot(self) -> dict:
        recent = list(self.turn_metrics)
        completed = [m for m in recent if m.get("status") == "completed"]
        fields = ("input_ms", "first_action_ms", "first_audio_ms", "model_ms",
                  "tool_ms", "tts_ms", "total_ms")
        summary = {}
        for field in fields:
            values = [m[field] for m in completed
                      if isinstance(m.get(field), (int, float))]
            if values:
                summary[field] = {
                    "p50": round(float(np.percentile(values, 50))),
                    "p95": round(float(np.percentile(values, 95))),
                    "max": round(max(values)),
                }
        statuses = {}
        for metric in recent:
            status = metric.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        return {
            "model": self.llm_model,
            "reasoning_effort": self.reasoning_effort,
            "audio_input": self.audio_input,
            "retained": len(recent),
            "statuses": statuses,
            "summary_ms": summary,
            "recent": recent[-20:],
        }

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

        logger.info("warming up Whisper...")
        try:
            self.transcribe(np.zeros(MIC_SR, dtype=np.float32))
        except Exception as e:
            logger.warning(f"whisper warmup failed: {e}")

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
        return self.transcribe_raw(audio)["text"].strip()

    def transcribe_raw(self, audio: np.ndarray) -> dict:
        return self.asr(
            {"raw": audio, "sampling_rate": MIC_SR},
            generate_kwargs={"language": "english"},
            return_timestamps=True,
        )

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

        @app.get("/metrics")
        async def metrics():
            return self.metrics_snapshot()

        @app.get("/recording-client.js")
        async def recording_client():
            return FileResponse(Path(__file__).parent / "static/recording-client.js",
                                headers={"Cache-Control": "no-store"})

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
                "recording_available": True,
            })
            try:
                while True:
                    msg = await ws.receive_json()
                    await session.on_message(msg)
            except WebSocketDisconnect:
                logger.info(f"session {session.id} disconnected")
            finally:
                tasks = [t for t in [session.response_task, session.eval_task,
                                     session.early[1] if session.early else None,
                                     *session.background_tasks] if t]
                session.cancel_response()
                session.cancel_eval()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if session.recorder:
                    await asyncio.to_thread(session.recorder.close, "disconnect")

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
                   choices=["none", "minimal", "low", "medium", "high"],
                   help="pass reasoning_effort to the LLM (reasoning models only)")
    p.add_argument("--audio-input", action="store_true",
                   help="send user turns as audio to an audio-native LLM "
                        "(falls back to the transcript if rejected)")
    p.add_argument("--classifier-url", default="http://127.0.0.1:8001/v1",
                   help="fast local LLM for end-of-turn classification "
                        "(heuristics-only if unreachable)")
    p.add_argument("--whisper", default="openai/whisper-large-v3-turbo")
    p.add_argument("--recordings-dir", default="data/recordings",
                   help="local training-data directory; recording is controlled by the client")
    p.add_argument("--history-budget", type=int, default=80000,
                   help="approx token budget for chat history before oldest "
                        "messages are dropped (drop is announced to the model "
                        "and the client). Use ~5000 for an 8k-context LLM.")
    args = p.parse_args()

    api_key = os.environ.get(args.llm_api_key_env) if args.llm_api_key_env else None
    if args.llm_api_key_env and not api_key:
        p.error(f"--llm-api-key-env: ${args.llm_api_key_env} is not set")
    classifier_url = None if args.classifier_url == args.llm_url else args.classifier_url

    server = SpeechServer(
        llm_url=args.llm_url, whisper_model=args.whisper,
        llm_model=args.llm_model, llm_api_key=api_key,
        classifier_url=classifier_url, reasoning_effort=args.reasoning_effort,
        audio_input=args.audio_input, recordings_dir=args.recordings_dir)
    server.history_budget = args.history_budget
    uvicorn.run(server.app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
