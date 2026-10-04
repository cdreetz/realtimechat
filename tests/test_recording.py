import asyncio
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from recording import Recording, validate_recording


def events(recording):
    return [json.loads(line) for line in (recording.path / "events.jsonl").read_text().splitlines()]


class WriterTests(unittest.TestCase):
    def test_lossless_offsets_permissions_and_corruption(self):
        with tempfile.TemporaryDirectory() as root:
            rec = Recording(root, {"test": True})
            pcm = struct.pack("<fff", -0.25, 0.0, 0.875)
            first = rec.audio("mic", pcm, 16000, capture_start_sample=100)
            second = rec.audio("mic", pcm, 16000, capture_start_sample=103)
            rec.close()
            self.assertEqual(first["audio"]["start_sample"], 0)
            self.assertEqual(second["audio"]["start_sample"], 3)
            self.assertEqual((rec.path / "mic.f32").read_bytes(), pcm * 2)
            self.assertTrue(validate_recording(rec.path)["valid"])
            self.assertEqual((rec.path.stat().st_mode & 0o777), 0o700)
            self.assertEqual(((rec.path / "events.jsonl").stat().st_mode & 0o777), 0o600)
            (rec.path / "mic.f32").write_bytes(b"\0" * 24)
            self.assertFalse(validate_recording(rec.path)["valid"])

    def test_overflow_stops_and_marks_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            rec = Recording(root, {}, max_queue_bytes=1)
            self.assertIsNone(rec.event("must_not_claim_saved"))
            rec.close()
            self.assertIn("queue", rec.error)
            self.assertFalse(json.loads((rec.path / "manifest.json").read_text())["complete"])

    def test_size_limit_and_disk_failure(self):
        with tempfile.TemporaryDirectory() as root:
            rec = Recording(root, {}, max_bytes=1)
            rec.close()
            self.assertIn("size limit", rec.error)
            rec2 = Recording(root, {}, min_free_bytes=10**30)
            rec2.close()
            self.assertIn("disk space", rec2.error)
            self.assertFalse(json.loads((rec2.path / "manifest.json").read_text())["complete"])

    def test_missing_event_log_is_not_a_valid_training_example(self):
        with tempfile.TemporaryDirectory() as root:
            rec = Recording(root, {})
            rec.close()
            (rec.path / "events.jsonl").write_text("")
            self.assertFalse(validate_recording(rec.path)["valid"])


# The app runs Whisper/Kokoro on a GPU. These tests substitute only the inference
# boundary and transport, exercising the actual Session/VAD/recording code.
import numpy as np
fake_torch = types.ModuleType("torch")
fake_torch.from_numpy = lambda x: x
fake_openai = types.ModuleType("openai")
fake_openai.AsyncOpenAI = object
spec = importlib.util.spec_from_file_location("recording_test_server", Path(__file__).resolve().parents[1] / "server/main.py")
server_module = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"torch": fake_torch, "openai": fake_openai}):
    spec.loader.exec_module(server_module)


class Chunk:
    def __init__(self, tool=False):
        tc = types.SimpleNamespace(index=0, id="test-call", function=types.SimpleNamespace(name="test_tool", arguments='{"value":7}'))
        self.choices = [types.SimpleNamespace(delta=types.SimpleNamespace(content=None if tool else "Done.", tool_calls=[tc] if tool else None))]
        self.tool = tool

    def model_dump(self, **kwargs):
        return {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "test-call", "function": {"name": "test_tool", "arguments": '{"value":7}'}}]}
                             if self.tool else {"content": "Done."}}]}


class Stream:
    def __init__(self, tool=False):
        self.chunk = Chunk(tool)
        self.consumed = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield self.chunk
        self.consumed.set()

    async def close(self):
        self.closed = True


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sent = []
        self.streams = []
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.auto_tool_result = True
        async def create(**kwargs):
            stream = Stream(tool=not self.streams)
            self.streams.append(stream)
            return stream
        async def send_json(msg):
            self.sent.append(msg)
            if msg.get("call_id") and self.auto_tool_result:
                await self.session.on_message({"type": "tool_result", "call_id": msg["call_id"], "result": "worked", "client_duration_ms": 2})
        class VAD:
            def __call__(self, frame, rate): return np.float32(0)
            def reset_states(self): pass
        self.server = types.SimpleNamespace(recordings_dir=self.tmp.name, llm_model="test-model", whisper_model="test-asr",
            classifier_model=None, load_vad=VAD, history_budget=80000, reasoning_effort=None, audio_input=False,
            wants_no_thinking=lambda _: False, llm=types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))),
            asr_executor=self.pool, tts_executor=self.pool, record_turn_metric=lambda _: None,
            synthesize=lambda *_: np.array([.1, -.2], dtype=np.float32),
            transcribe_raw=lambda _: {"text": "open a code editor", "chunks": [{"timestamp": (0, .1), "text": "open"}]})
        self.session = server_module.Session(self.server, types.SimpleNamespace(send_json=send_json), 1)
        self.session.client_tool_names.add("test_tool")
        await self.session.set_recording(True, {"test": True})
        self.rec = self.session.recorder

    async def asyncTearDown(self):
        self.pool.shutdown(wait=True)
        if self.session.recorder:
            self.session.recorder.close()
        self.tmp.cleanup()

    async def test_speculative_tool_is_recorded_but_never_executed(self):
        hold = asyncio.Event()
        task = asyncio.create_task(self.session.respond("open", asyncio.get_running_loop().time(), hold=hold))
        while not self.streams:
            await asyncio.sleep(0)
        await self.streams[0].consumed.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        self.rec.close()
        trace = events(self.rec)
        kinds = [e["kind"] for e in trace]
        self.assertIn("model.delta", kinds)
        self.assertIn("tool.proposed", kinds)
        self.assertNotIn("tool.started", kinds)
        self.assertNotIn("response.released", kinds)
        self.assertEqual(self.session.history, [])
        self.assertTrue(self.streams[0].closed)
        self.assertEqual(next(e["data"]["metric"]["status"] for e in trace if e["kind"] == "response.finished"), "cancelled")

    async def test_execution_full_results_audio_and_explicit_feedback(self):
        await self.session.respond("do it", asyncio.get_running_loop().time())
        await self.session.on_message({"type": "client_event", "event": "feedback", "recording_id": self.rec.id,
            "label": "incorrect", "target": {"kind": "tool", "call_id": "test-call"}, "client_event_id": "label-1"})
        self.rec.close()
        trace = events(self.rec)
        kinds = [e["kind"] for e in trace]
        self.assertLess(kinds.index("tool.proposed"), kinds.index("tool.started"))
        self.assertLess(kinds.index("tool.started"), kinds.index("tool.completed"))
        result = next(e["data"] for e in trace if e["kind"] == "tool.completed")
        self.assertEqual((result["result"], result["correctness"]), ("worked", "unlabeled"))
        self.assertEqual(sum(k == "client.feedback" for k in kinds), 1)
        self.assertTrue(validate_recording(self.rec.path)["valid"])
        self.assertEqual(validate_recording(self.rec.path)["audio_samples"]["tts.f32"], 2)

    async def test_pause_resume_sample_origins_and_late_results(self):
        import base64
        packet = {"type": "audio", "data": base64.b64encode(np.zeros(512, dtype="<f4").tobytes()).decode()}
        await self.session.on_message(packet)
        await self.session.set_recording(False)
        await self.session.on_message(packet)
        await self.session.set_recording(True)
        resumed = self.session.recorder
        await self.session.on_message(packet)
        await self.session.on_message({"type": "tool_result", "call_id": "already-cancelled", "result": "still finished"})
        await self.session.on_message({"type": "client_event", "event": "feedback", "recording_id": self.rec.id, "label": "accepted"})
        resumed.close()
        trace = events(resumed)
        mic = next(e["data"] for e in trace if e["kind"] == "audio.block")
        self.assertEqual(mic["input_start_sample"], 1024)
        self.assertEqual(mic["audio"]["start_sample"], 0)
        self.assertTrue(any(e["data"].get("late_tool_result") for e in trace))
        self.assertFalse(any(e["kind"] == "client.feedback" for e in trace))
        self.assertTrue(validate_recording(resumed.path)["valid"])

    async def test_asr_output_survives_consumer_cancellation(self):
        entered, release = threading.Event(), threading.Event()
        def transcribe(audio):
            entered.set()
            release.wait(5)
            return {"text": "Open a co", "chunks": [{"timestamp": [0, .1], "text": "Open"}]}
        self.server.transcribe_raw = transcribe
        task = asyncio.create_task(self.session.transcribe_async(np.zeros(512, dtype=np.float32), "early", 512))
        await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        release.set()
        await asyncio.to_thread(self.pool.shutdown, True)
        self.rec.close()
        trace = events(self.rec)
        self.assertIn("asr.cancelled", [e["kind"] for e in trace])
        result = next(e["data"] for e in trace if e["kind"] == "asr.inference_result")
        self.assertEqual(result["output"]["text"], "Open a co")

    async def test_vad_frame_offsets_span_packet_boundaries(self):
        gate = self.session.gate
        gate.feed(np.zeros(600, dtype=np.float32))
        self.assertEqual(gate.frame_observations[0][0], 0)
        gate.feed(np.zeros(500, dtype=np.float32))
        self.assertEqual(gate.frame_observations[0][0], 512)
        self.assertEqual(gate.processed_samples, 1024)
        gate.force_reset()
        gate.feed(np.zeros(500, dtype=np.float32))
        self.assertEqual(gate.frame_observations[0][0], 1024)


if __name__ == "__main__":
    unittest.main()
