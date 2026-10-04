"""Versioned, local training traces. No model or web framework dependencies.

The event loop only serializes/enqueues; one writer owns all disk operations.
Audio is lossless little-endian mono float32, with sample-addressed references.
Overflow/disk failure stops collection visibly rather than silently losing data.
"""
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid

SCHEMA_VERSION = 1


class Recording:
    def __init__(self, root, metadata, *, max_queue_bytes=16 * 1024**2,
                 max_bytes=10 * 1024**3, min_free_bytes=256 * 1024**2):
        self.id = uuid.uuid4().hex
        self.path = Path(root).expanduser().resolve() / self.id
        self.metadata = metadata
        self.max_queue_bytes = max_queue_bytes
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.error = None
        self.closed = False
        self.seq = 0
        self.samples = {}
        self._queued_bytes = 0
        self._total_bytes = 0
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._write, daemon=True,
                                        name=f"recording-{self.id[:8]}")
        self._thread.start()
        self.event("recording.started", metadata=metadata)

    def event(self, kind, **data):
        return self._enqueue(kind, data)

    def audio(self, track, pcm, sample_rate, **data):
        if track not in {"mic", "asr_input", "tts", "model_input"}:
            raise ValueError("unknown audio track")
        if len(pcm) % 4:
            raise ValueError("float32 audio must contain whole samples")
        return self._enqueue("audio.block", data, (track, bytes(pcm), sample_rate))

    def _enqueue(self, kind, data, audio=None):
        with self._lock:
            if self.closed or self.error:
                return None
            seq = self.seq + 1
            event = {"schema_version": SCHEMA_VERSION, "recording_id": self.id,
                     "seq": seq, "event_id": f"{self.id}:{seq}", "kind": kind,
                     "wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                     "data": data}
            ref = None
            if audio:
                track, pcm, rate = audio
                ref = {"path": f"{track}.f32", "sample_rate": rate, "channels": 1,
                       "encoding": "float32_le", "start_sample": self.samples.get(track, 0),
                       "samples": len(pcm) // 4, "sha256": hashlib.sha256(pcm).hexdigest()}
                event["data"] = {**data, "audio": ref}
            encoded = (json.dumps(event, ensure_ascii=False, allow_nan=False,
                                  separators=(",", ":")) + "\n").encode()
            size = len(encoded) + (len(audio[1]) if audio else 0)
            if self._queued_bytes + size > self.max_queue_bytes:
                self.error = "writer queue limit reached; recording is incomplete"
                return None
            if self._total_bytes + size > self.max_bytes:
                self.error = "recording size limit reached; recording is incomplete"
                return None
            self.seq = seq
            self._queued_bytes += size
            self._total_bytes += size
            if audio:
                self.samples[track] = ref["start_sample"] + ref["samples"]
            self._queue.put((encoded, audio, size))
            return {"event_id": event["event_id"], **({"audio": ref} if ref else {})}

    def close(self, reason="stopped"):
        """Call via asyncio.to_thread: drains accepted events and fsyncs files."""
        self.event("recording.stopped", reason=reason)
        with self._lock:
            if not self.closed:
                self.closed = True
                self._queue.put(None)
        self._thread.join()

    def flush(self):
        barrier = threading.Event()
        with self._lock:
            if self.closed or self.error:
                return False
            self._queue.put((None, barrier, 0))
        while not barrier.wait(.1):
            if not self._thread.is_alive():
                return False
        return not bool(self.error)

    def _manifest(self, complete):
        manifest = {"schema_version": SCHEMA_VERSION, "recording_id": self.id,
                    "complete": complete, "error": self.error,
                    "metadata": self.metadata}
        path = self.path / "manifest.json"
        tmp = self.path / "manifest.tmp"
        with open(tmp, "w", opener=self._private_open) as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)

    @staticmethod
    def _private_open(path, flags):
        return os.open(path, flags, 0o600)

    def _write(self):
        files = {}
        try:
            self.path.mkdir(parents=True, mode=0o700)
            self._manifest(False)  # A crash leaves this false.
            events = open(self.path / "events.jsonl", "ab", opener=self._private_open)
            files["events"] = events
            last_sync = 0
            while True:
                try:
                    item = self._queue.get(timeout=1)
                except queue.Empty:
                    for f in files.values():
                        f.flush()
                        os.fsync(f.fileno())
                    continue
                if item is None:
                    break
                encoded, audio, size = item
                if encoded is None:
                    for f in files.values():
                        f.flush()
                        os.fsync(f.fileno())
                    audio.set()  # explicit human feedback durability barrier
                    continue
                if time.monotonic() - last_sync >= 1:
                    if shutil.disk_usage(self.path).free < self.min_free_bytes:
                        raise OSError("free disk space below recording reserve")
                    for f in files.values():
                        f.flush()
                        os.fsync(f.fileno())
                    last_sync = time.monotonic()
                if audio:
                    track, pcm, _ = audio
                    if track not in files:
                        files[track] = open(self.path / f"{track}.f32", "ab",
                                           opener=self._private_open)
                    files[track].write(pcm)
                events.write(encoded)
                with self._lock:
                    self._queued_bytes -= size
            for f in files.values():
                f.flush()
                os.fsync(f.fileno())
            self._manifest(not bool(self.error))
        except Exception as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
            try:
                self._manifest(False)
            except Exception:
                pass
        finally:
            for f in files.values():
                f.close()
            # Release queued audio after failure, even if the session stays open.
            with self._lock:
                while not self._queue.empty():
                    self._queue.get_nowait()
                self._queued_bytes = 0


def validate_recording(path):
    """Check ordering, audio spans/hashes and clean shutdown without printing content."""
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    errors = []
    counts = {}
    ends = {}
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported manifest schema version")
    if not manifest.get("complete"):
        errors.append("recording was not closed cleanly: " + str(manifest.get("error")))
    with open(path / "events.jsonl") as f:
        for seq, line in enumerate(f, 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or event.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("unsupported event schema")
                if event["seq"] != seq or event["recording_id"] != manifest["recording_id"]:
                    errors.append(f"event {seq}: identity/sequence mismatch")
                if event.get("event_id") != f"{manifest['recording_id']}:{seq}":
                    errors.append(f"event {seq}: event ID mismatch")
                kind = event["kind"]
                counts[kind] = counts.get(kind, 0) + 1
                if kind == "audio.block":
                    ref = event["data"]["audio"]
                    if (ref.get("encoding") != "float32_le" or ref.get("channels") != 1
                            or ref.get("sample_rate") not in {16000, 24000}
                            or not isinstance(ref.get("start_sample"), int)
                            or not isinstance(ref.get("samples"), int)
                            or ref["start_sample"] < 0 or ref["samples"] < 0):
                        raise ValueError("invalid audio format/span")
                    name = ref["path"]
                    if name not in {"mic.f32", "asr_input.f32", "tts.f32", "model_input.f32"}:
                        raise ValueError("invalid audio path")
                    if ref["start_sample"] != ends.get(name, 0):
                        errors.append(f"event {seq}: noncontiguous {name}")
                    with open(path / name, "rb") as pcm:
                        pcm.seek(ref["start_sample"] * 4)
                        block = pcm.read(ref["samples"] * 4)
                    if len(block) != ref["samples"] * 4 or hashlib.sha256(block).hexdigest() != ref["sha256"]:
                        errors.append(f"event {seq}: damaged audio")
                    ends[name] = ref["start_sample"] + ref["samples"]
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(f"event {seq}: {exc}")
    if counts.get("recording.started") != 1 or counts.get("recording.stopped") != 1:
        errors.append("missing or duplicated recording boundaries")
    for name, end in ends.items():
        if (path / name).stat().st_size != end * 4:
            errors.append(f"{name}: unreferenced audio bytes")
    return {"recording_id": manifest["recording_id"], "valid": not errors,
            "events": counts, "audio_samples": ends, "errors": errors}
