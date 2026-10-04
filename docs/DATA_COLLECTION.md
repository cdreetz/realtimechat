# Training data collection

RetroVoice OS starts recording when you start the desktop and connect to a
server that supports recording. **Pause recording** stops collection and saves
the current segment; **Record session** starts a new one. The preference survives
reloads. The basic speech-chat page also has recording controls, initially off.
Recording never starts merely because the server is running.

Data stays on the machine running `server/main.py`, under
`data/recordings/<recording_id>/`. For the current GPU setup this is
`/home/ubuntu/realtimechat/data/recordings/`, not the Mac. Change the directory with
`--recordings-dir /your/persistent/disk/recordings`. There is no automatic upload,
retention deletion, training job, or cloud dataset registration.

## Contents and alignment

Each segment contains `manifest.json`, append-only `events.jsonl`, and lossless
mono little-endian float32 PCM tracks (`mic.f32`, `asr_input.f32`, `model_input.f32`,
`tts.f32`, created as needed). Each audio block includes its track, sample rate,
sample offset, sample count, and SHA-256. There are no container headers; read
with `numpy.fromfile(path, dtype='<f4')`. Preserve the JSONL next to the audio.

Every event has a schema version, recording ID, sequence number, event ID,
server monotonic timestamp and wall timestamp. Shared speech IDs, response run
IDs, utterance numbers, ASR IDs and tool call IDs link the pipeline. Scope short
IDs such as `utt` and `call_id` by `connection_id`; they can repeat across
connections. Reconnect creates another recording segment; client session/tab IDs
link related segments. Resuming a recording never backfills audio from a pause.
Model requests can contain earlier conversation history, including paused turns.

Recorded signals:

- Continuous microphone audio, including silence, at 16 kHz. This is the signal
  delivered to the app after browser echo cancellation/noise suppression/automatic
  gain control and any app resampling, **not untouched device audio**. Browser
  media settings and context sample rates are stored. Capture packet sequence and
  sample counters expose gaps when disconnected; server input sample counters
  index the stream that VAD actually processed.
- Per-frame VAD probabilities and sample positions, configuration, onset/pause/
  resume/end events, and utterance sample spans when an audio segment exists.
- Exact audio submitted for early/pause/final ASR, the raw ASR output with its
  segment timestamps, consumer completion/cancellation, and filtered transcript
  output. A completed inference remains observable after its consumer cancels,
  while the recording remains open. Current ASR is pause-based, so these are
  successive attempts, not continuously streamed partial hypotheses.
- Endpoint classifier input/output, heuristic decisions delivered to the client,
  speculative response starts, release decisions, and cancellation/error status.
- Model request snapshots (system instructions, visible history, tools, sampling
  parameters), all received streaming chunks including tool argument fragments,
  complete proposed tool calls, execution starts, full tool return values and
  late client results. Audio-native requests reference the source waveform and
  its WAV PCM16 conversion rather than duplicating base64. Token IDs/logprobs are
  not requested. Collection does not change the model's sampling configuration.
- Generated TTS waveform/text/voice, corresponding websocket sends, playback
  scheduling, stop/end observations and estimated played samples. Web Audio
  timing cannot establish what was actually audible or heard by the user.
- RetroVoice desktop snapshots at recording start, before/after tool calls, after
  direct input/selection/window interactions, and when periodic state changes.
  Snapshots include note text, shown code, current filenames, browser URLs and
  window geometry. They do not include screenshots, every filesystem revision,
  browser DOM history, or full unbounded terminal output. Tool return values keep
  the existing tool-level truncation limits.
- Client/server clock exchanges, including server receive/send times and browser
  reply times. Estimate clock offset and uncertainty from these round trips;
  do not compare the two machines' monotonic clocks directly. Microphone packet
  timestamps reflect main-thread delivery, not a calibrated device timestamp.

The manifest records model identifiers, VAD settings, runtime versions and source
file hashes. Model weight hashes and tokenizer versions are not yet resolved.
Future collectors can add image/video tracks, streaming transcript revisions,
device-level timestamps and richer environment diffs without flattening the
existing multimodal record.

## Labels and corrections

Transcript, response and tool-result rows in RetroVoice have **Correct**,
**Incorrect**, and **Correct / revise…** buttons while recording. The revision
button records the desired transcript, response or action as free text. These
are append-only explicit human annotations; they do not rewrite the raw event or
perform an action. An acknowledgement follows a disk flush. Old rows from a
closed recording cannot accidentally label a new session.

Tool success is `correctness: unlabeled`. Silence, a cancellation, resumed speech,
a user edit, or an interruption never automatically becomes a positive/negative
label. A later spoken correction remains in the timeline for review; the current
implementation does not infer its target. For offline annotation, write a
separate versioned JSONL sidecar keyed by `recording_id` + `event_id` (or scoped
`connection_id`/`utt`/`call_id`), retaining annotator, label, desired correction,
timestamp and evidence. Do not alter the original events.

## Durability and verification

A bounded writer thread handles filesystem I/O. Files are periodically flushed
and synced, then drained/synced on stop or disconnect. Recording directories use
mode 0700 and files 0600. Audio and events are excluded from Git. The manifest
starts with `complete: false`; clean shutdown marks it true. Process death or
disk/queue/size failure leaves an incomplete trace, which must be validated before
training. `complete` means a clean capture interval, not that every action finished
or every inference result was captured. Starting/stopping in mid-turn gives a
partial example.

Per segment: 16 MiB writer queue, 10 GiB total queued/written data limit and a
256 MiB free-disk reserve. Hitting a limit stops collection and displays an error;
conversation can continue. There is no silent rolling overwrite. Continuous mic
float32 alone uses about 230 MB/hour; TTS, repeated ASR/model segments and event
text add to that. No automatic retention policy is applied.

Run on the server or a local copy:

```bash
python scripts/inspect_recording.py data/recordings/RECORDING_ID
```

The command prints counts and integrity errors without dumping conversation
content. It checks clean closure, sequence/identity, audio offsets, lengths and
hashes. It returns nonzero for an incomplete or damaged recording. It does not
certify labels or task success. To copy data from the current GPU host to the Mac,
without deleting remote or local recordings:

```bash
mkdir -p data/recordings
rsync -av --partial --exclude='manifest.tmp' \
  ubuntu@154.54.100.205:realtimechat/data/recordings/ data/recordings/
```

Prefer copying after recording stops. A live copy may be inconsistent; recopy
after closure and run the validator. The GPU disk needs a separate backup before
the machine is destroyed. Protect these files as recordings of your conversations
and work; their exact text/audio can include sensitive content you discuss.

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -v
node --check server/static/recording-client.js
```

Inference is mocked in the unit tests; they exercise the real recorder, Session,
and VAD state machine. GPU smoke tests should use clearly identified synthetic
sessions, separate from human training examples.
