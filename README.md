# Realtime Speech Chat

Real-time voice conversations with an LLM: a GPU server runs the models, and
you talk to it from a browser (or a Python CLI client) on your local machine.

## Architecture

```
local machine                          GPU box (server/)
┌─────────────────────┐   WebSocket   ┌──────────────────────────────────┐
│ browser UI at :8000 │──────────────▶│ silero VAD → Whisper turbo (ASR) │
│  (or client/main.py)│  (SSH tunnel) │ Kokoro TTS ← sentence split      │
│ mic + playback +    │◀──────────────│        ▲                         │
│ live pipeline view  │               │  streaming tokens                │
└─────────────────────┘               │  vLLM (OpenAI API, port 8001)    │
                                      └──────────────────────────────────┘
```

- The client streams mic PCM continuously; the server does VAD/endpointing.
- LLM tokens stream out and are split into sentences; each sentence is
  synthesized and sent back immediately (~0.5s from end-of-speech to first audio).
- Barge-in: talking over the assistant cancels generation server-side and
  stops playback client-side.

## GPU box setup

```bash
# LLM: any OpenAI-compatible server on :8001, e.g. vLLM in docker
sudo docker run -d --name chatllm --restart unless-stopped --gpus '"device=0"' \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 127.0.0.1:8001:8000 --ipc=host vllm/vllm-openai:v0.19.0 \
  --model Qwen/Qwen3.5-35B-A3B --max-model-len 32768 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml

# speech server (Whisper + Kokoro go on the second GPU if present)
sudo apt-get install -y espeak-ng
uv venv -p 3.12 .venv
uv pip install -p .venv torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv -r server/requirements.txt pip
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python server/main.py            # binds 127.0.0.1:8000
```

## Using a different LLM backend

The chat model is any OpenAI-compatible endpoint. For a remote model, e.g.
Thinking Machines' Inkling on Tinker:

```bash
export TINKER_API_KEY=...
.venv/bin/python server/main.py \
  --llm-url https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1 \
  --llm-model thinkingmachines/Inkling \
  --llm-api-key-env TINKER_API_KEY \
  --reasoning-effort low \
  --audio-input
```

`--audio-input` sends each user turn as audio (WAV) for audio-native models,
so the model hears your actual voice; it falls back to the Whisper transcript
automatically if the endpoint rejects audio. Whisper still runs either way —
the UI transcript, chat history, and end-of-turn classification use it. The
end-of-turn classifier keeps using the fast local vLLM (`--classifier-url`)
so remote-model latency never delays endpointing.

## Talking to it

Open an SSH tunnel and point a browser at the server:

```bash
ssh -f -N -L 8000:127.0.0.1:8000 <gpu-box>
open http://localhost:8000
```

Press **Start** and talk. The page shows the whole pipeline live: a status
pill (listening → hearing you → transcribing → thinking → speaking), mic and
output level meters, the transcript with streaming replies and per-turn
latency, and a timestamped event log (VAD hits, transcriptions, TTS chunks,
barge-ins). Browser mic capture has echo cancellation, so speakers are fine —
just talk over the assistant to interrupt it.

### CLI client (alternative)

```bash
uv venv .venv && uv pip install -p .venv -r client/requirements.txt
.venv/bin/python client/main.py
```

Type text to send a text turn, `i`/`stop` to interrupt, `/v af_bella` to
switch voice, `/q` to quit. No echo cancellation — use headphones, or tune
`--barge-rms`.

## Testing without a mic

```bash
.venv/bin/python scripts/test_e2e.py --server http://localhost:8000
```

Sends a text turn and streams `sample/female.wav` as a fake mic; checks
transcription, reply, and returned audio, and prints latency numbers.

## Layout

- `server/` — the speech server (FastAPI websocket + browser UI in `static/`)
- `client/` — optional Python CLI client
- `scripts/` — end-to-end test
- `demos/` — apps built on the server's client-tools protocol: a client can
  `register_tools` over the websocket and the model's calls to those tools
  are forwarded to the client for execution (see `demos/retro-os/`)
- `old/` — the previous generation of this project (blocking pipeline,
  WebRTC experiment, manual Kokoro setup); kept for reference, not used

## Toward Jarvis: roadmap

The active feature backlog and proposed delivery order are in
[ROADMAP.md](ROADMAP.md), updated September 2026. The notes below preserve the
earlier roadmap and measurements.

Ideas for making the assistant feel instant, controllable, and vivid.
Numbered for reference; ✅ = done.

1. **Async agent + sub-agents** — long tool work (installs, builds) runs as
   background jobs while the conversation stays live; worker agents grind on
   big tasks and report back; completions surface proactively.
2. **Reflex + cortex model split** — fast local model answers instantly and
   handles simple turns; big remote model engaged for depth. Pre-synthesized
   instant acks ("Sure.") mask remaining think time.
3. ✅ **Speculative everything** — see below.
4. **Real STOP + undo** — "stop" kills playback, generation, and in-flight
   tools within ~100ms; undo stack for window ops and file edits; an
   always-answerable "what are you doing?".
5. **Voice-speed confirmation** — dangerous actions get a one-word approval
   flow instead of silent execution.
6. ✅ **"Show me" windows** — see below.
7. **Ambient context** — a compact desktop snapshot (windows, shown file,
   recent terminal) attached to every turn so "fix that" resolves without a
   lookup.
8. **Cross-session memory** — distilled facts and project state persisted
   and injected into future sessions.
9. **Reliability** — tunnel keep-alive or move ASR/TTS local to the laptop;
   supervisor + health chip in the UI. Long-game: fine-tune a small local
   model on this system's own tool-use traces (reflex layer handles ~90% of
   turns alone).

### Done: #3 speculative everything (server)

Three overlaps, all in `server/main.py`:

- **Early ASR**: transcription starts at 150ms of trailing silence — 200ms
  before the endpointing pause fires at 350ms — so the transcript is ready
  when the turn decision starts.
- **Speculative generation**: when the turn-complete verdict needs the LLM
  classifier, the response starts generating in parallel under a hold gate —
  nothing is sent or committed to history until DONE releases it; WAIT or
  resumed speech cancels it without a trace. Tool execution is never
  speculated (visible side effects).
- **Prewarm**: Whisper's first-call CUDA cost is paid at boot; each new
  session fires a 1-token LLM request to warm the connection.

Measured (local Qwen3.5-35B-A3B, e2e test with realtime-paced audio): the
LLM's first token now lands **together with** the endpoint verdict ~285ms
after the pause instead of after it (serialized it would be ~565ms — the
speculation refunds ~280ms), the early ASR hides another ~200ms of
transcription, and text turns hit **83–87ms first token** (prewarmed; the
first-turn penalty of ~250ms is gone). Net: last word → first audio ≈
**0.85s** (0.35s endpoint patience + ~0.5s pipeline), and turns that end
unambiguously (e.g. questions) skip the classifier entirely.

### Done: #6 "show me" windows (demo)

Two new code-editor verbs in `demos/retro-os`, both windows fully managed by
the generic window tools:

- **`show_image(window_id, path)`** — displays any image file from the
  sandbox in a retro viewer window (backend `/raw` endpoint serves sandbox
  file bytes with the right mimetype). Re-calling with the same path
  refreshes in place — save a chart, show it, regenerate, show again.
- **`open_preview(window_id, path?)`** — live browser (iframe) window
  pointed at whatever web server runs on **port 8000 inside the sandbox**;
  each sandbox's port 8000 is published to an ephemeral localhost port at
  container creation. The agent can build a FastAPI app, run it, and put the
  actual live page on your screen.

Verified end-to-end: PNG created in a sandbox renders through `/raw`
(image/png), and a server on sandbox port 8000 is reachable through the
published port. The model's instructions now say "show, don't tell."

### Model note (Jul 2026)

Default is back to the **local Qwen3.5-35B-A3B** (vLLM, GPU 0) — first
token ~285ms vs Inkling's ~1.8–2.3s (reasoning + WAN), and current tasks
don't need the extra depth. The vLLM container now runs with 32k context so
the 20k-token history budget fits. Inkling stays one relaunch away (see
"Using a different LLM backend") for when Tinker ships native audio input.


## Training data collection

RetroVoice now records audio and agent/app activity when the desktop starts, with
visible pause/resume controls and explicit feedback buttons. Data is saved on the
speech server. See [the collection guide](docs/DATA_COLLECTION.md) for the format,
coverage, storage location, copying to the Mac, and integrity checks.
