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
  --model Qwen/Qwen3.5-35B-A3B --max-model-len 8192 --gpu-memory-utilization 0.85

# speech server (Whisper + Kokoro go on the second GPU if present)
sudo apt-get install -y espeak-ng
uv venv -p 3.12 .venv
uv pip install -p .venv torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -p .venv -r server/requirements.txt pip
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python server/main.py            # binds 127.0.0.1:8000
```

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
- `old/` — the previous generation of this project (blocking pipeline,
  WebRTC experiment, manual Kokoro setup); kept for reference, not used
