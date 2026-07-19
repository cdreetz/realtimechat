# RetroVoice OS demo

A Win95-style desktop you control by voice. Ask the assistant to open
notepads, write notes, move/minimize/close windows, or open a **code
editor** — a real Docker sandbox (python:3.11-slim) where it can create
files, edit them, and run bash, with the editor and terminal visible on the
desktop. All of it happens via client-side tools this page registers with
the speech server over the websocket (`open_notepad`, `write_note`,
`move_notepad`, `min_notepad`, `close_notepad`, `get_desktop_state`,
`open_code_editor`, `create_file`, `edit_file`, `open_file`, `run_bash`,
`close_code_editor`). The main realtimechat server knows nothing about any
of this; it just forwards tool calls to whoever registered them.

`backend.py` serves the page and manages the sandboxes, keeping a warm pool
of 2 containers so opening an editor is instant. Sandboxes are capped
(1 GB RAM, 2 CPUs, 30s per command) and destroyed on close/shutdown.

## Run

Requires Docker running locally. The speech server must be reachable (see
the main README; typically an SSH tunnel to the GPU box on localhost:8000).

```bash
pip install fastapi uvicorn
python3 demos/retro-os/backend.py
```

Open http://localhost:8080 (add `?server=http://host:port` if the speech
server is not on localhost:8000), click **Start**, allow the mic, and say:

- "open a notepad"
- "write down: milk, eggs, bread"
- "move the notepad to the top right"
- "minimize it"
- "close all the notepads"
- "open a code editor and write a python script that prints the first 20
  primes, then run it"
- "now change it to print them in reverse"
