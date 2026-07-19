# RetroVoice OS demo

A Win95-style desktop you control by voice. Ask the assistant to open
notepads, write notes into them, move them around the screen, or minimize
them — it does it via client-side tools (`open_notepad`, `write_note`,
`move_notepad`, `min_notepad`, `close_notepad`, `get_desktop_state`) that
this page registers with the speech server over the websocket. The main realtimechat server knows nothing about
notepads; it just forwards tool calls to whoever registered them.

## Run

The speech server must be reachable (see the main README; typically an SSH
tunnel to the GPU box on localhost:8000). Then serve this folder statically:

```bash
cd demos/retro-os
python3 -m http.server 8080
```

Open http://localhost:8080 (add `?server=http://host:port` if the speech
server is not on localhost:8000), click **Start**, allow the mic, and say:

- "open a notepad"
- "write down: milk, eggs, bread"
- "move the notepad to the top right"
- "minimize it"
- "close all the notepads"
