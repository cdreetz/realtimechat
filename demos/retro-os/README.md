# RetroVoice OS demo

A Win95-style desktop you control by voice. Ask the assistant to open
notepads, read and edit notes, move/minimize/close windows, use an isolated
**headless Chromium browser**, or open a **code editor** — a real Docker
sandbox (python:3.11-slim) where it can create files, edit them, and run bash,
with everything visible on the desktop. All of it happens via client-side tools this page registers with
the speech server over the websocket. Every window shares one id space, so
window management is generic — `move_window`, `resize_window`,
`minimize_window`, `restore_window`, `close_window`, and
`get_desktop_state` work on notepads, code editors, and the assistant
window alike — while `open_window(app_type)` opens any app (an app registry
maps types to windows) and app tools cover app behavior: `write_note` /
`read_notepad` / `edit_notepad`,
`create_file` / `edit_file` / `open_file` / `run_bash`, and
`open_browser` / `browser_navigate` / `browser_read_page` / `browser_click` /
`browser_type` / `browser_back`. The main realtimechat server knows nothing about
any of this; it just forwards tool calls to whoever registered them.

Browser sessions run in `chromedp/headless-shell` Docker containers—never in
the host's normal browser or profile. The page shown to the user is a CDP
screenshot rendered inside a RetroVoice window. The model receives bounded
visible text and numbered interactive elements instead of unrestricted raw
CDP access. Browser containers and their in-memory page state survive backend
hot reloads and are destroyed when their window closes.

Code-editor command output streams into the terminal as it arrives. Sandbox
and tool failures are retained behind the **Errors (n)** button in each
editor's toolbar, which toggles between the file view and error details;
the adjacent **Clear** button resets the log.

The assistant window shows a compact timing line after every turn (speech
input/endpoint time, time to first model action, total model time, tool time,
TTS time, and total time).
The speech server also writes `turn_metric` / `tool_metric` JSON records to
its log and exposes the last 20 turns plus p50/p95 summaries at
`http://localhost:8000/metrics` when using the usual SSH tunnel.

Development mode is reload-safe: Uvicorn watches Python files, the page
polls the frontend version and refreshes itself when `index.html` changes,
desktop/notepad/editor state is restored from local storage, and Docker
sandboxes are reattached rather than deleted between workers.

`backend.py` serves the desktop and durable named workspaces. Use **Start →
workspaces** to create or reopen one. Files live under `.workspaces/<id>/files`
on the host, mounted at `/workspace` in its Docker container (1 GB RAM, 2 CPUs).
Closing an editor or terminal only closes its view. Existing legacy sandboxes
can be imported from Workspaces; their original containers are retained.

**Terminal** is an interactive xterm.js PTY with a persistent bash shell. Commands
retain cwd and environment within that shell; separate shells share files.
`run_bash` and `terminal_run` return task IDs immediately. **Task Manager** shows
shells and command status, reopens terminal views, and stops foreground commands
with SIGINT followed by escalation when needed. It does not manage autonomous
background agents yet, or guarantee cancellation of deliberately detached jobs.

Shells survive page closure and backend reloads. If their container restarts or
is replaced, previous shells are shown as exited and unfinished tasks as
interrupted; files and bounded terminal history survive. Installed packages
outside `/workspace` survive only while the same container exists.

The editor supports typing, Save/Ctrl+S, new files, retained unsaved drafts, and
conflict detection if a file changed since it was opened. Save or discard edits
before closing a dirty editor. Terminal writes are external edits and are checked
when saving, but do not participate in an atomic cross-process file lock.

## Run

Requires Docker running locally. The speech server must be reachable (see
the main README; typically an SSH tunnel to the GPU box on localhost:8000).

```bash
pip install fastapi 'uvicorn[standard]' httpx websockets
python3 demos/retro-os/backend.py  # Uvicorn reload mode is enabled
```

Open http://localhost:8080 (add `?server=http://host:port` if the speech
server is not on localhost:8000), click **Start**, allow the mic, and say (or use **Start without microphone**
for manual testing):

- "open a notepad"
- "write down: milk, eggs, bread"
- "read the notepad, then remove eggs"
- "open a browser and go to example.com"
- "move the notepad to the top right"
- "minimize it"
- "close all the notepads"
- "open a code editor and write a python script that prints the first 20
  primes, then run it"
- "now change it to print them in reverse"


## Recording for training

The assistant window shows **Recording audio + activity**, a pause/resume control,
and feedback buttons on transcripts, responses and tool results. Collection starts
with the desktop unless previously paused. Files are written on the speech server,
not by this backend. See [the collection guide](../../docs/DATA_COLLECTION.md).
Recording UI context does not change the agent's focus or prompt context.

## Window snapping

Use the **Snap…** menu in any title bar, or ask the agent to place apps in
left/right halves, thirds, two-thirds, or the full desktop. `snap_window` takes
`window_id` and `layout`; `restore` returns to the position and size before
snapping. Snapped layouts follow desktop resizing and persist with saved windows.
Moving a window manually or through `move_window` / `resize_window` clears its
snap layout. Geometry regression check: `node tests/test_window_snap.cjs`.

## Browser Use execution and background agents

Browser windows now use the official `browser-harness==0.1.13` package in a
separate Docker controller per Chromium container. The primary tool is
`browser_exec(window_id, code, timeout?)`: it returns a task ID immediately;
`read_browser_exec` reads results. Python variables and Browser Use's attached tab
persist across calls. Helpers are synchronous: `new_tab`, `goto_url`,
`wait_for_load`, `page_info`, `js`, `fill_input`, `press_key`, `click_at_xy`, `cdp`,
`list_tabs`, and `switch_tab`. Extract primitive values inside `js`, for example
`print(js("document.querySelector('h1')?.textContent"))`.

The **Python** button opens a human-editable script console and execution output.
**Take control** cancels a running script and blocks agent automation; the page
then accepts clicks, text/paste, navigation keys and scrolling. **Resume agent**
releases browser control. If a worker paused because of takeover, resume the worker
in Background Agents too. **Stop script** stops execution and resets Python
variables. Page actions already performed remain; cancellation is not rollback.
Scripts are serialized and bounded to 120 seconds. Duplicate submitted task IDs
are not replayed. Controller restarts reset variables/generation and execution
history; existing Chromium pages stay open. The controller has no host directory
mounts or Docker socket and shares only its associated Chromium network namespace.
Browser Use's optional third-party telemetry is disabled.

Use **Start → agents** or `spawn_agent(objective, name?, workspace_id?)` for an
independent model workflow. The service starts automatically on localhost:8082
and survives frontend closure, new voice turns, and desktop backend reloads.
Two model workers run simultaneously; up to eight can be active/queued. Each
gets a new persistent workspace by default and creates its own browser when
needed. Choosing an existing workspace is explicit; a second active worker on
that workspace is rejected. Human edits remain possible; worker file writes use
revision checks, while shell commands can modify files directly.

The agent window supports details, steering/replies, Pause, Resume, Stop, and
opening its editor, terminal or browser. Pause takes effect at a tool boundary;
Stop cancels the current model request and stops the worker's owned command and
browser script. A worker started from the desktop opens its workspace views as
resources become ready. Completion/failure appears once in the assistant feed,
without speaking over the user. Ask the primary assistant about progress using
`list_agents` / `read_agent`; `steer_agent`, `stop_agent`, and `resume_agent` control
workers. Workers cannot spawn nested workers.

The worker model defaults to `http://127.0.0.1:8001/v1`, using the first model from
`/models`. With the existing GPU host, forward both speech and model ports:

```bash
ssh -N -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 <gpu-box>
```

Set `RETROVOICE_LLM_URL`, `RETROVOICE_LLM_MODEL`, and optionally
`RETROVOICE_LLM_API_KEY` **before starting the worker service** to override it.
Defaults are 24 model rounds and 20 minutes per run, configurable within bounded
limits at spawn. Worker journals live in `.agents/<task_id>.json` and record
objectives, model messages, tool calls/results, steering and final outcomes.
They are operational task history, separate from the speech recording toggle.
A worker-service restart marks unfinished work interrupted; explicit Resume
repairs pending tool messages and inspects existing effects instead of silently
replaying actions. Shell/browser processes can outlive a crashed worker service;
inspect Task Manager before resuming. `.agents/worker-service.log` has diagnostics.

Validation (requires running Docker/app; last check also requires the live model
and speech server):

```bash
RETROVOICE_APP_TESTS=1 python -m unittest discover -s tests -p test_browser_execution.py
RETROVOICE_AGENT_TESTS=1 python -m unittest discover -s tests -p test_background_agents.py
```
