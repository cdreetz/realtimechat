# RetroVoice OS feature backlog

Updated September 12, 2026. This captures Christian's new ideas, proposals,
and relevant unfinished work from the earlier README roadmap. It is a planning
document. F19 has an initial implementation; the other additions remain backlog
items unless described as existing behavior.

## September 12 decisions

Approved directions: real terminal (F02), Browser Use execution harness (F03),
background agents (F04), persistent workspaces (F05), task manager (F06), human
control of every app (F07), inspectable diffs (F09), and regression checks (F13).
Named snapping regions (F01) are deferred. Focus awareness (F08) needs agreement
on how context is attached; no automatic focus injection is implemented.
Training data collection (F19) was requested for immediate implementation.

The product goal is a voice assistant you can work with for hours: talk naturally,
delegate work, see what is happening, intervene immediately, and return later
without rebuilding your context or losing files. Keep the retro desktop style;
make the behavior dependable.

## What we already have

- Streaming speech, semantic turn detection, speech interruption, and speculative
  response generation. Local Qwen is the last selected default; Inkling/native
  audio remains an alternative.
- Generic move/resize/minimize/restore/close tools and an app registry containing
  notepad, code editor, and browser. Pixel movement and resizing already exist.
- Docker-backed editors, streamed command output, visible errors, notepad tools,
  image viewers, web previews, and isolated Chromium controlled through CDP.
- Local desktop restoration and container reattachment across development
  reloads; reconnection logic and per-turn timing metrics.

The existing implementation still couples editor windows to disposable containers.
Closing an editor destroys its sandbox. Commands are awaited inside the response
loop, speech history belongs to a WebSocket connection, and speech interruption
does not cancel a tool already running in the client. These are the main constraints
behind the proposals below.

## Christian's feature ideas

### F01 — Natural window layouts

**Decision:** defer snapping for now; pixel move/resize tools remain available.

**Starting point:** move and resize tools exist. The missing layer is reliable
placement by intent, coordinated across several windows.

- Support left/right halves, quarters, maximize, side-by-side, and relative sizes.
- Let `open_window` take optional placement; add a batch layout operation for
  existing windows. Preserve pixel-level tools for precise adjustments.
- Use normalized bounds or named regions so layout survives a desktop resize.
- Apply geometry together and keep title bars and controls reachable. Handle
  minimum sizes and an odd number of pixels without unexpected overlap.
- Support mouse resize handles, snapping, and saving useful layouts. Keep the
  assistant visible without blocking the requested workspace.

**Done when:** “Open a code editor on the left half and a notepad on the right
half” produces the intended arrangement, survives reload, and adapts to a smaller
desktop. Repeating the placement on existing windows does not create duplicates.

**Dependencies:** can ship on the current window manager; migrate saved layouts
to persistent workspaces when F05 lands.

### F02 — A real terminal app

**Starting point:** an editor contains a command-output pane, but the app has no
standalone interactive terminal and each `run_bash` starts a separate shell.

- Add `terminal` to the app registry, attached to a named workspace.
- Back it with a persistent PTY: working directory, shell variables, stdin,
  interactive prompts, Ctrl-C, resize events, and scrollback.
- Let the user type directly and let the agent send commands/input to the same
  session. Show ownership while an agent is operating it.
- Editors and terminals in one workspace see the same files and environment.
- Support long-running servers as visible tasks. Closing the terminal view can
  detach it; explicitly stopping a process is a separate action.
- Build on the task lifecycle so output, completion, and cancellation behave
  consistently with other tools. Proposed API: terminal open/write/read plus
  shared task status/cancel operations; finalize names during implementation.

**Done when:** `cd` persists between commands, an interactive prompt accepts input,
the agent starts a preview server, the user can interrupt a foreground command,
and the terminal reattaches after a page reload.

**Dependencies:** F05 workspace identity and F06 task/process ownership.

### F03 — Browser execution harness

**Starting point:** separate browser navigate/read/click/type tools wrap CDP.
The intended change is one primary `browser_exec` tool where the agent writes
code to operate a persistent browser session. The reference is
[Browser Use's browser-harness](https://github.com/browser-use/browser-harness).
Its [Browser Use MCP integration](https://github.com/browser-use/browser-use/blob/main/browser_use/mcp/cli_mcp.py)
exposes `browser_exec(code)` running Python in a persistent namespace, with
preloaded browser helpers and raw CDP access, plus `browser_screenshot`.

- Proposed entry point: `browser_exec(window_id, code, timeout?)`, backed by a
  persistent isolated Python harness session with documented browser helpers.
- Support navigation, accessible locators, waits, extraction, screenshots, tabs,
  and small batches of dependent actions in one call. Return structured results,
  execution errors, and artifact references with bounded output.
- Keep the browser visible in RetroVoice OS and expose progress during a longer
  script. Add a user takeover/resume flow for manual interaction and login.
- Preserve handles between calls during a live runtime. After controller restart,
  clearly reset stale handles and reattach pages instead of pretending variables
  survived. Serialize conflicting actions on one page; separate worker pages.
- Run the controller in an isolated execution environment. Giving the model
  Python must not implicitly expose the Mac's filesystem, process environment,
  or all browser sessions. Enforce workspace ownership, execution deadlines, and
  task cancellation at the execution boundary.
- Migrate current click/type wrappers onto the same implementation before
  removing redundant tools. Keep old sessions compatible during that transition.

**Implementation distinction:** agent-written Python drives the browser through
helpers/CDP. The `js(...)` helper separately executes page JavaScript. Our adapter
must preserve isolated Chromium windows and user takeover. Research is complete;
the harness has not yet been installed or wired in.

**Done when:** an agent fills a local test form, submits it, waits for the result,
and extracts the result in one execution; a second call reuses the session;
the user sees the page update; cancellation settles the task; reconnect cannot
silently rerun the submission. Test controller restart and stale-handle recovery.

**Dependencies:** the harness can be prototyped against existing containers; its
finished lifecycle should use F05/F06. Exact harness reference is an implementation
question to resolve before choosing an adapter.

### F04 — Subagents and background agents

**Starting point:** no worker runtime or background task registry exists. The
response loop currently allows five model rounds and awaits tool results. New
speech cancels that response; an already-dispatched client command may continue,
but its result can lose its waiting response and the remaining plan does not have
an independent owner. That is incomplete concurrency, not a supervised background
workflow.

- Keep one conversational agent responsive while workers research, code, or test.
- Workers receive a bounded assignment, relevant context, workspace/tool access,
  and a budget. Return a task ID immediately instead of waiting for completion.
- Proposed operations: spawn an agent, list/read task status, send steering,
  and cancel. Use one task model for workers, commands, and browser executions.
- Show each worker's objective, state, latest useful progress, and resulting
  files or findings. “What are you doing?” reads actual task state.
- Report completion at a sensible gap in conversation, once. Do not speak over
  the user or narrate every internal step. Surface failures and requests for input.
- Coordinate writes: use file revisions/ownership, and isolated Git worktrees
  where appropriate. Two agents should not unknowingly overwrite the same file
  or drive the same browser page. Bound concurrency and nested spawning.
- Build process-backed jobs first, then a single model worker, then multiple
  workers. Measure contention before adding a model-routing policy.
- Treat additional speech as a new question, a new assignment, or a revision to
  a named/current task. It must not implicitly cancel all existing work. Each
  task has its own plan, context, execution state, and continuation across tools.
- Use a single conversation coordinator to incorporate user input and worker
  events into foreground history. Workers emit structured progress/results rather
  than concurrently appending arbitrary messages to the same history list.
- A clone/install command can be a background process without a separate model.
  A multi-step setup or browser investigation needs an owned workflow or worker
  that keeps deciding what to do next after individual tools finish.

**Done when:** “Build this in the background” starts visible work; the user can
continue talking and manipulating windows, ask about progress, steer the worker,
cancel it, and inspect its result. A second worker cannot silently clobber edits.
The specific prime-rl setup plus Slime browser investigation scenario below is
the main end-to-end target, including a third conversation while both tasks run.

**Dependencies:** F05/F06; F08/F09 improve grounding and review.

## Additional proposals

### F05 — Persistent named workspaces

**Why:** files and project state should outlive an individual window or container.

Introduce a stable workspace ID, durable file storage, associated terminals and
browser sessions, and explicit save/open/archive/delete operations. Closing a
window closes its view; deleting a workspace is a separate deliberate action.
Recreate containers around preserved files after a restart. Capture dependency
setup so a rebuilt environment is reproducible. Preserve existing sandboxes and
export their files during migration; do not run the old destructive cleanup path.

**Done when:** close all windows, restart the backend, reopen the workspace, and
recover its files and layout. Recreating a container preserves files and uses the
saved setup. File persistence and process survival have separate, truthful states.

### F06 — Background tasks, real stop, and progress

**Why:** independence needs explicit ownership and a way to regain control.

Add stable task IDs, durable state transitions, progress/artifact events, status,
and cancellation. Suggested states: queued, running, waiting for user, completed,
failed, cancelled, and interrupted. Persist call IDs and completion records so
reconnects do not blindly repeat side effects. After a crash, reconcile a live
process or mark it interrupted; do not automatically retry an uncertain action.

Separate “stop speaking” from “cancel that task” and a visible “stop all work.”
Cancellation must reach the subprocess group, browser execution, or worker and
report whether it actually stopped. It does not imply rollback of completed work.
Display completed steps and remaining partial output. Prompt/protocol details
stay in an inspector, while the normal UI says what is happening in plain language.

Separate the lifetimes of microphone capture, a spoken response, and each task.
Talking over the assistant stops its audio promptly while unrelated background
tasks continue. “Cancel the browser research” affects that task and its descendants;
“stop all work” cancels all owned tasks. Starting or resuming speech alone is not
task cancellation. Task state remains authoritative across WebSocket reconnects.

**Done when:** a long job does not block conversation, cancellation leaves no
owned child processes running, a reconnect yields one completion event, and
an ambiguous voice “stop” has a consistent documented default.

### F07 — Direct editing and handoff

**Why:** the code editor currently displays code in a read-only textarea; the user should be
able to collaborate directly when that is faster than dictating changes.

Make it a real editable file view with saving, selections, search, dirty-state
indication, and file revisions. User and agent edits use the same source of truth.
Show a conflict or diff instead of overwriting unsaved changes. Add manual browser
takeover and a small command palette for common actions without a voice round trip.

**Done when:** the user edits a function, asks the agent to modify the selection,
and both changes survive; a stale agent edit is detected rather than overwriting it.

### F08 — Awareness of what the user is pointing at

**Decision pending:** resolve “this” from the app the user points at. Clicking
another window must not redirect an existing task or authorize edits to it.
An explicit “use this” attachment is a predictable starting option; automatic
selection context is an alternative to agree on. Recording app observations does
not inject them into the model context.

Possible compact context: active workspace and window, shown
file and selected lines, browser URL, active tasks, and recent relevant errors.
Use stable IDs and revisions. Resolve “this,” “that error,” and “the other window”
from the actual UI. Fetch larger content on demand; show when context was omitted.

**Done when:** “Fix this error” from a focused terminal references its latest
failure, and “explain this” references the selected code without naming a file.
The agent verifies stale or ambiguous context before changing the wrong target.

### F09 — Changes you can inspect and undo

Give edits, note changes, and layouts a visible change history with checkpoints,
diffs, and undo. Associate changes with task IDs so a worker result is reviewable.
Keep meaningful outputs visible and linked to their source. Undo applies to
reversible local changes; do not imply that external actions can always be undone.

**Done when:** “Show what you changed” opens the actual diff, and “undo that edit”
restores the prior file without erasing a later user change. Closing a view never
counts as permission to destroy its project.

### F10 — Session continuity and editable memory

Persist conversation state independently of WebSocket connections and save
workspace goals, decisions, unfinished tasks, and concise resumable summaries.
Let the user inspect, correct, and forget saved facts. On recovery, reconcile
remembered state against actual files and tasks; never treat a summary as proof
that an action finished. Retain large tool outputs as artifacts with references
instead of silently cutting off the only useful copy in history.

**Done when:** reopening a workspace can answer “where were we?” with accurate
unfinished work and file links. Reconnecting does not start a memoryless assistant.

### F11 — Bring real files in and take results out

Add upload/drag-and-drop, a workspace file browser, download/export, and import
or clone of an existing project. Editors, terminal tasks, and browser downloads
should share a clear workspace artifact location. Let the agent show the exact
output file rather than explaining where a user might find it inside Docker.

**Done when:** drop in a CSV, ask for a chart, see the chart, download the output,
and recover both input and output after reopening the workspace.

### F12 — One reliable start/resume path

Provide one local launcher and an in-app health view for Docker, the demo backend,
the tunnel, the model endpoint, and speech readiness. Reconnect or restart only the
failed component. Show readiness and useful startup progress. Keep configuration
and model selection visible; routine frontend work should not reload GPU models.
Revalidate the older session's tunnel-supervision setup rather than assuming it
is still installed on this machine.

**Done when:** after laptop sleep or a broken tunnel, the app explains the problem
and recovers the connection without deleting workspaces or restarting healthy services.

### F13 — Replayable behavior and latency checks

Expand beyond the current text/audio smoke test with recorded text or supplied
audio fixtures and a local deterministic demo site. Check actual final state for
layout, edit/run/preview, navigation, interruption, reconnect, and worker control.
Save enough trace detail to inspect failures, with recording/retention visible
to the user. Offer a simulated model/tool runner for local iteration without a GPU.

Measure last audible user sample to first audible response, first visible action,
task completion, cancellation settlement, and task success. Keep model timing
separate from UI-observed timing; compare the same scenarios before and after.
Choose performance targets from a reproducible baseline rather than historical
README numbers measured from different starting points.

For continuous-input work, also measure recognized-clause end to first visible
action, partial transcript stability/correction rate, duplicate or premature
actions, and task success after a correction. For concurrency, measure foreground
response latency with zero, one, and two active workers; missed/orphaned results;
cancellation isolation; and time spent speaking over the user. Use realtime-paced
long audio fixtures, including negation, quotation, self-correction, and continued
speech beyond the current 45-second utterance cap. Compare the same fixtures and
report distributions rather than promising an unmeasured latency target.

**Done when:** a change can demonstrate which behaviors passed, what got slower,
and a replay of a failure. This should start with the first implementation batch.

### F14 — Reusable routines and extensible tools

Support saved routines such as “set up my coding workspace”: open a project,
arrange windows, start the dev server, and show a preview. Give app/tool extensions
a versioned registry and discoverable help. Load relevant tool groups on demand
as the catalog grows. Separate the UI, app handlers, browser runtime, and workspace
services incrementally so adding an app does not require editing one giant file.

**Done when:** replay a saved setup against a different named project with clear
inputs and results; adding an app registers its tools and window behavior through
one documented extension point.

### F15 — Fast conversation with optional deeper workers

Keep the currently selected fast conversation model and allow a worker to use a
different configured model for a harder task. Show the active model and task cost
or usage budget when available. Improve turn-taking with visible transcript
correction and selectable voice/push-to-talk modes. Investigate local speech
processing only after measuring where current latency and connection costs are.

**Done when:** a slow worker or model failure does not stall the voice agent;
completion is announced at a conversational gap; comparisons use F13's same tasks.

## Realtime experience additions

### F16 — Act on complete instructions while the user keeps speaking

**Source:** Christian's September 12 follow-up: deliver a long spoken request and
watch actions happen before the whole request ends.

**Current behavior:** audio streams continuously, but transcription starts only
after roughly 150 ms of trailing silence. Around 350 ms, the server evaluates
whether the turn is finished. It can generate a response speculatively while
waiting for that decision, but holds tool execution until DONE. A 2.4-second
silence or 45-second length cap also finalizes input. The present implementation
does not continuously transcribe and execute independent clauses during speech.
The durations are configured targets quantized to VAD frames, not exact deadlines.

**Additional speculation:** generate a candidate tool call from “Open a co…”, then
hold it until intent confidence and the action policy permit execution. Prepare,
commit, revise and discard are distinct events. Current generation starts at a
pause, not from continuously arriving partial transcripts.

**Proposed behavior:** “Open an editor on the left, put a notepad on the right,
and in the editor clone prime-rl while I explain what I want to do next.” The
windows appear and setup can begin as their instructions become clear. The user
keeps the conversational floor; the assistant can act visually without speaking
over the rest of the request.

- Add ongoing partial transcripts with a stable prefix and a revisable tail.
  Compare an incremental adapter around our current ASR with a streaming ASR
  backend on our own latency/accuracy fixtures. Do not assume shorter audio chunks
  alone make transcription accurate or instant. The
  [Whisper-Streaming paper](https://arxiv.org/abs/2307.14743) demonstrates one
  local-agreement approach; adopting it or meeting any speed target needs testing.
- Detect complete actionable clauses independently of end-of-turn. Transcript
  stability is evidence about the words, not proof of the user's intent. Track
  negation, quoted examples, hypothetical requests, dictation, and references to
  earlier clauses. “Take notes while I explain” or “wait until I'm finished”
  suppresses execution until the user releases it.
- Give each proposed action a stable identity tied to the speech segment and
  intent revision. Never create a new window or resubmit a job just because an
  overlapping transcript repeats the same instruction.
- Preview incomplete instructions, start clearly requested reversible actions
  early, and wait for missing parameters or uncertain intent. Existing boundaries
  for destructive or externally committing actions still apply. This must not
  become a confirmation question for every ordinary window operation.
- Treat “actually, put it on the right” as a revision of the same layout action.
  Reconcile what already happened, amend pending work, and cancel or undo the
  affected action when possible. A stable clause may still receive a later human
  correction; do not promise all action can be final before speech ends.
- Carry newly spoken details to an already-started task without dropping the
  remainder of the utterance. Use bounded buffers and process only useful new
  transcript evidence; avoid a full new LLM request for every token or building an
  ever-growing ASR backlog on the shared speech GPU.

**Done when:** a realtime-paced long recording opens an editor and a notepad
before the recording ends, starts one setup task, and incorporates later details.
A left-to-right correction moves the same window; transcript revisions and
reconnection create no duplicates. Quoted commands and “don't open it yet” do not
trigger actions. The assistant stays quiet while the user continues speaking.

**Dependencies:** F06 task/action identity, F08 context, and F13 recorded checks.
Build a visible partial-transcript and tentative-intent prototype before enabling
execution; then enable a narrow set of window actions before broader tool use.

### F17 — Attention, task routing, and conversation priority

**Why:** multiple successful workers can still make a poor voice experience if
they compete for the microphone, GPU, or the user's attention.

- Keep a fast foreground conversation coordinator. Worker completion can update
  a window immediately; spoken updates go through one scheduler and wait for a
  conversational gap. Coalesce low-value progress rather than reading logs aloud.
- Distinguish a follow-up to the current task, a new independent task, a general
  question, and cancellation. Combine explicit names with the focused window and
  recent conversation. Make the chosen target visible; clarify only when ambiguity
  would change the wrong task. “Meanwhile” is a strong signal for a new task.
- Prefer short grounded acknowledgments after a task is accepted, followed by
  useful progress in its window. Never preplay “done” or claim a task started
  before the runtime has accepted it. Support “just listen” and “hold updates.”
- Prioritize foreground ASR, turn decisions, conversation inference, and speech
  output over optional worker load. Cap active inference, context, and output
  budgets, and observe queue delay; asyncio alone does not prevent workers from
  saturating the same model server. Use separate capacity only if measurements
  justify it. Queue excess work visibly rather than silently slowing everything.
- Maintain a current task summary in foreground context, with detailed logs and
  artifacts fetched on demand. Route worker questions through the coordinator;
  preserve exactly-once notification state across reconnection.

**Done when:** two workers run while the user asks a third unrelated question;
neither worker is cancelled or loses progress. “For the setup, only prepare the
environment” steers the correct worker. A completion during user speech appears
visually and is announced once at the next suitable gap. Foreground latency under
load is measured against its no-worker baseline.

**Dependencies:** F04/F06/F08; start a minimal coordinator with the first worker.

### F18 — Decouple text, tool dispatch, and speech generation

**Starting point:** the LLM stream consumer awaits `flush_speak` on completed
sentences. `speak` finishes synthesis before sending both the text and audio; a
pre-tool phrase is also synthesized before the tool call is dispatched. Model
generation may proceed remotely, but our consumer can stop reading it during TTS.

- Give token ingestion, visible text, speech synthesis, and playback separate
  bounded queues with a shared response ID and cancellation state.
- Display committed text as it arrives; synthesize suitable short phrases while
  continuing to consume tokens. Preserve natural prosody and avoid reading code,
  raw logs, or unfinished tool arguments. Measure phrase-size tradeoffs with Kokoro.
- Once a complete authorized tool call is ready, dispatch it independently of a
  spoken acknowledgment. Keep speculative content and tool actions gated until
  their relevant input/intent is committed; decoupling must not bypass that gate.
- Track generated text separately from text actually spoken. On barge-in, discard
  stale audio throughout the queues, keep accurate conversation state, and retain
  unrelated tasks. Bound pre-generation to limit wasted work and stop latency.

**Done when:** text and ready tool actions do not wait for unrelated sentence
synthesis. A long spoken response streams smoothly; interruption emits no stale
audio afterward; speculative content stays invisible; the same latency fixtures
show the effect on visible actions, audio onset, and perceived speech quality.

**Dependencies:** F13 measurements and F06 response-versus-task lifetime rules.
Can improve the current pipeline before changing ASR or adding multiple workers.

## The target concurrent session

This is proposed application behavior, not a claim that these repositories were
cloned or inspected during planning.

1. **User:** “Open a code editor, pull prime-rl, and set it up.” The editor appears;
   a named setup task owns clone, environment preparation, and verification. Its
   terminal shows progress while the foreground agent returns to listening.
2. **User, during setup:** “Open a browser to the Slime RL repo and find where it
   handles multimodal media during training.” A second task owns its browser page
   and research plan. Setup continues with its own state and tool results.
3. **User, while both run:** asks an unrelated question or changes window layout.
   The main agent responds and neither task is abandoned.
4. **User:** “For prime-rl, just prepare the environment; don't launch training.”
   The setup task receives that steering. The browser investigation is unaffected.
5. **User:** “Cancel the browser research, keep setup going.” Only the browser
   task and its owned work stop. Its useful partial results remain inspectable.
6. **Setup completes while the user is speaking:** its window shows the actual
   setup/verification result immediately; a brief spoken summary waits for a gap.

Implement the first version with ordinary completed voice turns plus persistent
background tasks. It does not require acting on unfinished speech. Then layer F16
onto the same task system so the first clause can start work sooner. Use controlled
local repository/site fixtures for repeatable tests and separately verify the real
repository setup and research flows when implementing them.

## Suggested delivery order

1. **First useful batch:** F01 layouts and a small F13 behavior baseline. Capture
   current working changes and preserve existing workspace data before migration.
2. **Durable work and control:** F05 workspaces plus F06 tasks/cancellation, with
   F12 startup/recovery checks. Add a minimal F04/F17 worker slice here to prove
   conversation continues during a long setup command. These supply the shared
   contracts for later apps; full multi-step workers expand in batch 4.
3. **Everyday tools:** F02 terminal and F03 browser execution on those contracts;
   add the minimum F07 direct input and F11 file transfer needed to use them well.
4. **Independent workers:** F04, first one worker then bounded concurrency, with
   F08 context and F09 review/checkpoints for its changes. Verify the prime-rl plus
   Slime scenario with F17 routing and attention management.
5. **Continuity and polish:** F10 memory, remaining F07/F11 collaboration features,
   F14 routines/extensions, and F15 model routing guided by measurements.

F13 verification accompanies each batch. Browser harness investigation can happen
before the shared runtime is finished. F18 pipeline decoupling can run as a focused
early improvement. Add F16 partial-transcript/intent previews once the task rules
are defined, and enable mid-speech actions after ordinary-turn concurrency works.
These are dependency recommendations, not time estimates or a requirement for a
wholesale rewrite.

## How to turn an item into an independent coding task

Keep the existing speech-server/client-tool boundary: generic session and task
protocols in the engine, app behavior in RetroVoice OS. Each implementation task
should name one observable behavior, its owning modules, the data/protocol changes,
existing-data migration, cancellation/reconnect behavior where relevant, and the
acceptance scenario above. Land a working vertical slice before adding the next
layer. Check behavior in the UI and report any unverified portions explicitly.

Decisions that can wait for the relevant slice: exact browser harness adapter,
how existing host projects are imported or mounted, terminal/editor components,
the default meaning of spoken “stop,” worker concurrency/budgets, and whether any
worker uses a different model. No feature implementation or runtime change was
performed while preparing this backlog.


### F19 — Multimodal training data collection

**Status:** initial collector implemented September 12. See
[DATA_COLLECTION.md](docs/DATA_COLLECTION.md) for controls, schema, storage,
coverage, limits, labels and integrity checks.

Capture continuous app-input audio, silence, VAD frame scores, successive ASR
attempts with timestamps, classifier decisions, exact model context and streaming
output, proposed/released/executed/cancelled actions, tool results, generated TTS,
playback observations, app state, human interactions and explicit corrections.
Preserve raw evidence separately from labels and keep shared IDs and clocks.

Future extensions: device-level audio timing/raw channels, truly streaming ASR
hypotheses, screenshots/video, file-version diffs, resolved model/tokenizer hashes,
review UI for retrospective annotation and dataset export/split tooling. Do not
infer correctness from a successful tool return or lack of interruption.

## Implemented workspace milestone

F02/F05 and the command-management portion of F06 now have an initial implementation: named host-backed workspaces, interactive PTY terminals, asynchronous command tools, editable files with save conflicts and draft recovery, and a human-operated task manager. Shells survive frontend/backend reloads; container restarts retain files/history and mark interrupted work honestly. F04 autonomous background agents and F03 browser execution harness remain planned.

Window snapping is now implemented: human title-bar menu and agent `snap_window`,
halves/thirds/two-thirds/full desktop, restore previous bounds, responsive layout,
and persistence for saved windows. Browser Use harness (F03) and autonomous
background agents (F04) are still pending.

## Browser harness and worker implementation

F03 and F04 now have initial implementations. The official Browser Use package
runs persistent Python in a per-browser Docker controller; `browser_exec` has
asynchronous task results, deadlines, cancellation, explicit reset generations,
and a human Python console/takeover/input path. Legacy navigation tools remain
compatible and share the controller's selected tab and automation guards.

Background model workers run in a separate local service, survive conversation
interruptions and backend reloads, and have isolated default workspaces/browsers,
steering, input requests, stop/resume, bounded concurrency and model budgets.
Task Manager includes workers; the agent window exposes full controls. Live tests
cover simultaneous coding/browser workers, foreground conversation during work,
steering, workspace ownership, cancellation, and browser controller restart.
Completion is currently a visual notification; automatic spoken completion and
Git worktree merge/diff workflows remain future work.
