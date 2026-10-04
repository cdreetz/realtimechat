let currentWorkspaceId = null;
const workspaceCache = new Map();
const workspaceStates = new Map();

function actionButton(label, action) {
  const button = document.createElement("button");
  button.textContent = label;
  button.onclick = async () => {
    button.disabled = true;
    try { await action(); }
    catch (e) { feed("sys", `* ${e.message}`); }
    finally { button.disabled = false; }
  };
  return button;
}

async function chooseWorkspace(workspaceId) {
  const id = workspaceId || currentWorkspaceId;
  const w = id ? await api("POST", `/api/workspaces/${id}/open`)
               : await api("POST", "/api/workspaces", {name: "My workspace"});
  workspaceCache.set(w.workspace_id, w);
  currentWorkspaceId = w.workspace_id;
  scheduleDesktopSave();
  return w;
}

async function workspaceForSandbox(sandboxId) {
  const state = await api("GET", "/api/workspaces");
  const w = state.workspaces.find(w => w.container === sandboxId || w.legacy_sandbox === sandboxId);
  const result = w ? await chooseWorkspace(w.workspace_id)
                   : await api("POST", "/api/workspaces", {name: `Imported ${sandboxId.replace('retrovoice-sbx-', '')}`, legacy_sandbox: sandboxId});
  workspaceCache.set(result.workspace_id, result);
  return result;
}

async function workspaceEditor(workspaceId) {
  const w = await chooseWorkspace(workspaceId);
  const id = nextWindowId++;
  const ed = openEditorWindow(id, w.container, w.workspace_id);
  ed.win.el.querySelector(".tname").textContent = `Editor — ${w.name}`;
  await refreshTree(ed);
  return ed;
}

async function workspaceTerminal(workspaceId, terminalId, windowId) {
  const w = await chooseWorkspace(workspaceId);
  if (terminalId) {
    const existing = Object.values(windows).find(v => v.terminalId === terminalId && v.workspaceId === w.workspace_id);
    if (existing) { existing.win.restore(); return existing; }
  }
  const opened = terminalId ? await api("GET", `/api/workspaces/${w.workspace_id}/terminals/${terminalId}`)
                            : await api("POST", `/api/workspaces/${w.workspace_id}/terminals`);
  const id = windowId || nextWindowId++;
  nextWindowId = Math.max(nextWindowId, id + 1);
  let timer, resizeObserver, closed = false;
  const term = new Terminal({cursorBlink: true, fontSize: 13, scrollback: 5000,
    theme: {background: "#111820", foreground: "#e5e9ef"}});
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  const win = makeWindow({title: `Terminal — ${w.name}`, icon: "⌨️", x: 55, y: 65,
    w: Math.round(desktop.clientWidth * .62), h: Math.round(desktop.clientHeight * .64),
    onClose: () => { closed = true; clearTimeout(timer); resizeObserver?.disconnect(); term.dispose(); delete windows[id]; }});
  const bar = document.createElement("div"); bar.className = "workspace-toolbar";
  const status = document.createElement("span"); status.textContent = "connecting…";
  const terminal = {id, kind: "terminal", title: `Terminal — ${w.name}`, win, term, status,
    workspaceId: w.workspace_id, terminalId: opened.terminal_id, cursor: 0, pendingInput: Promise.resolve()};
  const sendInput = (data, command = false, owner = "human") => {
    terminal.pendingInput = terminal.pendingInput.catch(() => {}).then(() => api("POST",
      `/api/workspaces/${terminal.workspaceId}/terminals/${terminal.terminalId}/input`, {data, command, owner}));
    return terminal.pendingInput;
  };
  terminal.sendInput = sendInput;
  const stop = actionButton("Stop / Ctrl-C", () => stopTerminal(terminal));
  bar.append(status, stop, actionButton("Tasks", () => openTaskManager()),
    actionButton("New shell", () => workspaceTerminal(w.workspace_id)));
  const host = document.createElement("div"); host.className = "terminal-host";
  win.body.append(bar, host);
  windows[id] = terminal;
  term.open(host);
  term.onData(data => {
    training.event("terminal_input", {workspace_id: w.workspace_id, terminal_id: terminal.terminalId, data, owner: "human"});
    sendInput(data).catch(e => { status.textContent = e.message; });
  });
  term.onResize(size => api("POST", `/api/workspaces/${w.workspace_id}/terminals/${terminal.terminalId}/resize`, size).catch(() => {}));
  resizeObserver = new ResizeObserver(() => { if (!closed && !win.minimized) fit.fit(); });
  resizeObserver.observe(host);
  fit.fit();
  const poll = async () => {
    try {
      const s = await api("GET", `/api/workspaces/${w.workspace_id}/terminals/${terminal.terminalId}?cursor=${terminal.cursor}`);
      if (closed) return;
      if (s.truncated) term.write("\r\n[Earlier output is no longer retained]\r\n");
      if (s.output) {
        const bytes = Uint8Array.from(atob(s.output), ch => ch.charCodeAt(0));
        await new Promise(resolve => term.write(bytes, resolve));
        training.event("terminal_output", {workspace_id: w.workspace_id, terminal_id: terminal.terminalId,
          start: s.start, end: s.end, data: s.output, encoding: "base64"});
      }
      terminal.cursor = s.end;
      terminal.ready = s.ready;
      terminal.taskId = s.task_id;
      status.textContent = `${s.status}${s.task_id ? ' · task ' + s.task_id.slice(0,8) : ''}${s.status === 'exited' ? ' · open New shell to continue' : ''}`;
    } catch (e) { if (!closed) status.textContent = `${e.message} — retrying`; }
    if (!closed) timer = setTimeout(poll, win.minimized ? 1000 : 180);
  };
  poll();
  scheduleDesktopSave();
  return terminal;
}

async function stopTerminal(t) {
  await api("POST", `/api/workspaces/${t.workspaceId}/terminals/${t.terminalId}/stop`);
  training.event("terminal_stop", {workspace_id: t.workspaceId, terminal_id: t.terminalId});
}

async function startTerminalCommand(t, command, owner = "agent") {
  const deadline = performance.now() + 2000;
  while (!t.ready && performance.now() < deadline) await new Promise(r => setTimeout(r, 75));
  const task = await t.sendInput(String(command), true, owner);
  training.event("task_started", {workspace_id: t.workspaceId, ...task});
  return {workspace_id: t.workspaceId, window_id: t.id, ...task};
}

function openWorkspaceManager() {
  const existing = Object.values(windows).find(w => w.kind === "workspaces");
  if (existing) { existing.win.restore(); return [existing.id, "Workspaces opened"]; }
  const id = nextWindowId++;
  const win = makeWindow({title: "Workspaces", icon: "🗂️", x: 30, y: 30, w: 540, h: 420,
    onClose: () => { delete windows[id]; }});
  windows[id] = {id, kind: "workspaces", title: "Workspaces", win};
  const bar = document.createElement("div"); bar.className = "workspace-toolbar";
  const name = document.createElement("input"); name.placeholder = "New workspace name"; name.setAttribute("aria-label", "New workspace name");
  const list = document.createElement("div"); list.className = "workspace-list";
  const refresh = async () => {
    const state = await api("GET", "/api/workspaces");
    list.replaceChildren();
    for (const w of state.workspaces) {
      workspaceCache.set(w.workspace_id, w);
      const row = document.createElement("div"); row.className = "workspace-row";
      const title = document.createElement("input"); title.value = w.name; title.setAttribute("aria-label", `Name of ${w.name}`);
      row.append(title, actionButton("Rename", async () => { await api("PATCH", `/api/workspaces/${w.workspace_id}`, {name: title.value}); await refresh(); }),
        actionButton("Editor", () => workspaceEditor(w.workspace_id)),
        actionButton("Terminal", () => workspaceTerminal(w.workspace_id)));
      const details = document.createElement("div"); details.textContent = `Files saved locally · ${w.workspace_id}`;
      row.append(details); list.append(row);
    }
    for (const sid of state.legacy_sandboxes) {
      const row = document.createElement("div"); row.className = "workspace-row";
      row.textContent = `Existing sandbox ${sid.replace('retrovoice-sbx-', '')} `;
      row.append(actionButton("Import safely", async () => { await workspaceForSandbox(sid); await refresh(); }));
      list.append(row);
    }
  };
  bar.append(name, actionButton("Create", async () => {
    const w = await api("POST", "/api/workspaces", {name: name.value || "Untitled workspace"});
    currentWorkspaceId = w.workspace_id; name.value = ""; await refresh();
  }), actionButton("Refresh", refresh));
  win.body.append(bar, list);
  refresh().catch(e => { list.textContent = e.message; });
  return [id, "Workspaces opened. Create or reopen a workspace; closing its views preserves files."];
}

function openTaskManager() {
  const existing = Object.values(windows).find(w => w.kind === "task_manager");
  if (existing) { existing.win.restore(); return [existing.id, "Task manager opened"]; }
  const id = nextWindowId++;
  let timer, closed = false;
  const win = makeWindow({title: "Task Manager", icon: "📋", x: 95, y: 50, w: 600, h: 420,
    onClose: () => { closed = true; clearTimeout(timer); delete windows[id]; }});
  windows[id] = {id, kind: "task_manager", title: "Task Manager", win};
  win.body.append(actionButton("Background agents", () => openAgents()));
  const list = document.createElement("div"); list.className = "workspace-list"; win.body.append(list);
  const refresh = async () => {
    const workspaces = await api("GET", "/api/workspaces");
    const fragments = [];
    const agents = await api("GET", "/api/agents");
    for (const job of agents.tasks) {
      const row = document.createElement("div"); row.className = "workspace-row";
      row.textContent = `Agent · ${job.status} · ${job.name} `;
      row.append(actionButton("Details / steer", () => openAgents()));
      if (["queued", "running", "paused"].includes(job.status)) row.append(actionButton("Stop agent", () => api("POST", `/api/agents/${job.task_id}/stop`)));
      fragments.push(row);
    }
    for (const w of workspaces.workspaces) {
      const group = document.createElement("div"); group.className = "workspace-row";
      const title = document.createElement("strong"); title.textContent = w.name; group.append(title);
      try {
        const state = await api("GET", `/api/workspaces/${w.workspace_id}/state`);
        if (state.persistence_error) group.append(document.createTextNode(` · Persistence error: ${state.persistence_error}`));
        for (const t of state.terminals) {
          const row = document.createElement("div"); row.textContent = `Shell ${t.terminal_id.slice(0,8)} · ${t.status} `;
          row.append(actionButton("Show terminal", () => workspaceTerminal(w.workspace_id, t.terminal_id)));
          group.append(row);
        }
        for (const task of [...state.tasks].reverse().slice(0, 50)) {
          const row = document.createElement("div"); row.className = "task-row";
          const text = document.createElement("span"); text.textContent = `${task.status}${task.exit_code != null ? ' (exit '+task.exit_code+')' : ''} · ${task.command}`;
          row.append(text);
          if (["running", "stopping"].includes(task.status)) row.append(actionButton("Stop", async () => {
            await api("POST", `/api/workspaces/${w.workspace_id}/tasks/${task.task_id}/stop`);
          }));
          group.append(row);
        }
      } catch (e) { group.append(document.createTextNode(` · ${e.message}`)); }
      fragments.push(group);
    }
    if (!closed) {
      list.replaceChildren(...fragments);
      if (!fragments.length) list.textContent = "No workspaces yet. Open Workspaces to create one.";
      timer = setTimeout(() => refresh().catch(e => { list.textContent = e.message; }), 1500);
    }
  };
  refresh().catch(e => { list.textContent = e.message; });
  return [id, "Task manager opened. Stop commands separately from closing their windows."];
}

const workspaceTools = {
  async list_workspaces() { return JSON.stringify(await api("GET", "/api/workspaces")); },
  async create_workspace({name}) {
    const w = await api("POST", "/api/workspaces", {name});
    currentWorkspaceId = w.workspace_id; workspaceCache.set(w.workspace_id, w);
    return JSON.stringify(w);
  },
  async open_workspace({workspace_id, view = "terminal"}) {
    const w = view === "code_editor" ? await workspaceEditor(workspace_id) : await workspaceTerminal(workspace_id);
    return JSON.stringify({workspace_id: w.workspaceId, window_id: w.id, terminal_id: w.terminalId});
  },
  async terminal_run({window_id, command}) {
    const [t, err] = requireWindow(window_id, "terminal");
    return err || JSON.stringify(await startTerminalCommand(t, command));
  },
  async terminal_input({window_id, text}) {
    const [t, err] = requireWindow(window_id, "terminal");
    return err || JSON.stringify(await t.sendInput(text, false, "agent"));
  },
  async read_terminal({window_id, cursor = 0}) {
    const [t, err] = requireWindow(window_id, "terminal"); if (err) return err;
    const r = await api("GET", `/api/workspaces/${t.workspaceId}/terminals/${t.terminalId}?cursor=${cursor}`);
    const bytes = Uint8Array.from(atob(r.output), c => c.charCodeAt(0));
    return JSON.stringify({...r, output: new TextDecoder().decode(bytes).slice(-12000)});
  },
  async list_tasks({workspace_id}) { return JSON.stringify(await api("GET", `/api/workspaces/${workspace_id}/state`)); },
  async stop_task({workspace_id, task_id}) { return JSON.stringify(await api("POST", `/api/workspaces/${workspace_id}/tasks/${task_id}/stop`)); },
};

function workspaceToolSchemas() {
  const string = {type: "string"}, integer = {type: "integer"};
  const spec = (name, description, properties, required = Object.keys(properties)) => ({type: "function", function: {name, description,
    parameters: {type: "object", properties, required}}});
  return [
    spec("list_workspaces", "List durable workspaces and unimported legacy sandboxes.", {}),
    spec("create_workspace", "Create a named persistent workspace. Files survive closed windows and runtime replacement.", {name: string}),
    spec("open_workspace", "Open an existing workspace in a terminal or editable code view; returns window_id.", {workspace_id: string, view: {type: "string", enum: ["terminal", "code_editor"]}}, ["workspace_id"]),
    spec("terminal_run", "Submit one command to an idle persistent shell. Returns a task ID immediately. Use list_tasks/read_terminal for status and output; this is not an autonomous background agent. cwd/environment persist. Busy shells reject new commands.", {window_id: integer, command: string}),
    spec("terminal_input", "Send literal input to the user's terminal, including prompts. Include newline to submit; use read_terminal to understand its current state first.", {window_id: integer, text: string}),
    spec("read_terminal", "Read terminal output and execution status. cursor is a byte offset returned as end; output is bounded.", {window_id: integer, cursor: integer}, ["window_id"]),
    spec("list_tasks", "Get workspace shell and command states, exit codes and task IDs. Commands survive view closure/backend reload; tasks marked interrupted were lost on runtime restart.", {workspace_id: string}),
    spec("stop_task", "Request interruption of one running command; escalate if it ignores Ctrl-C. Check list_tasks for its settled state. Does not undo completed changes.", {workspace_id: string, task_id: string}),
  ];
}
