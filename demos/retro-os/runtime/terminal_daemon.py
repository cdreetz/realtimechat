"""Container-owned PTYs: browser/backend disconnects never own shell lifetime.

Only the loopback-published backend port can reach this token-authenticated API.
Shell prompt markers establish command exit status; terminal bytes remain bytes.
"""
import base64
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import pty
import re
import secrets
import signal
import struct
import termios
import threading
import time

STATE = Path(os.environ.get("RETRO_STATE", "/state"))
ROOT = os.environ.get("RETRO_WORKDIR", "/workspace")
TOKEN = os.environ.get("RETRO_TOKEN", "")
LIMIT = 1024 * 1024
lock = threading.RLock()
terminals = {}
tasks = {}
dirty = threading.Event()


def persist():
    with lock:
        value = {"version": 1, "tasks": list(tasks.values())[-500:],
                 "terminals": [t.snapshot() for t in terminals.values()]}
    STATE.mkdir(parents=True, exist_ok=True)
    tmp = STATE / "sessions.tmp"
    with open(tmp, "w") as f:
        json.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(STATE / "sessions.json")


class Terminal:
    def __init__(self, terminal_id=None, saved=None):
        self.id = terminal_id or secrets.token_hex(8)
        self.pid = self.fd = None
        self.current = None
        self.ready = False
        self.partial_input = ""
        self.status = "exited" if saved else "starting"
        self.output = base64.b64decode(saved.get("output", "")) if saved else b""
        self.end = saved.get("end", len(self.output)) if saved else 0
        self.exit_code = saved.get("exit_code") if saved else None
        self.marker = secrets.token_hex(16)
        self.pending = b""
        if saved:
            return  # A dead process is never represented as a live shell.
        pid, fd = pty.fork()
        if pid == 0:
            os.chdir(ROOT)
            env = {**os.environ, "TERM": "xterm-256color", "PS1": "\\[\\e[32m\\]workspace\\[\\e[0m\\]:\\w$ ",
                   "PROMPT_COMMAND": f"printf '\\033]777;{self.marker};%s\\007' \"$?\""}
            env.pop("RETRO_TOKEN", None)
            os.execve("/bin/bash", ["bash", "--noprofile", "--norc", "-i"], env)
        self.pid, self.fd = pid, fd
        self.resize(100, 28)
        threading.Thread(target=self.read_loop, daemon=True).start()

    def snapshot(self, cursor=None):
        start = self.end - len(self.output)
        offset = max(start, min(self.end, cursor if cursor is not None else start))
        return {"terminal_id": self.id, "status": self.status, "ready": self.ready,
                "task_id": self.current, "exit_code": self.exit_code,
                "start": offset, "end": self.end,
                "truncated": cursor is not None and cursor < start,
                "output": base64.b64encode(self.output[offset-start:]).decode()}

    def append(self, data):
        self.output = (self.output + data)[-LIMIT:]
        self.end += len(data)
        dirty.set()

    def finish(self, code):
        if self.current:
            task = tasks[self.current]
            task.update(status="cancelled" if task.get("stop_requested") else "completed" if code == 0 else "failed",
                        exit_code=code, finished_at=time.time(), end=self.end)
            self.current = None
        self.ready = True
        self.status = "idle"
        self.partial_input = ""
        dirty.set()

    def read_loop(self):
        prefix = f"\x1b]777;{self.marker};".encode()
        pattern = re.compile(re.escape(prefix) + rb"(\d+)\x07")
        try:
            while True:
                chunk = os.read(self.fd, 8192)
                if not chunk:
                    break
                with lock:
                    self.pending += chunk
                    while True:
                        match = pattern.search(self.pending)
                        if match:
                            self.append(self.pending[:match.start()])
                            self.finish(int(match[1]))
                            self.pending = self.pending[match.end():]
                            continue
                        # Retain only a possibly split marker; don't delay normal output.
                        pos = self.pending.find(prefix)
                        if pos < 0:
                            keep = next((n for n in range(min(len(prefix), len(self.pending)), 0, -1)
                                         if prefix.startswith(self.pending[-n:])), 0)
                            pos = len(self.pending) - keep
                        self.append(self.pending[:pos])
                        self.pending = self.pending[pos:]
                        break
        except OSError:
            pass  # Linux PTYs return EIO when the shell exits.
        finally:
            _, status = os.waitpid(self.pid, 0)
            with lock:
                self.append(self.pending)
                self.exit_code = os.waitstatus_to_exitcode(status)
                self.finish(self.exit_code)
                self.status, self.ready = "exited", False
                os.close(self.fd)
                self.fd = None

    def new_task(self, command, owner):
        if len(tasks) >= 500:
            for key in list(tasks):
                if tasks[key]["status"] not in {"running", "stopping"}:
                    del tasks[key]
                    break
        task_id = secrets.token_hex(8)
        tasks[task_id] = {"task_id": task_id, "terminal_id": self.id, "command": command,
                          "owner": owner, "status": "running", "started_at": time.time(),
                          "start": self.end, "exit_code": None}
        self.current, self.ready, self.status = task_id, False, "running"
        dirty.set()
        return tasks[task_id]

    def write(self, data, owner="human", command=False):
        if self.fd is None:
            raise ValueError("shell exited; open a new terminal")
        if command:
            if not self.ready or self.current or self.partial_input:
                raise ValueError("terminal is busy or has unfinished human input; use a separate terminal or finish that input")
            if "\n" in data or "\r" in data:
                # One shell submission, preserving multiline programs through eval.
                import shlex
                data = "eval " + shlex.quote(data)
            task = self.new_task(data, owner)
            os.write(self.fd, (data + "\n").encode())
            return task
        if self.ready:
            self.partial_input += data
            if "\r" in data or "\n" in data:
                self.new_task(self.partial_input.strip()[:500] or "interactive command", owner)
                self.partial_input = ""
        os.write(self.fd, data.encode())
        return {"terminal_id": self.id, "task_id": self.current}

    def resize(self, cols, rows):
        if self.fd is not None:
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", max(2, min(300, rows)), max(2, min(500, cols)), 0, 0))

    def stop(self, task_id=None):
        if task_id and task_id != self.current:
            return
        if self.fd is None:
            return
        task_id = self.current
        if task_id:
            tasks[task_id].update(stop_requested=True, status="stopping")
        group = os.tcgetpgrp(self.fd)
        try:
            os.killpg(group, signal.SIGINT)
        except ProcessLookupError:
            pass
        def escalate():
            with lock:
                if task_id and self.current == task_id and self.fd is not None:
                    try:
                        if os.tcgetpgrp(self.fd) == group:
                            os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        timer = threading.Timer(2, escalate)
        timer.daemon = True
        timer.start()
        dirty.set()


def dispatch(path, data):
    with lock:
        if path == "/state":
            return {"terminals": [{k: v for k, v in t.snapshot(t.end).items() if k != "output"} for t in terminals.values()],
                    "tasks": list(tasks.values()), "persistence_error": persistence_error}
        if path == "/open":
            if sum(t.fd is not None for t in terminals.values()) >= 16:
                raise ValueError("workspace already has 16 live terminals")
            terminal = Terminal()
            terminals[terminal.id] = terminal
            dirty.set()
            return terminal.snapshot()
        terminal = terminals.get(data.get("terminal_id"))
        if not terminal:
            raise KeyError("unknown terminal")
        if path == "/read":
            return terminal.snapshot(int(data.get("cursor", 0)))
        if path == "/write":
            if len(data.get("data", "")) > 65536:
                raise ValueError("input too large")
            return terminal.write(str(data.get("data", "")), data.get("owner", "human"), data.get("command", False))
        if path == "/resize":
            terminal.resize(int(data.get("cols", 100)), int(data.get("rows", 28)))
        elif path == "/stop":
            terminal.stop(data.get("task_id"))
        else:
            raise KeyError("unknown operation")
        return terminal.snapshot(terminal.end)


persistence_error = None
def save_loop():
    global persistence_error
    while True:
        dirty.wait()
        time.sleep(.5)
        dirty.clear()
        try:
            persist()
            persistence_error = None
        except OSError as exc:
            persistence_error = str(exc)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not TOKEN or self.headers.get("Authorization") != "Bearer " + TOKEN:
            self.send_error(403)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 100000:
                raise ValueError("request too large")
            data = json.loads(self.rfile.read(size) or b"{}")
            result = dispatch(self.path, data)
            body, code = json.dumps(result).encode(), 200
        except (KeyError, ValueError, TypeError) as exc:
            body, code = json.dumps({"detail": str(exc)}).encode(), 409
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    try:
        saved = json.loads((STATE / "sessions.json").read_text())
        for t in saved.get("tasks", []):
            if t["status"] in {"running", "stopping"}:
                t.update(status="interrupted", finished_at=time.time())
            tasks[t["task_id"]] = t
        for t in saved.get("terminals", []):
            terminals[t["terminal_id"]] = Terminal(t["terminal_id"], saved=t)
    except FileNotFoundError:
        pass
    threading.Thread(target=save_loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", 8765), Handler).serve_forever()
