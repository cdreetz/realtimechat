#!/usr/bin/env python3
"""RetroVoice OS demo backend.

Serves the demo page and manages Docker containers backing code-editor and
headless-Chromium windows. Workspaces keep host-backed files and container-owned
PTYs; closing their views preserves work. Browser windows use CDP screenshots
and retain their existing close-to-destroy behavior.

Run:  python backend.py   (needs Docker running; serves http://localhost:8080)
"""
import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
import httpx
import websockets
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from workspaces import WorkspaceManager
from browser_execution import BrowserExecution
from worker_bridge import router as worker_router, ensure_workers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retro-backend")

IMAGE = "python:3.11-slim"
LABEL = "retrovoice-demo=1"
POOL_SIZE = 2
WORKDIR = "/workspace"
EXEC_TIMEOUT_S = 30
MAX_OUTPUT = 8000
BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR / ".sandboxes.json"
BROWSER_STATE_PATH = BASE_DIR / ".browsers.json"
BROWSER_IMAGE = "chromedp/headless-shell:latest"
BROWSER_LABEL = "retrovoice-browser=1"
DOCKER_BIN = (os.environ.get("DOCKER_BIN") or shutil.which("docker")
              or "/usr/local/bin/docker")


async def sh_bytes(*args, stdin: bytes = None, timeout: float = 120):
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE if stdin is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise HTTPException(504, "command timed out")
    return proc.returncode, out, err.decode(errors="replace")


async def sh(*args, stdin: bytes = None, timeout: float = 120):
    rc, out, err = await sh_bytes(*args, stdin=stdin, timeout=timeout)
    return rc, out.decode(errors="replace"), err


def safe_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        raise HTTPException(400, "empty path")
    if not re.fullmatch(r"[A-Za-z0-9._/ +-]+", p):
        raise HTTPException(400, "path contains unsupported characters")
    if not p.startswith("/"):
        p = f"{WORKDIR}/{p}"
    norm = Path(p)
    if ".." in norm.parts:
        raise HTTPException(400, "path may not contain ..")
    workspace = Path(WORKDIR)
    if norm != workspace and workspace not in norm.parents:
        raise HTTPException(400, f"paths must live under {WORKDIR}")
    return str(norm)


class SandboxPool:
    def __init__(self):
        self.warm: list[str] = []
        self.active: set[str] = set()
        self.lock = asyncio.Lock()

    def _save_state(self):
        data = {"warm": self.warm, "active": sorted(self.active)}
        temp = STATE_PATH.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp.replace(STATE_PATH)

    def _load_state(self) -> tuple[list[str], set[str]]:
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            return list(data.get("warm") or []), set(data.get("active") or [])
        except (OSError, json.JSONDecodeError, TypeError):
            return [], set()

    async def start(self):
        rc, out, _ = await sh(
            DOCKER_BIN, "ps", "--format", "{{.Names}}",
            "--filter", f"label={LABEL}", "--filter", "status=running")
        existing = set(out.split()) if rc == 0 else set()
        saved_warm, saved_active = self._load_state()
        self.warm = [name for name in saved_warm if name in existing]
        self.active = {name for name in saved_active if name in existing}
        claimed = set(self.warm) | self.active
        unclaimed = existing - claimed
        # A crash or the first reload-safe launch has no complete state file.
        # Preserve unknown containers as active so open browser windows can
        # immediately reattach instead of losing their work.
        self.active.update(unclaimed)
        if existing:
            log.info(
                f"reattached {len(self.active)} active and {len(self.warm)} "
                f"warm sandbox(es)")
        rc, _, _ = await sh(DOCKER_BIN, "image", "inspect", IMAGE)
        if rc != 0:
            log.info(f"pulling {IMAGE}...")
            rc, _, err = await sh(DOCKER_BIN, "pull", IMAGE, timeout=600)
            if rc != 0:
                raise RuntimeError(f"could not pull {IMAGE}: {err}")
        await self.refill()
        self._save_state()
        log.info(f"warm pool ready ({len(self.warm)} sandbox(es))")

    async def _spawn(self) -> str:
        name = f"retrovoice-sbx-{secrets.token_hex(4)}"
        rc, _, err = await sh(
            DOCKER_BIN, "run", "-d", "--rm", "--label", LABEL,
            "--memory", "1g", "--cpus", "2", "-w", WORKDIR,
            # container port 8000 -> ephemeral host port, for app previews
            "-p", "127.0.0.1:0:8000",
            "--name", name, IMAGE, "sleep", "infinity")
        if rc != 0:
            raise HTTPException(500, f"docker run failed: {err.strip()}")
        return name

    async def refill(self):
        async with self.lock:
            while len(self.warm) < POOL_SIZE:
                self.warm.append(await self._spawn())
            self._save_state()

    async def acquire(self) -> str:
        async with self.lock:
            name = self.warm.pop(0) if self.warm else await self._spawn()
            self.active.add(name)
            self._save_state()
        asyncio.create_task(self.refill())
        log.info(f"sandbox acquired: {name}")
        return name

    async def release(self, name: str):
        async with self.lock:
            self.active.discard(name)
            if name in self.warm:
                self.warm.remove(name)
            await sh(DOCKER_BIN, "rm", "-f", name)
            self._save_state()
        log.info(f"sandbox released: {name}")

    async def shutdown(self):
        # Uvicorn reloads run the lifespan shutdown hook. Containers are
        # deliberately preserved and reattached by the next worker.
        self._save_state()
        log.info(
            f"preserved {len(self.active)} active and {len(self.warm)} "
            f"warm sandbox(es) for reload")

    def require(self, name: str) -> str:
        managed = workspaces.container_for(name)
        if managed:
            return managed
        if name not in self.active:
            raise HTTPException(404, f"unknown sandbox {name!r}")
        return name


pool = SandboxPool()
workspaces = WorkspaceManager(BASE_DIR, DOCKER_BIN, sh, pool)


class BrowserManager:
    """Own headless Chromium containers and expose a small CDP surface."""

    def __init__(self):
        self.sessions: dict[str, dict] = {}
        self.lock = asyncio.Lock()

    def _save_state(self):
        data = {"sessions": self.sessions}
        temp = BROWSER_STATE_PATH.with_suffix(".json.tmp")
        temp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temp.replace(BROWSER_STATE_PATH)

    async def _http_json(self, port: int, path: str):
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=3) as client:
                response = await client.get(f"http://127.0.0.1:{port}{path}")
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            raise HTTPException(502, f"Chromium CDP is unavailable: {exc}")

    async def start(self):
        try:
            saved = json.loads(BROWSER_STATE_PATH.read_text(encoding="utf-8"))
            candidates = saved.get("sessions") or {}
        except (OSError, json.JSONDecodeError, TypeError):
            candidates = {}
        rc, out, _ = await sh(
            DOCKER_BIN, "ps", "--format", "{{.Names}}",
            "--filter", f"label={BROWSER_LABEL}", "--filter", "status=running")
        existing = set(out.split()) if rc == 0 else set()
        for browser_id, info in candidates.items():
            container = str(info.get("container") or "")
            if container not in existing:
                continue
            try:
                port = int(info["port"])
                await self._http_json(port, "/json/version")
            except Exception:
                continue
            self.sessions[browser_id] = {"port": port, "container": container}
        claimed = {info["container"] for info in self.sessions.values()}
        for container in existing - claimed:
            if not container.startswith("retrovoice-browser-"):
                continue
            browser_id = container.removeprefix("retrovoice-browser-")
            rc, out, _ = await sh(DOCKER_BIN, "port", container, "9222/tcp")
            try:
                port = int(out.strip().splitlines()[0].rsplit(":", 1)[1])
                await self._http_json(port, "/json/version")
            except (IndexError, ValueError, HTTPException):
                continue
            self.sessions[browser_id] = {"port": port, "container": container}
        rc, _, _ = await sh(DOCKER_BIN, "image", "inspect", BROWSER_IMAGE)
        if rc != 0:
            log.info(f"pulling {BROWSER_IMAGE}...")
            rc, _, err = await sh(DOCKER_BIN, "pull", BROWSER_IMAGE, timeout=600)
            if rc != 0:
                raise RuntimeError(f"could not pull {BROWSER_IMAGE}: {err}")
        self._save_state()
        if self.sessions:
            log.info(f"reattached {len(self.sessions)} Chromium session(s)")

    async def shutdown(self):
        # Chrome sessions deliberately survive Uvicorn reloads.
        self._save_state()

    def require(self, browser_id: str) -> dict:
        session = self.sessions.get(browser_id)
        if not session:
            raise HTTPException(404, f"unknown browser session {browser_id!r}")
        return session

    async def create(self, url: str) -> tuple[str, dict]:
        browser_id = secrets.token_hex(6)
        container = f"retrovoice-browser-{browser_id}"
        rc, _, err = await sh(
            DOCKER_BIN, "run", "-d", "--rm", "--label", BROWSER_LABEL,
            "--memory", "1g", "--cpus", "2", "--shm-size", "256m",
            "-p", "127.0.0.1:0:9222", "--name", container,
            BROWSER_IMAGE,
            "--no-sandbox", "--disable-gpu", "--hide-scrollbars",
            "--window-size=1280,800",
            "about:blank")
        if rc != 0:
            raise HTTPException(500, f"could not start Chromium container: {err.strip()}")
        rc, out, err = await sh(DOCKER_BIN, "port", container, "9222/tcp")
        try:
            port = int(out.strip().splitlines()[0].rsplit(":", 1)[1])
        except (IndexError, ValueError):
            await sh(DOCKER_BIN, "rm", "-f", container)
            raise HTTPException(502, f"Chromium CDP port was not published: {err.strip()}")
        ready = False
        for _ in range(100):
            try:
                await self._http_json(port, "/json/version")
                ready = True
                break
            except HTTPException:
                await asyncio.sleep(0.1)
        if not ready:
            await sh(DOCKER_BIN, "rm", "-f", container)
            raise HTTPException(502, "Chromium did not expose its DevTools port")
        async with self.lock:
            self.sessions[browser_id] = {"port": port, "container": container}
            self._save_state()
        state = await self.navigate(browser_id, url)
        log.info(f"Chromium session opened: {browser_id} on CDP port {port}")
        return browser_id, state

    async def _target_ws(self, browser_id: str) -> str:
        session = self.require(browser_id)
        targets = await self._http_json(session["port"], "/json/list")
        pages = [target for target in targets
                 if target.get("type") == "page" and target.get("webSocketDebuggerUrl")]
        if not pages:
            raise HTTPException(502, "Chromium session has no page target")
        target_id = await browser_execution.target(browser_id)
        page = next((p for p in pages if p.get("id") == target_id), pages[0])
        return page["webSocketDebuggerUrl"]

    async def _browser_ws(self, browser_id: str) -> str:
        session = self.require(browser_id)
        version = await self._http_json(session["port"], "/json/version")
        try:
            return version["webSocketDebuggerUrl"]
        except KeyError:
            raise HTTPException(502, "Chromium did not return a browser target")

    async def cdp(self, browser_id: str, method: str, params: dict = None,
                  *, browser_target: bool = False, before: tuple = ()):
        ws_url = (await self._browser_ws(browser_id) if browser_target
                  else await self._target_ws(browser_id))
        try:
            async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
                for call_id, (command, arguments) in enumerate((*before, (method, params or {})), 1):
                    await ws.send(json.dumps({"id": call_id, "method": command, "params": arguments}))
                    while True:
                        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                        if message.get("id") != call_id:
                            continue
                        if "error" in message:
                            raise HTTPException(502, message["error"].get("message", "CDP error"))
                        result = message.get("result") or {}
                        break
                return result
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"Chromium CDP command failed: {exc}")

    async def evaluate(self, browser_id: str, expression: str):
        result = await self.cdp(browser_id, "Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True,
            "userGesture": True,
        })
        if result.get("exceptionDetails"):
            detail = result["exceptionDetails"].get("text", "JavaScript evaluation failed")
            raise HTTPException(502, detail)
        return result.get("result", {}).get("value")

    async def state(self, browser_id: str) -> dict:
        value = await self.evaluate(browser_id, "({title: document.title, url: location.href})")
        return {"browser_id": browser_id, **(value or {})}

    async def navigate(self, browser_id: str, url: str) -> dict:
        await self.cdp(browser_id, "Page.navigate", {"url": url})
        for _ in range(40):
            await asyncio.sleep(0.2)
            try:
                ready = await self.evaluate(browser_id, "document.readyState")
                if ready in {"interactive", "complete"}:
                    break
            except HTTPException:
                pass
        return await self.state(browser_id)

    async def inspect(self, browser_id: str) -> dict:
        expression = r"""(() => {
          document.querySelectorAll('[data-retro-agent-id]').forEach(
            el => el.removeAttribute('data-retro-agent-id'));
          const visible = el => {
            const r = el.getBoundingClientRect(), s = getComputedStyle(el);
            return r.width > 0 && r.height > 0 && s.visibility !== 'hidden' && s.display !== 'none';
          };
          const nodes = [...document.querySelectorAll(
            'a,button,input,textarea,select,[role="button"],[contenteditable="true"]')]
            .filter(visible).slice(0, 100);
          const elements = nodes.map((el, i) => {
            const id = `e${i}`; el.setAttribute('data-retro-agent-id', id);
            return {id, tag: el.tagName.toLowerCase(), type: el.type || '',
              text: (el.innerText || el.value || el.getAttribute('aria-label') ||
                     el.getAttribute('title') || '').trim().slice(0, 300),
              href: el.href || '', name: el.name || '', disabled: !!el.disabled};
          });
          return {title: document.title, url: location.href,
            text: (document.body?.innerText || '').trim().slice(0, 12000), elements};
        })()"""
        value = await self.evaluate(browser_id, expression)
        return {"browser_id": browser_id, **(value or {})}

    async def screenshot(self, browser_id: str, width: int = 1280, height: int = 800) -> bytes:
        result = await self.cdp(browser_id, "Page.captureScreenshot", {
            "format": "jpeg", "quality": 70, "fromSurface": True,
            "captureBeyondViewport": False,
        }, before=(("Emulation.setDeviceMetricsOverride", {
            "width": width, "height": height, "deviceScaleFactor": 1, "mobile": False,
        }),))
        try:
            return base64.b64decode(result["data"])
        except (KeyError, ValueError):
            raise HTTPException(502, "Chromium did not return a screenshot")

    async def click(self, browser_id: str, element_id: str) -> dict:
        expression = f"""(() => {{
          const el = document.querySelector('[data-retro-agent-id={json.dumps(element_id)}]');
          if (!el) return {{ok:false, error:'element not found; inspect the page again'}};
          el.scrollIntoView({{block:'center'}}); el.click(); return {{ok:true}};
        }})()"""
        result = await self.evaluate(browser_id, expression)
        if not result or not result.get("ok"):
            raise HTTPException(409, (result or {}).get("error", "element not found"))
        await asyncio.sleep(0.4)
        return await self.state(browser_id)

    async def type_text(self, browser_id: str, element_id: str,
                        text: str, press_enter: bool) -> dict:
        expression = f"""(() => {{
          const el = document.querySelector('[data-retro-agent-id={json.dumps(element_id)}]');
          if (!el) return {{ok:false, error:'element not found; inspect the page again'}};
          el.scrollIntoView({{block:'center'}}); el.focus();
          const proto = el instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype :
            (el instanceof HTMLInputElement ? HTMLInputElement.prototype : null);
          const setter = proto && Object.getOwnPropertyDescriptor(proto, 'value')?.set;
          if (setter) setter.call(el, {json.dumps(text)}); else el.textContent = {json.dumps(text)};
          el.dispatchEvent(new Event('input', {{bubbles:true}}));
          el.dispatchEvent(new Event('change', {{bubbles:true}}));
          return {{ok:true}};
        }})()"""
        result = await self.evaluate(browser_id, expression)
        if not result or not result.get("ok"):
            raise HTTPException(409, (result or {}).get("error", "element not found"))
        if press_enter:
            for event_type in ("keyDown", "char", "keyUp"):
                await self.cdp(browser_id, "Input.dispatchKeyEvent", {
                    "type": event_type, "key": "Enter", "code": "Enter",
                    "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13,
                    **({"text": "\r"} if event_type == "char" else {}),
                })
            await asyncio.sleep(0.5)
        return await self.state(browser_id)

    async def close(self, browser_id: str):
        session = self.require(browser_id)
        try:
            await self.cdp(browser_id, "Browser.close", browser_target=True)
        except HTTPException:
            pass
        await sh(DOCKER_BIN, "rm", "-f", session["container"])
        async with self.lock:
            self.sessions.pop(browser_id, None)
            self._save_state()
        log.info(f"Chromium session closed: {browser_id}")


browsers = BrowserManager()
browser_execution = BrowserExecution(browsers, DOCKER_BIN, sh, BASE_DIR)


@asynccontextmanager
async def lifespan(app):
    await pool.start()
    workspaces.load()
    await browsers.start()
    await ensure_workers()
    yield
    await browsers.shutdown()
    await pool.shutdown()


app = FastAPI(lifespan=lifespan)
app.include_router(workspaces.router)
app.include_router(browser_execution.router)
app.include_router(worker_router)
app.mount("/vendor", StaticFiles(directory=BASE_DIR / "vendor"), name="vendor")


class WriteReq(BaseModel):
    path: str
    content: str
    expected_content: str | None = None
    expected_absent: bool = False


class EditReq(BaseModel):
    path: str
    old_string: str
    new_string: str


class ExecReq(BaseModel):
    command: str
    timeout: int = EXEC_TIMEOUT_S


class BrowserOpenReq(BaseModel):
    url: str = "about:blank"


class BrowserNavigateReq(BaseModel):
    url: str
    owner: str = "agent"


class BrowserElementReq(BaseModel):
    element_id: str
    owner: str = "agent"


class BrowserTypeReq(BrowserElementReq):
    text: str
    press_enter: bool = False


def browser_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return "about:blank"
    if "://" not in value and value != "about:blank":
        value = "https://" + value
    if not (value.startswith("http://") or value.startswith("https://")
            or value == "about:blank"):
        raise HTTPException(400, "browser URLs must use http or https")
    if len(value) > 4000:
        raise HTTPException(400, "browser URL is too long")
    return value


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html",
                        headers={"Cache-Control": "no-store"})


@app.get("/recording-client.js")
async def recording_client():
    return FileResponse(BASE_DIR.parent.parent / "server/static/recording-client.js",
                        headers={"Cache-Control": "no-store"})


@app.get("/advanced-ui.js")
async def advanced_ui():
    return FileResponse(BASE_DIR / "advanced-ui.js", headers={"Cache-Control": "no-store"})


@app.get("/workspace-ui.js")
async def workspace_ui():
    return FileResponse(BASE_DIR / "workspace-ui.js", headers={"Cache-Control": "no-store"})


@app.get("/api/dev/version")
async def frontend_version():
    return {"version": max(path.stat().st_mtime_ns for path in (
        BASE_DIR / "index.html", BASE_DIR / "workspace-ui.js", BASE_DIR / "advanced-ui.js",
        BASE_DIR.parent.parent / "server/static/recording-client.js"))}


@app.post("/api/sandbox")
async def sandbox_acquire():
    workspace = await workspaces.create("Untitled workspace")
    return {"sandbox_id": workspace["container"], **workspace}


@app.delete("/api/sandbox/{sid}")
async def sandbox_release(sid: str):
    pool.require(sid)
    if not workspaces.container_for(sid):
        await workspaces.create(f"Imported {sid.removeprefix('retrovoice-sbx-')}", sid)
    return {"ok": True, "preserved": True}


@app.get("/api/sandboxes")
async def sandbox_state():
    return {"active": sorted(pool.active), "warm": list(pool.warm),
            "reload_safe": True}


@app.get("/api/browsers")
async def browser_sessions():
    states = []
    for browser_id in list(browsers.sessions):
        try:
            states.append(await browsers.state(browser_id))
        except HTTPException:
            pass
    return {"sessions": states, "reload_safe": True}


@app.post("/api/browser")
async def browser_open(req: BrowserOpenReq):
    browser_id, state = await browsers.create(browser_url(req.url))
    return {"browser_id": browser_id, **state}


@app.delete("/api/browser/{browser_id}")
async def browser_close(browser_id: str):
    await browser_execution.close(browser_id)
    await browsers.close(browser_id)
    return {"ok": True}


@app.get("/api/browser/{browser_id}")
async def browser_get_state(browser_id: str):
    return await browsers.state(browser_id)


@app.post("/api/browser/{browser_id}/navigate")
async def browser_navigate(browser_id: str, req: BrowserNavigateReq):
    await browser_execution.ensure(browser_id)
    async with browser_execution.action_locks.setdefault(browser_id, asyncio.Lock()):
        await browser_execution.automation_allowed(browser_id, req.owner)
        return await browsers.navigate(browser_id, browser_url(req.url))


@app.post("/api/browser/{browser_id}/back")
async def browser_back(browser_id: str, owner: str = "agent"):
    await browser_execution.ensure(browser_id)
    async with browser_execution.action_locks.setdefault(browser_id, asyncio.Lock()):
        await browser_execution.automation_allowed(browser_id, owner)
        await browsers.evaluate(browser_id, "history.back()")
        await asyncio.sleep(0.4)
        return await browsers.state(browser_id)


@app.get("/api/browser/{browser_id}/inspect")
async def browser_inspect(browser_id: str):
    return await browsers.inspect(browser_id)


@app.get("/api/browser/{browser_id}/screenshot")
async def browser_screenshot(browser_id: str,
                             width: int = Query(1280, ge=1, le=4096),
                             height: int = Query(800, ge=1, le=4096)):
    return Response(await browsers.screenshot(browser_id, width, height), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/browser/{browser_id}/click")
async def browser_click(browser_id: str, req: BrowserElementReq):
    await browser_execution.ensure(browser_id)
    async with browser_execution.action_locks.setdefault(browser_id, asyncio.Lock()):
        await browser_execution.automation_allowed(browser_id, req.owner)
        return await browsers.click(browser_id, req.element_id)


@app.post("/api/browser/{browser_id}/type")
async def browser_type(browser_id: str, req: BrowserTypeReq):
    await browser_execution.ensure(browser_id)
    async with browser_execution.action_locks.setdefault(browser_id, asyncio.Lock()):
        await browser_execution.automation_allowed(browser_id, req.owner)
        return await browsers.type_text(browser_id, req.element_id, req.text, req.press_enter)


async def read_file(sid: str, path: str) -> str:
    rc, out, err = await sh(DOCKER_BIN, "exec", sid, "cat", path)
    if rc != 0:
        raise HTTPException(404, err.strip() or f"cannot read {path}")
    return out


@app.get("/api/sandbox/{sid}/port")
async def sandbox_port(sid: str):
    """Host port mapped to the sandbox's container port 8000 (for previews)."""
    sid = pool.require(sid)
    rc, out, _ = await sh(DOCKER_BIN, "port", sid, "8000/tcp")
    lines = [l for l in out.splitlines() if ":" in l]
    if rc != 0 or not lines:
        raise HTTPException(404, "sandbox has no published port")
    return {"host_port": int(lines[0].rsplit(":", 1)[1])}


@app.get("/api/sandbox/{sid}/raw")
async def file_raw(sid: str, path: str):
    """Raw file bytes with a guessed mimetype (images etc. for viewers)."""
    sid = pool.require(sid)
    p = safe_path(path)
    rc, out, err = await sh_bytes(DOCKER_BIN, "exec", sid, "cat", p)
    if rc != 0:
        raise HTTPException(404, err.strip() or f"cannot read {p}")
    if len(out) > 20 * 1024 * 1024:
        raise HTTPException(413, "file too large")
    mime = mimetypes.guess_type(p)[0] or "application/octet-stream"
    return Response(out, media_type=mime,
                    headers={"Cache-Control": "no-store"})


@app.get("/api/sandbox/{sid}/tree")
async def file_tree(sid: str):
    sid = pool.require(sid)
    rc, out, _ = await sh(
        DOCKER_BIN, "exec", sid, "sh", "-c",
        f"cd {WORKDIR} && find . -type f "
        "-not -path '*/node_modules/*' -not -path '*/.git/*' "
        "-not -path '*/__pycache__/*' -not -path '*/.venv/*' "
        "| sed 's|^\\./||' | sort | head -300")
    files = [f for f in out.splitlines() if f.strip()] if rc == 0 else []
    return {"files": files}


@app.get("/api/sandbox/{sid}/file")
async def file_read(sid: str, path: str):
    sid = pool.require(sid)
    return {"path": path, "content": await read_file(sid, safe_path(path))}


file_locks = {}


@app.post("/api/sandbox/{sid}/file")
async def file_write(sid: str, req: WriteReq):
    sid = pool.require(sid)
    path = safe_path(req.path)
    async with file_locks.setdefault((sid, path), asyncio.Lock()):
        if req.expected_absent:
            rc, _, _ = await sh(DOCKER_BIN, "exec", sid, "test", "-e", path)
            if rc == 0: raise HTTPException(409, "file already exists; read it before editing")
        if req.expected_content is not None:
            current = await read_file(sid, path)
            if current != req.expected_content:
                raise HTTPException(409, "file changed on disk; your edits are retained. Reload the file after copying or saving your changes elsewhere.")
        rc, _, err = await sh(
            DOCKER_BIN, "exec", "-i", sid, "sh", "-c",
            f'mkdir -p "$(dirname \'{path}\')" && cat > \'{path}\'',
            stdin=req.content.encode())
    if rc != 0:
        raise HTTPException(500, err.strip() or "write failed")
    return {"path": path, "bytes": len(req.content.encode())}


@app.patch("/api/sandbox/{sid}/file")
async def file_edit(sid: str, req: EditReq):
    sid = pool.require(sid)
    path = safe_path(req.path)
    async with file_locks.setdefault((sid, path), asyncio.Lock()):
        content = await read_file(sid, path)
        n = content.count(req.old_string) if req.old_string else 0
        if n == 0:
            return {"path": path, "replaced": 0, "content": content}
        new_content = content.replace(req.old_string, req.new_string, 1)
        rc, _, err = await sh(
            DOCKER_BIN, "exec", "-i", sid, "sh", "-c", f"cat > '{path}'",
            stdin=new_content.encode())
        if rc != 0:
            raise HTTPException(500, err.strip() or "write failed")
        return {"path": path, "replaced": 1, "occurrences": n, "content": new_content}


@app.post("/api/sandbox/{sid}/exec")
async def sandbox_exec(sid: str, req: ExecReq):
    sid = pool.require(sid)
    t = max(1, min(req.timeout, 120))
    rc, out, err = await sh(
        DOCKER_BIN, "exec", "-w", WORKDIR, sid,
        "timeout", "--preserve-status", "--kill-after=2", str(t),
        "bash", "-lc", req.command,
        timeout=t + 10)
    output = (out + (("\n" + err) if err else "")).strip()
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + f"\n… (truncated, {len(output)} chars total)"
    if rc in {137, 143}:
        output += f"\n(command killed after {t}s timeout)"
    return {"exit_code": rc, "output": output}


async def stream_exec_events(sid: str, command: str, timeout: int):
    """Run a command and emit newline-delimited JSON as output arrives."""
    proc = await asyncio.create_subprocess_exec(
        DOCKER_BIN, "exec", "-t", "-e", "TERM=dumb", "-e", "NO_COLOR=1",
        "-w", WORKDIR, sid,
        "timeout", "--preserve-status", "--kill-after=2", str(timeout),
        "bash", "-lc", command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT)
    captured = []
    captured_len = 0
    total_len = 0
    try:
        while True:
            chunk = await proc.stdout.read(1024)
            if not chunk:
                break
            text = chunk.decode(errors="replace").replace("\r\n", "\n")
            total_len += len(text)
            if captured_len < MAX_OUTPUT:
                visible = text[:MAX_OUTPUT - captured_len]
                captured.append(visible)
                captured_len += len(visible)
                yield json.dumps({"type": "output", "data": visible}) + "\n"
        rc = await proc.wait()
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise

    output = "".join(captured).strip()
    suffixes = []
    if total_len > MAX_OUTPUT:
        suffixes.append(f"… (truncated, {total_len} chars total)")
    if rc in {137, 143}:
        suffixes.append(f"(command killed after {timeout}s timeout)")
    if suffixes:
        suffix = "\n" + "\n".join(suffixes)
        output += suffix
        yield json.dumps({"type": "output", "data": suffix + "\n"}) + "\n"
    yield json.dumps({"type": "done", "exit_code": rc,
                      "output": output}) + "\n"


@app.post("/api/sandbox/{sid}/exec/stream")
async def sandbox_exec_stream(sid: str, req: ExecReq):
    sid = pool.require(sid)
    timeout = max(1, min(req.timeout, 120))
    return StreamingResponse(
        stream_exec_events(sid, req.command, timeout),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    uvicorn.run(
        "backend:app", app_dir=str(BASE_DIR),
        host="127.0.0.1", port=8080,
        reload=True, reload_dirs=[str(BASE_DIR)],
        reload_includes=["*.py"], log_level="info")
