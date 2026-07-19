#!/usr/bin/env python3
"""RetroVoice OS demo backend.

Serves the demo page and manages Docker sandboxes that back the code editor
windows. A warm pool of containers is kept ready so opening an editor is
instant; each sandbox is a python:3.11-slim container with a /workspace
directory, capped CPU/memory, and removed on release/shutdown.

Run:  python backend.py   (needs Docker running; serves http://localhost:8080)
"""
import asyncio
import logging
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retro-backend")

IMAGE = "python:3.11-slim"
LABEL = "retrovoice-demo=1"
POOL_SIZE = 2
WORKDIR = "/workspace"
EXEC_TIMEOUT_S = 30
MAX_OUTPUT = 8000
BASE_DIR = Path(__file__).resolve().parent


async def sh(*args, stdin: bytes = None, timeout: float = 120):
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
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


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
    if not str(norm).startswith(WORKDIR):
        raise HTTPException(400, f"paths must live under {WORKDIR}")
    return str(norm)


class SandboxPool:
    def __init__(self):
        self.warm: list[str] = []
        self.active: set[str] = set()
        self.lock = asyncio.Lock()

    async def start(self):
        rc, out, _ = await sh("docker", "ps", "-aq", "--filter", f"label={LABEL}")
        stale = out.split()
        if stale:
            await sh("docker", "rm", "-f", *stale)
            log.info(f"removed {len(stale)} stale sandbox(es)")
        rc, _, _ = await sh("docker", "image", "inspect", IMAGE)
        if rc != 0:
            log.info(f"pulling {IMAGE}...")
            rc, _, err = await sh("docker", "pull", IMAGE, timeout=600)
            if rc != 0:
                raise RuntimeError(f"could not pull {IMAGE}: {err}")
        await self.refill()
        log.info(f"warm pool ready ({len(self.warm)} sandbox(es))")

    async def _spawn(self) -> str:
        name = f"retrovoice-sbx-{secrets.token_hex(4)}"
        rc, _, err = await sh(
            "docker", "run", "-d", "--rm", "--label", LABEL,
            "--memory", "1g", "--cpus", "2", "-w", WORKDIR,
            "--name", name, IMAGE, "sleep", "infinity")
        if rc != 0:
            raise HTTPException(500, f"docker run failed: {err.strip()}")
        return name

    async def refill(self):
        async with self.lock:
            while len(self.warm) < POOL_SIZE:
                self.warm.append(await self._spawn())

    async def acquire(self) -> str:
        async with self.lock:
            name = self.warm.pop(0) if self.warm else await self._spawn()
            self.active.add(name)
        asyncio.create_task(self.refill())
        log.info(f"sandbox acquired: {name}")
        return name

    async def release(self, name: str):
        self.active.discard(name)
        await sh("docker", "rm", "-f", name)
        log.info(f"sandbox released: {name}")

    async def shutdown(self):
        names = self.warm + list(self.active)
        self.warm = []
        self.active = set()
        if names:
            await sh("docker", "rm", "-f", *names)
        log.info(f"cleaned up {len(names)} sandbox(es)")

    def require(self, name: str) -> str:
        if name not in self.active:
            raise HTTPException(404, f"unknown sandbox {name!r}")
        return name


pool = SandboxPool()


@asynccontextmanager
async def lifespan(app):
    await pool.start()
    yield
    await pool.shutdown()


app = FastAPI(lifespan=lifespan)


class WriteReq(BaseModel):
    path: str
    content: str


class EditReq(BaseModel):
    path: str
    old_string: str
    new_string: str


class ExecReq(BaseModel):
    command: str
    timeout: int = EXEC_TIMEOUT_S


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/sandbox")
async def sandbox_acquire():
    return {"sandbox_id": await pool.acquire()}


@app.delete("/api/sandbox/{sid}")
async def sandbox_release(sid: str):
    await pool.release(pool.require(sid))
    return {"ok": True}


async def read_file(sid: str, path: str) -> str:
    rc, out, err = await sh("docker", "exec", sid, "cat", path)
    if rc != 0:
        raise HTTPException(404, err.strip() or f"cannot read {path}")
    return out


@app.get("/api/sandbox/{sid}/tree")
async def file_tree(sid: str):
    sid = pool.require(sid)
    rc, out, _ = await sh(
        "docker", "exec", sid, "sh", "-c",
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


@app.post("/api/sandbox/{sid}/file")
async def file_write(sid: str, req: WriteReq):
    sid = pool.require(sid)
    path = safe_path(req.path)
    rc, _, err = await sh(
        "docker", "exec", "-i", sid, "sh", "-c",
        f'mkdir -p "$(dirname \'{path}\')" && cat > \'{path}\'',
        stdin=req.content.encode())
    if rc != 0:
        raise HTTPException(500, err.strip() or "write failed")
    return {"path": path, "bytes": len(req.content.encode())}


@app.patch("/api/sandbox/{sid}/file")
async def file_edit(sid: str, req: EditReq):
    sid = pool.require(sid)
    path = safe_path(req.path)
    content = await read_file(sid, path)
    n = content.count(req.old_string) if req.old_string else 0
    if n == 0:
        return {"path": path, "replaced": 0, "content": content}
    new_content = content.replace(req.old_string, req.new_string, 1)
    rc, _, err = await sh(
        "docker", "exec", "-i", sid, "sh", "-c", f"cat > '{path}'",
        stdin=new_content.encode())
    if rc != 0:
        raise HTTPException(500, err.strip() or "write failed")
    return {"path": path, "replaced": 1, "occurrences": n, "content": new_content}


@app.post("/api/sandbox/{sid}/exec")
async def sandbox_exec(sid: str, req: ExecReq):
    sid = pool.require(sid)
    t = max(1, min(req.timeout, 120))
    rc, out, err = await sh(
        "docker", "exec", "-w", WORKDIR, sid,
        "timeout", str(t), "bash", "-lc", req.command,
        timeout=t + 10)
    output = (out + (("\n" + err) if err else "")).strip()
    if len(output) > MAX_OUTPUT:
        output = output[:MAX_OUTPUT] + f"\n… (truncated, {len(output)} chars total)"
    if rc == 124:
        output += f"\n(command killed after {t}s timeout)"
    return {"exit_code": rc, "output": output}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")
