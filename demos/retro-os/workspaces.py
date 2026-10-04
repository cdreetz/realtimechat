"""Durable workspace identities and container runtime adapters."""
import asyncio
import json
import secrets
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

IMAGE = "retrovoice-workspace:v1"


class WorkspaceCreate(BaseModel):
    name: str = Field(default="Untitled workspace", min_length=1, max_length=100)
    legacy_sandbox: str | None = None


class TerminalInput(BaseModel):
    data: str = Field(max_length=65536)
    command: bool = False
    owner: str = "human"


class TerminalSize(BaseModel):
    cols: int = Field(ge=2, le=500)
    rows: int = Field(ge=2, le=300)


class WorkspaceManager:
    def __init__(self, base, docker, sh, legacy_pool):
        self.base, self.docker, self.sh, self.pool = Path(base), docker, sh, legacy_pool
        self.root = self.base / ".workspaces"
        self.items = {}
        self.lock = asyncio.Lock()
        self.image_ready = False
        self.router = APIRouter()
        self.routes()

    def load(self):
        self.root.mkdir(mode=0o700, exist_ok=True)
        for path in self.root.glob("*/workspace.json"):
            value = json.loads(path.read_text())
            if value["workspace_id"] != path.parent.name:
                raise RuntimeError(f"invalid workspace manifest: {path}")
            self.items[value["workspace_id"]] = value

    def save(self, item):
        path = self.root / item["workspace_id"] / "workspace.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(item, indent=2))
        temp.chmod(0o600)
        temp.replace(path)

    def public(self, item):
        return {k: v for k, v in item.items() if k not in {"token", "port"}}

    def require(self, workspace_id):
        if workspace_id not in self.items:
            raise HTTPException(404, "unknown workspace")
        return self.items[workspace_id]

    def container_for(self, name):
        for item in self.items.values():
            if name in {item["container"], item.get("legacy_sandbox")}:
                return item["container"]
        return None

    async def build(self):
        if self.image_ready:
            return
        rc, _, _ = await self.sh(self.docker, "image", "inspect", IMAGE)
        if rc:
            rc, _, err = await self.sh(self.docker, "build", "-t", IMAGE,
                                      str(self.base / "runtime"), timeout=600)
            if rc:
                raise HTTPException(503, f"workspace image build failed: {err[-3000:]}")
        self.image_ready = True

    async def ensure(self, item):
        await self.build()
        container = item["container"]
        rc, out, _ = await self.sh(self.docker, "inspect", "-f", "{{.State.Running}}", container)
        if rc:
            directory = self.root / item["workspace_id"]
            rc, _, err = await self.sh(self.docker, "run", "-d", "--init",
                "--name", container, "--label", "retrovoice-workspace=1",
                "--memory", "1g", "--cpus", "2", "--pids-limit", "256",
                "-p", "127.0.0.1:0:8000", "-p", "127.0.0.1:0:8765",
                "--mount", f"type=bind,src={directory / 'files'},dst=/workspace",
                "--mount", f"type=bind,src={directory / 'runtime'},dst=/state",
                "-e", f"RETRO_TOKEN={item['token']}", IMAGE)
        elif out.strip() != "true":
            rc, _, err = await self.sh(self.docker, "start", container)
        else:
            err = ""
        if rc:
            raise HTTPException(503, f"workspace runtime unavailable: {err.strip()}")
        rc, out, _ = await self.sh(self.docker, "port", container, "8765/tcp")
        if rc or not out.strip():
            raise HTTPException(503, "workspace runtime port is unavailable")
        item["port"] = int(out.strip().splitlines()[0].rsplit(":", 1)[1])
        self.save(item)
        for _ in range(30):
            try:
                await self.rpc(item, "/state")
                return self.public(item)
            except HTTPException:
                await asyncio.sleep(.1)
        raise HTTPException(503, "terminal supervisor did not become ready")

    async def create(self, name, legacy=None):
        async with self.lock:
            if legacy:
                existing = next((w for w in self.items.values() if w.get("legacy_sandbox") == legacy), None)
                if existing:
                    return await self.ensure(existing)
                self.pool.require(legacy)
            wid = secrets.token_hex(8)
            directory = self.root / wid
            (directory / "files").mkdir(parents=True, mode=0o700)
            (directory / "runtime").mkdir(mode=0o700)
            item = {"workspace_id": wid, "name": name.strip() or "Untitled workspace",
                    "container": f"retrovoice-ws-{wid}", "token": secrets.token_hex(32),
                    "legacy_sandbox": legacy}
            if legacy:
                rc, _, err = await self.sh(self.docker, "cp", f"{legacy}:/workspace/.", str(directory / "files"))
                if rc:
                    raise HTTPException(500, f"legacy import failed; original untouched: {err}")
            self.items[wid] = item
            self.save(item)  # Persist identity/files before starting any runtime.
            return await self.ensure(item)

    async def rpc(self, item, path, payload=None):
        if not item.get("port"):
            raise HTTPException(503, "workspace is offline; reopen it")
        try:
            async with httpx.AsyncClient(trust_env=False, timeout=5) as client:
                r = await client.post(f"http://127.0.0.1:{item['port']}{path}",
                    headers={"Authorization": "Bearer " + item["token"]}, json=payload or {})
            if not r.is_success:
                raise HTTPException(r.status_code, r.json().get("detail", "terminal operation failed"))
            return r.json()
        except httpx.HTTPError as exc:
            raise HTTPException(503, "workspace is offline; reopen it to reconnect") from exc

    def routes(self):
        router = self.router

        @router.get("/api/workspaces")
        async def list_workspaces():
            imported = {w.get("legacy_sandbox") for w in self.items.values()}
            return {"workspaces": [self.public(w) for w in self.items.values()],
                    "legacy_sandboxes": sorted(self.pool.active - imported)}

        @router.post("/api/workspaces")
        async def create(req: WorkspaceCreate):
            return await self.create(req.name, req.legacy_sandbox)

        @router.post("/api/workspaces/{wid}/open")
        async def open_workspace(wid: str):
            async with self.lock:
                return await self.ensure(self.require(wid))

        @router.patch("/api/workspaces/{wid}")
        async def rename(wid: str, req: WorkspaceCreate):
            item = self.require(wid)
            item["name"] = req.name.strip() or "Untitled workspace"
            self.save(item)
            return self.public(item)

        @router.get("/api/workspaces/{wid}/state")
        async def state(wid: str):
            return await self.rpc(self.require(wid), "/state")

        @router.post("/api/workspaces/{wid}/terminals")
        async def open_terminal(wid: str):
            return await self.rpc(self.require(wid), "/open")

        @router.get("/api/workspaces/{wid}/terminals/{tid}")
        async def read_terminal(wid: str, tid: str, cursor: int = 0):
            return await self.rpc(self.require(wid), "/read", {"terminal_id": tid, "cursor": cursor})

        @router.post("/api/workspaces/{wid}/terminals/{tid}/input")
        async def write_terminal(wid: str, tid: str, req: TerminalInput):
            return await self.rpc(self.require(wid), "/write", {"terminal_id": tid, **req.model_dump()})

        @router.post("/api/workspaces/{wid}/terminals/{tid}/resize")
        async def resize_terminal(wid: str, tid: str, req: TerminalSize):
            return await self.rpc(self.require(wid), "/resize", {"terminal_id": tid, **req.model_dump()})

        @router.post("/api/workspaces/{wid}/terminals/{tid}/stop")
        async def stop_terminal(wid: str, tid: str):
            return await self.rpc(self.require(wid), "/stop", {"terminal_id": tid})

        @router.post("/api/workspaces/{wid}/tasks/{task_id}/stop")
        async def stop_task(wid: str, task_id: str):
            item = self.require(wid)
            state = await self.rpc(item, "/state")
            task = next((t for t in state["tasks"] if t["task_id"] == task_id), None)
            if not task:
                raise HTTPException(404, "unknown task")
            await self.rpc(item, "/stop", {"terminal_id": task["terminal_id"], "task_id": task_id})
            return {"task_id": task_id, "stop_requested": True}
