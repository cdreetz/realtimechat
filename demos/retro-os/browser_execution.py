"""Browser Use container adapter; code never runs in the desktop backend."""
import asyncio
import json
import secrets
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

IMAGE = 'retrovoice-browser-harness:v2'
CLIENT = '''import sys,json,os,urllib.request,urllib.error
path,payload=json.load(sys.stdin)
req=urllib.request.Request('http://127.0.0.1:8766'+path,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+os.environ.get('RETRO_HARNESS_TOKEN','')})
try:
 with urllib.request.urlopen(req,timeout=8) as r: print(r.read().decode())
except urllib.error.HTTPError as e: print(e.read().decode());sys.exit(2)
'''

class Script(BaseModel):
    code: str = Field(min_length=1, max_length=64000)
    timeout: float = Field(default=30, ge=1, le=120)
    owner: str = 'agent'
    generation: str | None = None
    job_id: str | None = None

class Mode(BaseModel):
    mode: str

class Input(BaseModel):
    kind: str
    x: float = 0
    y: float = 0
    text: str = Field(default='', max_length=16000)
    key: str = ''
    delta_y: float = 0

class BrowserExecution:
    def __init__(self, browsers, docker, sh, base):
        self.browsers, self.docker, self.sh, self.base = browsers, docker, sh, base
        self.locks = {}
        self.ready = set()
        self.targets = {}
        self.action_locks = {}
        self.router = APIRouter()
        self.routes()

    def container(self, bid):
        self.browsers.require(bid)
        return 'retrovoice-harness-' + bid

    async def ensure(self, bid):
        async with self.locks.setdefault(bid, asyncio.Lock()):
            container = self.container(bid)
            if bid in self.ready: return
            image_rc, image_name, _ = await self.sh(self.docker, 'inspect', '-f', '{{.Config.Image}}', container)
            if image_rc == 0 and image_name.strip() != IMAGE:
                await self.sh(self.docker, 'rm', '-f', container)  # Only disposable controller; Chromium stays alive.
            rc, out, _ = await self.sh(self.docker, 'inspect', '-f', '{{.State.Running}}', container)
            if rc:
                rc, _, _ = await self.sh(self.docker, 'image', 'inspect', IMAGE)
                if rc:
                    rc, _, err = await self.sh(self.docker, 'build', '-t', IMAGE, str(self.base / 'browser-runtime'), timeout=600)
                    if rc: raise HTTPException(503, err[-2000:])
                rc, _, err = await self.sh(self.docker, 'run', '-d', '--init', '--name', container,
                    '--network', 'container:' + self.browsers.require(bid)['container'],
                    '--memory', '512m', '--cpus', '1', '--pids-limit', '128', '-e', 'RETRO_HARNESS_TOKEN='+secrets.token_hex(32), IMAGE)
            elif out.strip() != 'true':
                rc, _, err = await self.sh(self.docker, 'start', container)
            if rc: raise HTTPException(503, err[-2000:])
            for _ in range(30):
                try:
                    await self.rpc(bid, '/state'); self.ready.add(bid); return
                except HTTPException: await asyncio.sleep(.1)
            raise HTTPException(503, 'Browser harness did not start')

    async def rpc(self, bid, path, data=None):
        rc, out, err = await self.sh(self.docker, 'exec', '-i', self.container(bid), 'python', '-c', CLIENT,
            stdin=json.dumps([path, data or {}]).encode(), timeout=15)
        if rc:
            self.ready.discard(bid)
            try: detail = json.loads(out)['detail']
            except (ValueError, KeyError): detail = 'Browser controller unavailable; inspect state before retrying'
            raise HTTPException(409 if rc == 2 else 503, detail)
        try: return json.loads(out)
        except ValueError: raise HTTPException(502, 'Invalid browser controller response')

    async def target(self, bid):
        if bid not in self.ready:
            rc, out, _ = await self.sh(self.docker, 'inspect', '-f', '{{.State.Running}}', self.container(bid))
            if rc or out.strip() != 'true': return None
        try:
            view = await self.rpc(bid, '/view')
            return view.get('targetId')
        except HTTPException: return None

    async def close(self, bid):
        await self.sh(self.docker, 'rm', '-f', self.container(bid))
        self.ready.discard(bid)

    async def automation_allowed(self, bid, owner="agent"):
        await self.ensure(bid)
        state = await self.rpc(bid, '/state')
        if (state['mode'] == 'human' and owner != 'human') or any(j['status'] == 'running' for j in state['jobs']):
            raise HTTPException(409, 'Browser is under human control or executing a script')

    def routes(self):
        @self.router.post('/api/browser/{bid}/exec')
        async def execute(bid: str, req: Script):
            await self.ensure(bid)
            async with self.action_locks.setdefault(bid, asyncio.Lock()):
                return await self.rpc(bid, '/submit', req.model_dump())

        @self.router.get('/api/browser/{bid}/execution')
        async def state(bid: str):
            await self.ensure(bid)
            return await self.rpc(bid, '/state')

        @self.router.post('/api/browser/{bid}/execution/stop')
        async def stop(bid: str):
            await self.ensure(bid)
            return await self.rpc(bid, '/stop')

        @self.router.post('/api/browser/{bid}/control')
        async def control(bid: str, req: Mode):
            await self.ensure(bid)
            return await self.rpc(bid, '/mode', req.model_dump())

        @self.router.post('/api/browser/{bid}/input')
        async def input_event(bid: str, req: Input):
            await self.ensure(bid)
            state = await self.rpc(bid, '/state')
            if state['mode'] != 'human': raise HTTPException(409, 'Choose Take control first')
            if req.kind == 'click':
                for typ in ['mousePressed', 'mouseReleased']:
                    await self.browsers.cdp(bid, 'Input.dispatchMouseEvent', {'type': typ, 'x': req.x, 'y': req.y, 'button': 'left', 'clickCount': 1})
            elif req.kind == 'text':
                await self.browsers.cdp(bid, 'Input.insertText', {'text': req.text})
            elif req.kind == 'key':
                codes = {'Enter':13,'Tab':9,'Backspace':8,'Escape':27,'ArrowLeft':37,'ArrowUp':38,'ArrowRight':39,'ArrowDown':40,'Delete':46}
                if req.key not in codes: raise HTTPException(400, 'unsupported key')
                for typ in ['keyDown', 'keyUp']:
                    await self.browsers.cdp(bid, 'Input.dispatchKeyEvent', {'type':typ,'key':req.key,'windowsVirtualKeyCode':codes[req.key], **({'text':'\r'} if typ == 'keyDown' and req.key == 'Enter' else {})})
            elif req.kind == 'scroll':
                await self.browsers.cdp(bid, 'Input.dispatchMouseEvent', {'type':'mouseWheel','x':req.x,'y':req.y,'deltaY':req.delta_y,'deltaX':0})
            else: raise HTTPException(400, 'unsupported input')
            return {'ok':True}
