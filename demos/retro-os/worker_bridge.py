"""Attach to a worker process whose lifetime is independent of Uvicorn reload."""
import asyncio
import os
from pathlib import Path
import subprocess
import sys
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

URL = 'http://127.0.0.1:' + os.environ.get('RETROVOICE_WORKER_PORT', '8082')
lock = asyncio.Lock()
router = APIRouter()

async def ensure_workers():
    async with lock:
        async with httpx.AsyncClient(trust_env=False,timeout=2) as client:
            try:
                r=await client.get(URL+'/health')
                if r.json().get('service')=='retrovoice-workers': return
                raise RuntimeError('Worker port belongs to another service')
            except httpx.HTTPError: pass
            base=Path(__file__).resolve().parent
            (base/'.agents').mkdir(mode=0o700,exist_ok=True)
            with open(base/'.agents/worker-service.log','ab') as log:
                subprocess.Popen([sys.executable,str(base/'worker_service.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd=base)
            for _ in range(50):
                try:
                    r=await client.get(URL+'/health')
                    if r.json().get('service')=='retrovoice-workers':return
                except httpx.HTTPError:pass
                await asyncio.sleep(.1)
            raise HTTPException(503,'Worker service did not start; see .agents/worker-service.log')

async def forward(method,path,data=None):
    await ensure_workers()
    async with httpx.AsyncClient(trust_env=False,timeout=15) as c:
        r=await c.request(method,URL+path,json=data)
    return JSONResponse(r.json(),status_code=r.status_code)

@router.get('/api/agents')
async def listing():return await forward('GET','/tasks')
@router.post('/api/agents')
async def spawn(request:Request):return await forward('POST','/tasks',await request.json())
@router.get('/api/agents/{tid}')
async def read(tid:str):return await forward('GET','/tasks/'+tid)
@router.post('/api/agents/{tid}/{action}')
async def control(tid:str,action:str,request:Request):
    if action not in {'steer','stop','pause','resume'}:raise HTTPException(404,'unknown worker action')
    return await forward('POST',f'/tasks/{tid}/{action}',await request.json() if action=='steer' else {})
