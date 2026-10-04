"""Real-model checks: RETROVOICE_AGENT_TESTS=1 python -m unittest discover -s tests -p test_background_agents.py."""
import asyncio
import json
import os
from pathlib import Path
import time
import unittest
import httpx
import websockets

@unittest.skipUnless(os.environ.get('RETROVOICE_AGENT_TESTS')=='1','requires running desktop, speech server and model')
class BackgroundAgentTests(unittest.TestCase):
    def test_steering_isolation_conversation_and_cancellation(self):
        c=httpx.Client(base_url='http://localhost:8080',trust_env=False,timeout=30)
        owned=[]
        def req(method,path,**kw):
            r=c.request(method,path,**kw);r.raise_for_status();return r.json()
        def wait_for(tid,predicate):
            end=time.monotonic()+90
            while time.monotonic()<end:
                j=req('GET','/api/agents/'+tid)
                if predicate(j):return j
                if j['status'] in {'failed','interrupted','cancelled'}:self.fail(str(j))
                time.sleep(.25)
            self.fail('Worker did not reach expected state')
        try:
            job=req('POST','/api/agents',json={'name':'QA steering','objective':'First call ask_user asking what value to put in choice.txt. Wait for the answer. Then write choice.txt with that value, read it back, and report. This is an integration test.','max_rounds':10})
            tid=job['task_id'];owned.append(tid)
            paused=wait_for(tid,lambda j:j['status']=='paused')
            collision=c.post('/api/agents',json={'objective':'Should not start','workspace_id':paused['workspace_id']})
            self.assertEqual(collision.status_code,409)
            req('POST',f'/api/agents/{tid}/steer',json={'message':'Use the exact text revised-value, then verify it.'})
            completed=wait_for(tid,lambda j:j['status']=='completed')
            r=c.get(f'/api/sandbox/{completed["container"]}/file',params={'path':'choice.txt'});r.raise_for_status();self.assertEqual(r.json()['content'].strip(),'revised-value')
            long=req('POST','/api/agents',json={'name':'QA cancellation','objective':'Run the command sleep 60 in your terminal. Only after it finishes, create must-not-exist.txt. Do not skip or shorten the sleep; this is a cancellation integration test.','max_rounds':8})
            lid=long['task_id'];owned.append(lid)
            running=wait_for(lid,lambda j:bool(j.get('command_id')))
            # A backend reload must not own the model worker's lifetime.
            Path('demos/retro-os/backend.py').touch()
            async def foreground():
                async with websockets.connect('ws://localhost:8000/ws',max_size=10*1024*1024) as ws:
                    await ws.recv()
                    await ws.send(json.dumps({'type':'text','data':'What is two plus two? Answer briefly.'}))
                    chunks=[]
                    async with asyncio.timeout(40):
                        while True:
                            msg=json.loads(await ws.recv())
                            if msg['type']=='chat_chunk':chunks.append(msg['data'])
                            if msg['type']=='chat_done':return ''.join(chunks)
            answer=asyncio.run(foreground())
            self.assertTrue('4' in answer or 'four' in answer.lower(),answer)
            current=wait_for(lid,lambda j:j['status']=='running')
            self.assertEqual(current['command_id'],running['command_id'])
            req('POST',f'/api/agents/{lid}/stop')
            end=time.monotonic()+15
            while time.monotonic()<end:
                j=req('GET','/api/agents/'+lid)
                if j['status']=='cancelled':break
                time.sleep(.2)
            self.assertEqual(j['status'],'cancelled',j)
            state=req('GET',f'/api/workspaces/{j["workspace_id"]}/state')
            task=next(t for t in state['tasks'] if t['task_id']==j['command_id'])
            self.assertEqual(task['status'],'cancelled')
            self.assertEqual(c.get(f'/api/sandbox/{j["container"]}/file',params={'path':'must-not-exist.txt'}).status_code,404)
        finally:
            for tid in owned:c.post(f'/api/agents/{tid}/stop')
            c.close()
