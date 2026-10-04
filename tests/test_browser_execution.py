"""Live app integration: RETROVOICE_APP_TESTS=1 python -m unittest discover -s tests -p test_browser_execution.py."""
import json
import os
import subprocess
import time
import unittest
import httpx

@unittest.skipUnless(os.environ.get('RETROVOICE_APP_TESTS')=='1','requires running desktop and Docker')
class BrowserExecutionTests(unittest.TestCase):
    def test_real_harness_persistence_cancellation_takeover_and_restart(self):
        with httpx.Client(base_url='http://localhost:8080',trust_env=False,timeout=60) as c:
            def request(method,path,**args):
                r=c.request(method,path,**args);r.raise_for_status();return r.json()
            browser=request('POST','/api/browser',json={'url':'about:blank'})['browser_id']
            base=f'/api/browser/{browser}'
            try:
                def settled(task_id):
                    deadline=time.monotonic()+35
                    while time.monotonic()<deadline:
                        state=request('GET',base+'/execution')
                        job=next(j for j in state['jobs'] if j['task_id']==task_id)
                        if job['status']!='running':return job
                        time.sleep(.15)
                    self.fail('Execution did not settle')
                def execute(code,task_id,**extra):
                    return request('POST',base+'/exec',json={'code':code,'job_id':task_id,**extra})
                html='<form id="form"><input id="name"><button>Submit</button></form><div id="result"></div>'
                js='document.body.innerHTML='+json.dumps(html)+'; document.getElementById("form").onsubmit=e=>{e.preventDefault();document.getElementById("result").textContent=document.getElementById("name").value;};'
                code=f'new_tab("about:blank")\njs({js!r})\nfill_input("#name", "harness verified")\npress_key("Enter")\nprint(js("document.getElementById(\\"result\\").textContent"))\nanswer=41'
                execute(code,'form')
                result=settled('form');self.assertEqual(result['status'],'completed',result);self.assertIn('harness verified',result['output'])
                execute('answer+=1; print(answer)','variables')
                self.assertIn('42',settled('variables')['output'])
                execute('answer+=100','variables')  # Same id must not rerun, even with different source.
                execute('print(answer)','check-replay')
                self.assertIn('42',settled('check-replay')['output'])
                old=request('GET',base+'/execution')['generation']
                execute('while True: pass','cancel')
                request('POST',base+'/execution/stop')
                self.assertEqual(settled('cancel')['status'],'cancelled')
                self.assertNotEqual(old,request('GET',base+'/execution')['generation'])
                self.assertEqual(c.post(base+'/exec',json={'code':'print(answer)','generation':old}).status_code,409)
                request('POST',base+'/control',json={'mode':'human'})
                self.assertEqual(c.post(base+'/exec',json={'code':'print(1)'}).status_code,409)
                self.assertEqual(c.post(base+'/navigate',json={'url':'about:blank'}).status_code,409)
                execute('print(page_info()["url"])','human',owner='human')
                self.assertEqual(settled('human')['status'],'completed')
                execute("js(" + repr('document.getElementById("name").focus()') + ")",'focus',owner='human')
                self.assertEqual(settled('focus')['status'],'completed')
                request('POST',base+'/input',json={'kind':'text','text':' manual'})
                request('POST',base+'/input',json={'kind':'key','key':'Enter'})
                execute("print(js(" + repr('document.getElementById("result").textContent') + "))",'manual-result',owner='human')
                self.assertIn('manual',settled('manual-result')['output'])
                request('POST',base+'/control',json={'mode':'agent'})
                execute('while True: pass','timeout',timeout=1)
                self.assertEqual(settled('timeout')['status'],'timed_out')
                old=request('GET',base+'/execution')['generation']
                subprocess.run(['docker','restart','-t','1','retrovoice-harness-'+browser],check=True,stdout=subprocess.DEVNULL)
                for _ in range(30):
                    try: state=request('GET',base+'/execution');break
                    except httpx.HTTPError:time.sleep(.1)
                self.assertNotEqual(old,state['generation']);self.assertEqual(state['jobs'],[])
                self.assertIn('browser_id',request('GET',base))  # Chromium itself survives controller reset.
            finally:request('DELETE',base)
