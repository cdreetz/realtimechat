"""Isolated persistent Python evaluator using the real Browser Use helpers."""
import contextlib
import io
import json
import multiprocessing as mp
import os
import secrets
import signal
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Output(io.TextIOBase):
    def __init__(self): self.text = ''; self.truncated = False
    def write(self, text):
        self.text = (self.text + str(text))[-24000:]
        self.truncated |= len(self.text) == 24000
        return len(text)
    def flush(self): pass

def interpreter(pipe):
    os.setsid()
    from browser_harness.admin import ensure_daemon
    import browser_harness.helpers as helpers
    ensure_daemon()
    namespace = {name: getattr(helpers, name) for name in dir(helpers) if not name.startswith('_')}
    namespace['__builtins__'] = __builtins__
    while True:
        code = pipe.recv()
        output = Output()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(code, '<browser_exec>', 'exec'), namespace)
            result = {'status': 'completed', 'output': output.text, 'truncated': output.truncated}
        except BaseException:
            result = {'status': 'failed', 'output': output.text, 'error': traceback.format_exc()[-6000:]}
        pipe.send(result)

lock = threading.RLock()
jobs = {}
process = connection = None
generation = secrets.token_hex(8)
mode = 'agent'
TOKEN = os.environ.get('RETRO_HARNESS_TOKEN', '')

def reset():
    global process, connection, generation
    if process and process.is_alive():
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: process.kill()
        process.join(2)
    if connection: connection.close()
    process = connection = None
    generation = secrets.token_hex(8)

def run(job, code, timeout):
    global process, connection
    with lock:
        if job['status'] != 'running': return
        if process is None or not process.is_alive():
            reset()
            connection, child = mp.Pipe()
            process = mp.Process(target=interpreter, args=(child,), daemon=False)
            process.start()
        job['generation'] = generation
        conn = connection
        conn.send(code)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            with lock:
                if job['status'] != 'running': return
                if conn.poll(.05):
                    job.update(conn.recv(), finished_at=time.time()); return
            time.sleep(.03)
        with lock:
            if job['status'] == 'running':
                job.update(status='timed_out', error='Execution deadline reached; Python variables reset. Inspect the page before retrying: completed actions are not undone.', finished_at=time.time())
                reset()
    except (EOFError, OSError, BrokenPipeError) as exc:
        with lock:
            if job['status'] == 'running':
                job.update(status='interrupted', error=str(exc), finished_at=time.time()); reset()

def stop():
    for job in jobs.values():
        if job['status'] == 'running':
            job.update(status='cancelled', finished_at=time.time(), error='Stopped; Python variables reset. Completed browser actions are not undone.')
    reset()

def dispatch(path, data):
    global mode
    with lock:
        if path == '/view':
            if process is None: return {}
            from browser_harness.helpers import current_tab
            try: return current_tab()
            except Exception: return {}
        if path == '/state':
            return {'generation': generation, 'mode': mode, 'jobs': list(jobs.values())}
        if path == '/mode':
            if data['mode'] not in {'human', 'agent'}: raise ValueError('invalid mode')
            if data['mode'] == 'human': stop()
            mode = data['mode']
            return {'mode': mode, 'generation': generation}
        if path == '/stop':
            stop(); return {'generation': generation, 'stopped': True}
        if path == '/submit':
            if data.get('job_id') in jobs: return jobs[data['job_id']]
            if mode == 'human' and data.get('owner') != 'human': raise ValueError('User has control; wait for Resume agent')
            if any(j['status'] == 'running' for j in jobs.values()): raise ValueError('Browser already executing a script')
            if len(data.get('code', '')) > 64000: raise ValueError('script too long')
            if data.get('generation') and data['generation'] != generation: raise ValueError('Python session reset; stale generation. Inspect state and rebuild variables explicitly.')
            job_id = data.get('job_id') or secrets.token_hex(8)
            if job_id in jobs: return jobs[job_id]  # Idempotency, never replay a submitted action.
            job = {'task_id': job_id, 'kind': 'browser_exec', 'status': 'running', 'started_at': time.time(), 'code': data['code'], 'owner': data.get('owner', 'agent')}
            jobs[job_id] = job
            for old in list(jobs)[:-100]:
                if jobs[old]['status'] != 'running': del jobs[old]
            threading.Thread(target=run, args=(job, data['code'], max(1, min(120, float(data.get('timeout', 30))))), daemon=True).start()
            return dict(job)
        raise ValueError('unknown operation')

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not TOKEN or self.headers.get('Authorization') != 'Bearer ' + TOKEN:
            self.send_error(403); return
        try:
            size = int(self.headers.get('Content-Length', 0))
            if size > 100000: raise ValueError('request too large')
            result = dispatch(self.path, json.loads(self.rfile.read(size) or '{}'))
            code = 200
        except (ValueError, KeyError) as exc: result, code = {'detail': str(exc)}, 409
        body = json.dumps(result).encode()
        self.send_response(code); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *args): pass

if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', 8766), Handler).serve_forever()
