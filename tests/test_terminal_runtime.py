"""Opt-in real Docker checks: RETROVOICE_DOCKER_TESTS=1 python -m unittest discover -s tests."""
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
import time
import unittest
import urllib.request


@unittest.skipUnless(os.environ.get('RETROVOICE_DOCKER_TESTS') == '1', 'requires Docker')
class TerminalRuntimeTests(unittest.TestCase):
    def test_shell_tasks_and_restart_recovery(self):
        name = 'retrovoice-test-' + secrets.token_hex(6)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'files').mkdir()
            (root / 'state').mkdir()
            def docker(*args):
                return subprocess.check_output(['docker', *args], text=True).strip()
            docker('run', '-d', '--name', name, '-p', '127.0.0.1:0:8765',
                   '-v', f'{root / "files"}:/workspace', '-v', f'{root / "state"}:/state',
                   '-e', 'RETRO_TOKEN=test-only', 'retrovoice-workspace:v1')
            try:
                def rpc(path, **data):
                    port = docker('port', name, '8765/tcp').rsplit(':', 1)[1]
                    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}',
                        data=json.dumps(data).encode(), headers={'Authorization': 'Bearer test-only'})
                    with urllib.request.urlopen(req, timeout=3) as response:
                        return json.load(response)
                def until(fn):
                    deadline = time.monotonic() + 12
                    while time.monotonic() < deadline:
                        try:
                            value = fn()
                            if value:
                                return value
                        except (OSError, ValueError):
                            pass
                        time.sleep(.1)
                    self.fail('terminal did not reach expected state')
                until(lambda: rpc('/state'))
                tid = rpc('/open')['terminal_id']
                def ready():
                    return rpc('/read', terminal_id=tid)['ready']
                def run(command):
                    until(ready)
                    return rpc('/write', terminal_id=tid, data=command, command=True)['task_id']
                def task(task_id):
                    return next(t for t in rpc('/state')['tasks'] if t['task_id'] == task_id)
                run('mkdir -p demo && cd demo && export CHECK_STATE=retained && printf saved > hello.txt')
                run('printf "STATE:%s:%s\\n" "$PWD" "$CHECK_STATE"')
                until(ready)
                output = base64.b64decode(rpc('/read', terminal_id=tid)['output'])
                self.assertIn(b'STATE:/workspace/demo:retained', output)
                self.assertEqual((root / 'files/demo/hello.txt').read_text(), 'saved')
                rpc('/resize', terminal_id=tid, cols=93, rows=27)
                run('stty size')
                until(ready)
                self.assertIn(b'27 93', base64.b64decode(rpc('/read', terminal_id=tid)['output']))
                running = run('sleep 60')
                rpc('/stop', terminal_id=tid, task_id=running)
                until(lambda: task(running)['status'] == 'cancelled')
                stubborn = run('python -c "import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); print(12345,flush=True); time.sleep(60)"')
                until(lambda: b'12345\r\n' in base64.b64decode(rpc('/read', terminal_id=tid)['output']))
                rpc('/stop', terminal_id=tid, task_id=stubborn)
                until(lambda: task(stubborn)['status'] == 'cancelled')
                interrupted = run('sleep 60')
                until(lambda: (root / 'state/sessions.json').exists() and
                      any(t['task_id'] == interrupted for t in json.loads((root / 'state/sessions.json').read_text())['tasks']))
                docker('restart', '-t', '1', name)
                until(lambda: task(interrupted)['status'] == 'interrupted')
                self.assertEqual(rpc('/read', terminal_id=tid)['status'], 'exited')
                self.assertEqual((root / 'files/demo/hello.txt').read_text(), 'saved')
            finally:
                docker('rm', '-f', name)  # Only this fixture's uniquely named container.
