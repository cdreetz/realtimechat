import importlib.util
from pathlib import Path
import sys
import unittest

spec=importlib.util.spec_from_file_location('retro_worker_test',Path(__file__).resolve().parents[1]/'demos/retro-os/worker_service.py')
worker=importlib.util.module_from_spec(spec);sys.modules[spec.name]=worker;spec.loader.exec_module(worker)

class ContextTests(unittest.TestCase):
    def test_trimming_keeps_tool_pairs_and_recent_steering(self):
        messages=[{'role':'system','content':'system'},{'role':'user','content':'objective'}]
        for i in range(20):
            messages.extend([{'role':'assistant','tool_calls':[{'id':str(i),'type':'function','function':{'name':'read_file','arguments':'{}'}}]}, {'role':'tool','tool_call_id':str(i),'content':'x'*18000}])
        result=worker.model_messages({'messages':messages,'events':[{'kind':'steering_applied','text':'new direction'}]})
        self.assertLess(len(result),len(messages))
        self.assertIn('new direction',result[2]['content'])
        calls={c['id'] for m in result for c in m.get('tool_calls',[])}
        answers={m['tool_call_id'] for m in result if m['role']=='tool'}
        self.assertEqual(calls,answers)
        self.assertIn('19',answers)
    def test_small_history_is_preserved(self):
        messages=[{'role':'system','content':'system'},{'role':'user','content':'objective'},{'role':'assistant','content':'done'}]
        self.assertEqual(worker.model_messages({'messages':messages,'events':[]}),messages)
