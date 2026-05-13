import json
import tempfile
import unittest
from pathlib import Path

import yaml

from agent_core.schema import validate_required_fields, validate_action_type, validate_reward_components
from eval.swebench_runner import summarize


class TestSchemaAndEval(unittest.TestCase):
    def test_schema_validation(self):
        schema = yaml.safe_load(Path('slime-coding-agent/configs/rollout_event_schema.yaml').read_text())
        event = {
            'event_id': '1', 'timestamp': 1, 'repo_id': 'r', 'task_id': 't', 'benchmark_id': 'b',
            'commit_base': 'HEAD', 'action_type': 'run_tests', 'action_payload': {},
            'command_results': [], 'test_results': {},
            'reward_components': {'pass': 1, 'quality': 1, 'safety': 1, 'human': 1, 'cost': 0.1},
            'total_reward': 3.9, 'final_status': 'success'
        }
        validate_required_fields(event, schema['required_fields'])
        validate_action_type(event, schema['action_types'])
        validate_reward_components(event)

    def test_summarize(self):
        events = [{'final_status': 'success'}, {'final_status': 'failed'}]
        metrics = summarize(events)
        self.assertEqual(metrics['resolve_rate'], 0.5)
        self.assertEqual(metrics['pass_at_1'], 0.5)


if __name__ == '__main__':
    unittest.main()
