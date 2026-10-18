"""Verify dependent submission without accessing a real scheduler."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class SlurmSubmissionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('bash'), 'Bash is required for the Slurm example')
    def test_temperature_array_waits_for_qha_and_preserves_spaced_paths(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root/'calls.jsonl'
            scheduler = root/'sbatch'
            scheduler.write_text(f'#!{sys.executable}\n' +
                'import json, os, sys\n'
                'with open(os.environ["TEST_SBATCH_LOG"], "a") as f:\n'
                '    f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                'print("12345;cluster")\n')
            scheduler.chmod(0o755)
            env = dict(os.environ, PATH=str(root)+os.pathsep+os.environ.get('PATH', ''),
                       UMA_PYTHON=sys.executable, TEST_SBATCH_LOG=str(log))
            env.pop('ACCOUNT', None)
            env.pop('PARTITION', None)
            output = root/'run with spaces'
            subprocess.run(['bash', str(repo/'examples/slurm-qha.sh'), 'submit',
                            str(repo/'examples/synthetic-water.cif'), str(output)],
                           env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            self.assertIn('--dependency=afterok:12345', calls[1])
            self.assertIn('--array=0-5', calls[1])
            self.assertEqual(calls[0][-1], str(output))
            self.assertEqual(calls[1][-1], str(output))
            self.assertEqual(calls[0][-3], 'qha')
            self.assertEqual(calls[1][-3], 'temperature')
