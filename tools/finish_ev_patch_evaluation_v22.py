"""One bounded evaluation after the active training run; never install or retrain."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
report = ROOT/'models/ev_patch_v22/training.json'
deadline = time.monotonic() + 90*60
while not report.exists() and time.monotonic() < deadline:
    time.sleep(15)
if not report.exists():
    raise SystemExit('Training did not complete within the bounded wait; no retry.')
if not json.loads(report.read_text(encoding='utf-8'))['selected']:
    raise SystemExit('Validation rejected every candidate; no evaluation or installation.')
if (report.parent/'evaluation.json').exists():
    raise SystemExit('An evaluation already exists; do not repeat or overwrite it.')
result = subprocess.run([sys.executable, str(ROOT/'tools/evaluate_patch_v22.py')], cwd=ROOT, timeout=45*60)
raise SystemExit(result.returncode)
