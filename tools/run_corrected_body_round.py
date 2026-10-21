"""Single finite local preparation -> training -> heldout evaluation chain."""
import sys,time,subprocess,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];ready=root/'models/body_expert_candidate/dataset.json';deadline=time.monotonic()+1200
while not ready.exists():
    if time.monotonic()>deadline:raise SystemExit('Preparation deadline reached; no retries, old model retained')
    time.sleep(10)
subprocess.run([sys.executable,str(root/'tools/train_body_expert.py'),'--epochs','12'],cwd=root,check=True,timeout=5400)
info=json.loads((root/'models/body_expert_candidate/training.json').read_text(encoding='utf-8'))
if not info['selected']:raise SystemExit('Validation rejected corrected candidate; old model retained')
with (root/'outputs/evaluate-body-v20.log').open('w',encoding='utf-8') as stdout,(root/'outputs/evaluate-body-v20.err').open('w',encoding='utf-8') as stderr:
    subprocess.run([sys.executable,str(root/'tools/evaluate_body_expert.py')],cwd=root,check=True,stdout=stdout,stderr=stderr,timeout=3600)
