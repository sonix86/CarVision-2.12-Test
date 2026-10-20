"""One bounded local experiment follow-up: wait for training, fit references, never promote."""
from pathlib import Path
import json,time,subprocess,sys
root=Path(__file__).resolve().parents[1];target=root/'models/expanded_parts_candidate/training.json';deadline=time.monotonic()+5400
while time.monotonic()<deadline:
    if target.exists():
        try:info=json.loads(target.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError):time.sleep(2);continue
        if info.get('best_epoch',0)<=0:raise SystemExit('No selected training checkpoint; retain current release')
        print('Training complete; fitting train-only references once',flush=True)
        run=subprocess.run([sys.executable,'-u',str(root/'tools/fit_expanded_references.py')],cwd=root,timeout=1200)
        print('Reference fitting exit code',run.returncode,flush=True);raise SystemExit(run.returncode)
    error=root/'outputs/train-expanded-v18.err'
    if error.exists() and 'Traceback (most recent call last)' in error.read_text(encoding='utf-8',errors='replace'):raise SystemExit('Training failed: inspect train-expanded-v18.err')
    time.sleep(30)
raise SystemExit('Finite follow-up wait expired; no repeated runs or promotion')
