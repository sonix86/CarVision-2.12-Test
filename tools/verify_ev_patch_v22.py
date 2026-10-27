"""Install onto the previously verified recipient copy, run its unchanged EXE offline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT/'outputs/CarVision-EV-Patch-2.2'
TARGET = ROOT/'outputs/Transfer extracted София/CarVision'
OUT = ROOT/'outputs/ev22-recipient-verification'
OUT.mkdir(exist_ok=True)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((PATCH/'PATCH_MANIFEST.json').read_text(encoding='utf-8'))
exe = TARGET/'CarVision.exe'
assert sha(exe) == manifest['exe_sha256']
env = os.environ.copy()
system_root = next(v for k,v in env.items() if k.lower()=='systemroot')
env.update(PATH=os.path.join(system_root, 'System32')+';'+system_root,
           PYTHONHOME='', PYTHONPATH='', PYTHONIOENCODING='utf-8',
           OMP_NUM_THREADS='3', MKL_NUM_THREADS='3')
ps = Path(system_root)/'System32/WindowsPowerShell/v1.0/powershell.exe'
steps = []
def run(name, args, timeout):
    start = time.perf_counter()
    with (OUT/(name+'.stdout.txt')).open('wb') as stdout, (OUT/(name+'.stderr.txt')).open('wb') as stderr:
        p = subprocess.run(args, cwd=TARGET, env=env, stdout=stdout, stderr=stderr, timeout=timeout)
    steps.append(dict(name=name, code=p.returncode, seconds=time.perf_counter()-start))
    assert p.returncode == 0, (name, p.returncode)
    print(name, 'PASS', flush=True)

labels = TARGET/'data/own_oct21/labels.jsonl'
before_labels = sha(labels) if labels.exists() else None
run('installer', [str(ps), '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(PATCH/'APPLY_EV_PATCH.ps1'), '-Target', str(TARGET)], 120)
for f in manifest['files']:
    assert sha(TARGET/f['path']) == f['sha256']
assert sha(exe) == manifest['exe_sha256']
assert before_labels == (sha(labels) if labels.exists() else None)
run('complete-selfcheck', [str(ps), '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(TARGET/'CHECK_ON_NEW_PC.ps1')], 600)
photo = Path('C:/Temp/codex-clipboard-779e498d-79c4-4db9-9fd4-61cd3491b753.png')
run('night-diagnostic', [str(exe), 'predict', str(photo), '--mode', 'fast', '--output', str(OUT/'night')], 180)
prediction = json.loads((OUT/'night-diagnostic.stdout.txt').read_text(encoding='utf-8-sig'))
assert prediction['cars']
assert all(len(r['model_ranking']) == 42 for r in prediction['cars'])
assert any('Weltmeister' == a['brand'] for a in prediction['cars'][0]['model_ranking'])
for f in manifest['files']:
    assert sha(TARGET/f['path']) == f['sha256']
report = dict(passed=True, steps=steps, executable_sha256=sha(exe),
              candidate_sha256=manifest['candidate_model_sha256'],
              owner_annotations_preserved=before_labels is not None,
              test_location=str(TARGET), python_in_path=False,
              scope='Separate extracted copy on the same physical Windows PC; no second-PC claim. Night image diagnostic only, not a scored independent accuracy test.')
(OUT/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
