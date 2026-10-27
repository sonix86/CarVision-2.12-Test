"""One finite local continuation of the already-running fit. No new training/search."""
import argparse,ctypes,json,os,subprocess,sys,time,traceback
from ctypes import wintypes
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs';STATE=OUT/'finish-v23-state.json'
parser=argparse.ArgumentParser();parser.add_argument('--training-pid',required=True,type=int);args=parser.parse_args()
started=time.time();steps=[]

def state(phase,**extra):
    payload=dict(phase=phase,runner_pid=os.getpid(),training_pid=args.training_pid,started_epoch=started,updated_epoch=time.time(),steps=steps,**extra)
    temporary=STATE.with_suffix('.tmp');temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(STATE)
    print(phase,flush=True)

def run(name,script,timeout):
    state(name)
    with (OUT/(name+'.log')).open('wb') as log:
        result=subprocess.run([sys.executable,str(ROOT/'tools'/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW)
    steps.append(dict(name=name,exit_code=result.returncode));assert result.returncode==0,f'{name} failed; see outputs/{name}.log'

try:
    # An exclusive marker prevents accidentally repeating the heldout evaluation.
    marker=OUT/'finish-v23-once.json'
    with marker.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),started_epoch=started,training_pid=args.training_pid),f)
    state('waiting_for_current_training')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=kernel.OpenProcess(0x100000,False,args.training_pid)
    if handle:
        try:assert kernel.WaitForSingleObject(handle,7200000)==0,'Training did not finish within the finite two-hour continuation wait'
        finally:kernel.CloseHandle(handle)
    training=ROOT/'models/make_scope_v23/training.json'
    assert training.exists(),'Training stopped without a selected checkpoint; active weights remain unchanged'
    assert json.loads(training.read_text(encoding='utf-8'))['selected']
    assert not (ROOT/'models/make_scope_v23/evaluation.json').exists(),'Heldout evaluation already exists; do not repeat it'
    run('evaluate-make-v23-final','evaluate_make_scope_v23.py',1800)
    # Packaging enforces the predeclared stop gates before touching the recipient.
    run('package-make-v23-final','package_make_scope_v23.py',900)
    run('verify-make-v23-final','verify_make_update_v23.py',2400)
    report=json.loads((ROOT/'outputs/make23-recipient-verification/archive.json').read_text(encoding='utf-8'))
    assert report['verified']
    state('verified_patch_ready',archive=report)
except Exception as exc:
    state('stopped_without_release',error=str(exc),traceback=traceback.format_exc())
    raise
