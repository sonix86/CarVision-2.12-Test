"""Finite continuation of the existing feature jobs, two supervised fits only."""
import ctypes,json,os,subprocess,sys,time,traceback
from ctypes import wintypes
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';state_path=OUT/'finish-v24-state.json';started=time.time()
def state(phase,**kw):
    p=state_path.with_suffix('.tmp');p.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),updated=time.time(),**kw),indent=2),encoding='utf-8');p.replace(state_path);print(phase,flush=True)
def run(script,log,timeout):
    with (OUT/log).open('wb') as f:r=subprocess.run([sys.executable,str(ROOT/'tools'/script)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW)
    assert r.returncode==0,script+' failed; see '+log
try:
    with (OUT/'finish-v24-once.json').open('x') as f:json.dump(dict(started=started,pid=os.getpid()),f)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];kernel.WaitForSingleObject.restype=wintypes.DWORD;kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    state('waiting_for_existing_feature_jobs')
    for pid in (1672,11252):
        handle=kernel.OpenProcess(0x100000,False,pid)
        if handle:
            try:assert kernel.WaitForSingleObject(handle,max(1,int((started+5400-time.time())*1000)))==0,'Finite feature wait expired'
            finally:kernel.CloseHandle(handle)
    assert 'FEATURES COMPLETE' in (OUT/'cache-multimodal-v24.log').read_text(encoding='utf-8-sig')
    assert 'OCR FEATURES COMPLETE' in (OUT/'cache-text-v24.log').read_text(encoding='utf-8-sig')
    state('append_private_training_features')
    run('cache_multimodal_v24.py','cache-multimodal-v24-private.log',600)
    state('two_finite_training_rounds')
    run('train_multimodal_v24.py','train-multimodal-v24.log',1200)
    report=json.loads((ROOT/'models/multimodal_v24/training.json').read_text(encoding='utf-8'));rules=report['policy']['release_requirements']
    valid=report['accepted_validation_views']>=rules['validation_confirmation_views_min'] and (report['accepted_validation_precision'] or 0)>=rules['validation_confirmation_precision_min']
    state('validation_ready_for_review' if valid else 'validation_rejected_no_release',summary={k:report[k] for k in ('selected_round','selected_epoch','selected_metrics','confirmed_make_count','accepted_validation_views','accepted_validation_precision')},active_models_unchanged=True)
except Exception as e:
    state('stopped_without_release',error=str(e),traceback=traceback.format_exc());raise
