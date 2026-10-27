"""Verify the integrated EXE, then seal and hash every portable archive entry."""
import hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));target=ROOT/'outputs/CarVision-2.6-Full/CarVision';out=ROOT/'outputs/full26-recipient-verification';out.mkdir(exist_ok=True)
with (out/'once.json').open('x') as f:json.dump(dict(started=time.time()),f)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
env=os.environ.copy();system=Path(next(v for k,v in env.items() if k.lower()=='systemroot'));env.update(PATH=str(system/'System32')+';'+str(system),PYTHONHOME='',PYTHONPATH='',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3');ps=system/'System32/WindowsPowerShell/v1.0/powershell.exe';steps=[]
def run(name,args,timeout=300,cwd=target):
    started=time.perf_counter()
    with (out/(name+'.stdout.txt')).open('wb') as stdout,(out/(name+'.stderr.txt')).open('wb') as stderr:p=subprocess.run(args,cwd=cwd,env=env,stdout=stdout,stderr=stderr,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    steps.append(dict(name=name,exit_code=p.returncode,wall_seconds=time.perf_counter()-started));assert p.returncode==0,(name,p.returncode);print(name,'PASS',flush=True)
labels=target/'data/own_oct21/labels.jsonl';before=sha(labels);active=json.loads((target/'Development/ACTIVE_MODEL_SHA256.json').read_text())
for name,digest in active.items():assert sha(target/'models'/name)==digest,name
run('selfcheck',[str(ps),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(target/'CHECK_ON_NEW_PC.ps1')],1200)
exe=target/'CarVision.exe';photo=target/'samples/images/03.jpg';run('source-parity',[str(exe),'predict',str(photo),'--mode','fast','--output',str(out/'sample')]);actual=json.loads((out/'source-parity.stdout.txt').read_text(encoding='utf-8-sig'))
from carvision.core import Pipeline
import torch
torch.set_num_threads(3);p=Pipeline(target/'models',mode='fast');expected=p.predict(photo)
def signature(d):return [(r['brand'],r['colour'],r.get('make_evidence'),r.get('model'),r.get('model_alternatives'),r['box'],r.get('contours')) for r in d['cars']]
assert signature(actual)==signature(expected),'New EXE is stale or differs from source'
renamed=out/'independent-random-name.jpg';shutil.copy2(photo,renamed);run('renamed-photo',[str(exe),'predict',str(renamed),'--mode','fast','--output',str(out/'renamed')]);assert signature(actual)==signature(json.loads((out/'renamed-photo.stdout.txt').read_text(encoding='utf-8-sig')))
diagnostics=[];timings=[];old_exe=ROOT/'outputs/Проверка обновления 2.5.3/CarVision/CarVision.exe'
for number in (2,5,16):
    image=Path(f'D:/машинки/photo_{number}_2026-10-25_20-44-40.jpg');name=f'owner-{number}';run(name,[str(exe),'predict',str(image),'--mode','fast','--output',str(out/name)]);d=json.loads((out/(name+'.stdout.txt')).read_text(encoding='utf-8-sig'));r=max(d['cars'],key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]),default={});e=r.get('make_evidence') or {};diagnostics.append(dict(photo=number,brand=r.get('brand'),candidate=e.get('candidate'),confidence=e.get('confidence'),colour=r.get('colour'),analysis_seconds=d.get('seconds'),expected_make='BYD' if number==5 else 'Toyota' if number==16 else None,scope='Known development diagnostic, excluded from fitting/selection; not an independent phone exam'))
    run('baseline-'+name,[str(old_exe),'predict',str(image),'--mode','fast','--output',str(out/('baseline-'+name))],cwd=old_exe.parent);b=json.loads((out/('baseline-'+name+'.stdout.txt')).read_text(encoding='utf-8-sig'));timings.append(dict(photo=number,old_analysis_seconds=b.get('seconds'),new_analysis_seconds=d.get('seconds'),note='One cold command per version, same PC; wall/startup timings separately in steps, not a robust speed benchmark'))
    print('DIAGNOSTIC',json.dumps(diagnostics[-1]),flush=True)
night=Path('C:/Temp/codex-clipboard-779e498d-79c4-4db9-9fd4-61cd3491b753.png')
if night.exists():
    run('night',[str(exe),'predict',str(night),'--mode','fast','--output',str(out/'night')]);d=json.loads((out/'night.stdout.txt').read_text(encoding='utf-8-sig'));r=max(d['cars'],key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]),default={});diagnostics.append(dict(photo='known-night',brand=r.get('brand'),candidate=(r.get('make_evidence') or {}).get('candidate'),colour=r.get('colour'),scope='Unscored known user diagnostic'))
for name,digest in active.items():assert sha(target/'models'/name)==digest,name
assert sha(labels)==before
verification=dict(passed=True,steps=steps,diagnostics=diagnostics,timings=timings,source_exe_equal=True,filename_independent=True,owner_annotations_preserved=True,python_in_path=False,executable_sha256=sha(exe),quality_goal_achieved=False,scope='Engineering checks on this physical PC. Correct car recognition is assessed separately. No second-PC, 100%, all-model or blind-phone claim.')
(out/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8');(target/'Development/EV26/RECIPIENT_VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
status='CarVision 2.6: технические проверки пройдены. Ошибки распознавания возможны.\nНовая модель включена: '+str(json.loads((target/'models/model_version.json').read_text(encoding='utf-8'))['new_training_promoted'])+'\nКонтрольные ответы:\n'+'\n'.join(str(r) for r in diagnostics)+'\nПолная инструкция: GUIDE_FULL_2_6_RU.txt.\n'
(target/'QUALITY_STATUS_RU.txt').write_text(status,encoding='utf-8')
# Exclude generated test outputs and backups from the transferable snapshot.
paths=sorted(p for p in target.rglob('*') if p.is_file() and not any(part in ('outputs','patch-backups','__pycache__') for part in p.relative_to(target).parts) and p.name not in ('SHA256SUMS.txt','TRANSFER_CONTENTS.json'))
manifest={p.relative_to(target).as_posix():sha(p) for p in paths};(target/'TRANSFER_CONTENTS.json').write_text(json.dumps(dict(version='2.6',files=manifest,excludes=['outputs','patch-backups','__pycache__'],technical_verification='Development/EV26/RECIPIENT_VERIFICATION.json'),indent=2),encoding='utf-8');(target/'SHA256SUMS.txt').write_text('\n'.join(v+'  '+k for k,v in manifest.items())+'\n',encoding='utf-8');paths += [target/'TRANSFER_CONTENTS.json',target/'SHA256SUMS.txt']
archive=ROOT/'CarVision-2.6-Full.zip';assert not archive.exists(),'Never overwrite an already sealed release'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
    z.write(ROOT/'outputs/CarVision-2.6-Full/READ_FIRST_RU.txt','READ_FIRST_RU.txt')
    for pth in paths:z.write(pth,'CarVision/'+pth.relative_to(target).as_posix())
print('ARCHIVE WRITTEN; VERIFYING EVERY ENTRY',flush=True)
with zipfile.ZipFile(archive) as z:
    for info in z.infolist():
        h=hashlib.sha256()
        with z.open(info) as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        name=info.filename.removeprefix('CarVision/')
        if name in manifest:assert h.hexdigest()==manifest[name],name
digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii');report=dict(verified=True,path=str(archive),bytes=archive.stat().st_size,sha256=digest,files=len(paths)+1,all_entries_read_and_crc_checked=True,payload_sha256_verified=True);(out/'archive.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
