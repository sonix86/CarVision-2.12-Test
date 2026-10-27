"""Apply the actual patch, test its real EXE offline, then seal one verified ZIP."""
import hashlib,json,os,subprocess,sys,time,zipfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
stage=ROOT/'outputs/CarVision-Make-Update-2.3';target=ROOT/'outputs/Transfer extracted София/CarVision'
out=ROOT/'outputs/make23-recipient-verification';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((stage/'UPDATE_MANIFEST.json').read_text(encoding='utf-8'))
env=os.environ.copy();system=Path(next(v for k,v in env.items() if k.lower()=='systemroot'))
env.update(PATH=str(system/'System32')+';'+str(system),PYTHONHOME='',PYTHONPATH='',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='3',MKL_NUM_THREADS='3')
ps=system/'System32/WindowsPowerShell/v1.0/powershell.exe';steps=[]
def run(name,args,timeout):
    start=time.perf_counter()
    with (out/(name+'.stdout.txt')).open('wb') as stdout,(out/(name+'.stderr.txt')).open('wb') as stderr:
        p=subprocess.run(args,cwd=target,env=env,stdout=stdout,stderr=stderr,timeout=timeout,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    steps.append(dict(name=name,code=p.returncode,seconds=time.perf_counter()-start));assert p.returncode==0,(name,p.returncode)
    print(name,'PASS',flush=True)
labels=target/'data/own_oct21/labels.jsonl';before=sha(labels)
run('install',[str(ps),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(stage/'APPLY_UPDATE_2_3.ps1'),'-Target',str(target)],300)
assert sha(labels)==before
for f in manifest['files']:assert sha(target/f['path'])==f['sha256'],f['path']
run('selfcheck',[str(ps),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(target/'CHECK_ON_NEW_PC.ps1')],1200)
exe=target/'CarVision.exe';photo=target/'samples/images/03.jpg'
run('make-parity',[str(exe),'predict',str(photo),'--mode','fast','--output',str(out/'parity')],240)
actual=json.loads((out/'make-parity.stdout.txt').read_text(encoding='utf-8-sig'))
from carvision.core import Pipeline
import torch
torch.set_num_threads(3)
expected=Pipeline(target/'models',mode='fast').predict(photo)
def signature(result):
    return [(r['brand'],r.get('model'),r.get('model_candidate'),r.get('make_evidence'),r['colour'],r['box'],r.get('contours')) for r in result['cars']]
assert signature(actual)==signature(expected),'EXE/source results differ'
renamed=out/'independent-name.jpg';shutil.copy2(photo,renamed)
run('renamed-photo',[str(exe),'predict',str(renamed),'--mode','fast','--output',str(out/'renamed')],240)
renamed_result=json.loads((out/'renamed-photo.stdout.txt').read_text(encoding='utf-8-sig'))
assert signature(actual)==signature(renamed_result),'Prediction depends on the photograph filename'
assert all(r.get('make_first') and r.get('make_evidence') for r in actual['cars'])
for r in actual['cars']:
    if r['brand']!='unknown':assert all(v['brand']==r['brand'] for v in r.get('model_alternatives',[]))
# Diagnostic only AFTER selecting and evaluating the final weights. Never feeds fitting.
night=Path('C:/Temp/codex-clipboard-779e498d-79c4-4db9-9fd4-61cd3491b753.png')
if night.exists():run('night-diagnostic',[str(exe),'predict',str(night),'--mode','fast','--output',str(out/'night')],240)
for f in manifest['files']:assert sha(target/f['path'])==f['sha256']
assert sha(labels)==before
verification=dict(passed=True,steps=steps,source_exe_predictions_equal=True,filename_independent=True,owner_annotations_preserved=True,python_in_path=False,executable_sha256=sha(exe),
                  make_model_sha256=manifest['make_model_sha256'],scope='Separate complete recipient folder on the same physical PC. Not a second-PC or field-accuracy claim.')
(out/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
(stage/'VERIFICATION.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
archive=ROOT/'CarVision-Make-Update-2.3.zip';assert not archive.exists()
paths=sorted(p for p in stage.rglob('*') if p.is_file())
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
    for p in paths:z.write(p,p.relative_to(stage).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for f in manifest['files']:assert hashlib.sha256(z.read('payload/'+f['path'])).hexdigest()==f['sha256']
archive_sha=sha(archive);archive.with_suffix('.zip.sha256').write_text(archive_sha+'  '+archive.name+'\n',encoding='ascii')
summary=dict(verified=True,path=str(archive),bytes=archive.stat().st_size,sha256=archive_sha,files=len(paths),payload_files=len(manifest['files']))
(out/'archive.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
