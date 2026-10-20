"""Package verified portable distribution, sources and hashes, excluding local test datasets."""
import hashlib
import shutil
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1];app=root/'dist/CarVision'
if not (app/'CarVision.exe').exists():raise SystemExit('Build EXE first')
src=app/'source';src.mkdir(exist_ok=True)
for name in ['carvision','tests','tools']:
    shutil.copytree(root/name,src/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for name in ['launch.py','build.ps1','requirements.txt','requirements-lock.txt','README.md','CHECKLIST.md','THIRD_PARTY.md','TRANSFER.txt','VALIDATION.md','analysis.example.json','.gitignore']:
    shutil.copy2(root/name,src/name)
(src/'data/images').mkdir(parents=True,exist_ok=True)
(src/'data/labels.jsonl').write_text('',encoding='utf-8')
(src/'models').mkdir(exist_ok=True)
shutil.copytree(root/'samples',app/'samples',dirs_exist_ok=True)
shutil.copytree(root/'samples',src/'samples',dirs_exist_ok=True)
for name in ['README.md','CHECKLIST.md','THIRD_PARTY.md','TRANSFER.txt','VALIDATION.md','requirements-lock.txt','analysis.example.json']:
    shutil.copy2(root/name,app/name)
files=sorted(p for p in app.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(app).as_posix() for p in files)+'\n'
(app/'SHA256SUMS.txt').write_text(manifest,encoding='utf-8')
output=root/'TashkentCarVision-Portable.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in sorted(app.rglob('*')):
        if p.is_file():z.write(p,'CarVision/'+p.relative_to(app).as_posix())
print(output,round(output.stat().st_size/1024**2,1),'MiB')
with zipfile.ZipFile(output) as z:
    bad=z.testzip()
    if bad:raise RuntimeError('Archive CRC failed: '+bad)
print('Archive CRC passed; all portable files and editable sources included.')
