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
for name in ['launch.py','build.ps1','requirements.txt','requirements-data.txt','requirements-lock.txt','README.md','CHECKLIST.md','REQUIREMENTS_AUDIT.md','THIRD_PARTY.md','TRANSFER.txt','VALIDATION.md','PROTOTYPE_VALIDATION.md','analysis.example.json','.gitignore']:
    shutil.copy2(root/name,src/name)
(src/'data/images').mkdir(parents=True,exist_ok=True)
(src/'data/labels.jsonl').write_text('',encoding='utf-8')
(src/'models').mkdir(exist_ok=True)
metadata=src/'data/public_prototype';metadata.mkdir(exist_ok=True)
for name in ['labels.jsonl','provenance.json','colour_reference_review.json','colour_development_review.json','colour_dataset_manifest.json','vcor_download_provenance.json','vcor_source_metadata.json']:shutil.copy2(root/'data/public_prototype'/name,metadata/name)
for destination in [app/'data/regional',src/'data/regional']:
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['labels.jsonl','provenance.json','source_manifest_snapshot.json']:shutil.copy2(root/'data/regional'/name,destination/name)
shutil.copytree(root/'samples',app/'samples',dirs_exist_ok=True)
shutil.copytree(root/'samples',src/'samples',dirs_exist_ok=True)
for name in ['README.md','CHECKLIST.md','REQUIREMENTS_AUDIT.md','THIRD_PARTY.md','TRANSFER.txt','VALIDATION.md','PROTOTYPE_VALIDATION.md','requirements-lock.txt','analysis.example.json']:
    shutil.copy2(root/name,app/name)
(app/'docs').mkdir(exist_ok=True)
shutil.copy2(root/'output/pdf/CarVision_Prototype_Report.pdf',app/'docs/CarVision_Prototype_Report.pdf')
def included(path):return not path.relative_to(app).as_posix().startswith(('outputs/','models/backups/','cache/'))
files=sorted(p for p in app.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt' and included(p))
manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(app).as_posix() for p in files)+'\n'
(app/'SHA256SUMS.txt').write_text(manifest,encoding='utf-8')
output=root/'TashkentCarVision-v1.3-Portable.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in sorted(app.rglob('*')):
        if p.is_file() and included(p):z.write(p,'CarVision/'+p.relative_to(app).as_posix())
print(output,round(output.stat().st_size/1024**2,1),'MiB')
with zipfile.ZipFile(output) as z:
    bad=z.testzip()
    if bad:raise RuntimeError('Archive CRC failed: '+bad)
print('Archive CRC passed; all portable files and editable sources included.')
