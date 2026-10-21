"""Refresh handoff tooling without recopied datasets or any model changes."""
import hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];stage=ROOT/'outputs/full-transfer-21/CarVision';archive=ROOT/'CarVision-2.1-Full-Transfer.zip';pending=archive.with_suffix('.building.zip')
if not stage.exists() or pending.exists():raise RuntimeError('Inspect current stage before refreshing')
for directory in ['tools','handoff']:
 for path in (ROOT/directory).glob('*'):
  if path.is_file():shutil.copy2(path,stage/directory/path.name)
for name in ['CHECK_ON_NEW_PC.ps1','CHECK_ON_NEW_PC.cmd']:shutil.copy2(ROOT/'handoff'/name,stage/name)
dev=stage/'Development';bundle=dev/'CarVision-history-new.bundle'
subprocess.run(['git','bundle','create',str(bundle),'--all'],cwd=ROOT,check=True,capture_output=True)
bundle.replace(dev/'CarVision-history.bundle')
check=subprocess.run(['git','bundle','verify',str(dev/'CarVision-history.bundle')],cwd=ROOT,check=True,capture_output=True,text=True)
(dev/'GIT_BUNDLE_VERIFY.txt').write_text(check.stdout+check.stderr,encoding='utf-8')
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
(dev/'SOURCE_REVISION.json').write_text(json.dumps(dict(commit=revision,history='Actual local Git history; no remote publishing or reconstructed commits'),indent=2),encoding='utf-8')
p=stage/'TRANSFER_CONTENTS.json';info=json.loads(p.read_text(encoding='utf-8'));info['source_commit']=revision;p.write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
manifest=[]
for path in sorted(stage.rglob('*')):
 if path.is_file() and path.name!='SHA256SUMS.txt':
  h=hashlib.sha256()
  with path.open('rb') as f:
   for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
  manifest.append(h.hexdigest()+'  '+path.relative_to(stage).as_posix())
(stage/'SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n',encoding='utf-8')
print('Create corrected full archive',flush=True)
with zipfile.ZipFile(pending,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
 for path in sorted(stage.rglob('*')):
  if path.is_file():z.write(path,'CarVision/'+path.relative_to(stage).as_posix())
pending.replace(archive)
print('Corrected ZIP complete',archive.stat().st_size,flush=True)
