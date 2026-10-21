"""Verify and extract the ACTUAL delivery ZIP, then run the recipient's Windows check."""
import hashlib,json,os,subprocess,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
archive=ROOT/'CarVision-2.1-Full-Transfer.zip'
destination=ROOT/'outputs/Transfer extracted София'
if destination.exists():raise RuntimeError('Fresh extraction directory required')
destination.mkdir()
start=time.time();count=0
with zipfile.ZipFile(archive) as z:
 names=z.namelist()
 required=['CarVision/CarVision.exe','CarVision/CHECK_ON_NEW_PC.cmd','CarVision/CHECK_ON_NEW_PC.ps1','CarVision/HANDOFF_GUIDE_RU.txt','CarVision/Development/CarVision-history.bundle','CarVision/docs/CarVision_Report_v2.1_DRAFT.pdf']
 assert all(p in names for p in required)
 expected={}
 for line in z.read('CarVision/SHA256SUMS.txt').decode('utf-8').splitlines():
  sha,name=line.split('  ',1);expected['CarVision/'+name]=sha
 assert set(names)==set(expected)|{'CarVision/SHA256SUMS.txt'},'Unlisted or missing files'
 for name in names:
  target=destination/name
  assert target.resolve().is_relative_to(destination.resolve()),'Invalid archive path'
  target.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
  # Reading every ZIP entry to EOF verifies its CRC as well as SHA256.
  with z.open(name) as source,target.open('wb') as output:
   for block in iter(lambda:source.read(1024*1024),b''):h.update(block);output.write(block)
  if name in expected:assert h.hexdigest()==expected[name],name
  count+=1
  if count%2500==0:print('Extracted and verified',count,'/',len(names),flush=True)
app=destination/'CarVision'
subprocess.run(['git','bundle','verify',str(app/'Development/CarVision-history.bundle')],cwd=ROOT,check=True,capture_output=True)
assert (app/'data/labels.jsonl').read_bytes()==(ROOT/'dist/CarVision/data/labels.jsonl').read_bytes(),'Current user annotations changed'
for path in (ROOT/'dist/CarVision/data/images').rglob('*'):
 if path.is_file():assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256((app/'data/images'/path.relative_to(ROOT/'dist/CarVision/data/images')).read_bytes()).digest()
print('ZIP CRC, all SHA256, Git bundle and user annotations verified. Run Windows recipient check.',flush=True)
powershell=Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
result=subprocess.run([str(powershell),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(app/'CHECK_ON_NEW_PC.ps1')],cwd=app,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=1500)
(ROOT/'outputs/full-transfer-selfcheck.stdout.txt').write_text(result.stdout,encoding='utf-8');(ROOT/'outputs/full-transfer-selfcheck.stderr.txt').write_text(result.stderr,encoding='utf-8')
print(result.stdout,flush=True)
if result.returncode:raise RuntimeError('Recipient check failed; see full-transfer-selfcheck logs')
reports=sorted((app/'outputs/transfer-selfcheck').glob('*/SELF_CHECK_RESULT.json'))
assert reports,'Recipient result not generated'
actual=json.loads(reports[-1].read_text(encoding='utf-8-sig'));assert actual['passed'] and not actual['quick']
summary=dict(passed=True,zip=str(archive),zip_bytes=archive.stat().st_size,files=count,crc_and_all_sha256_passed=True,extracted_to=str(app),real_exe_checked=True,synthetic_training_checked=True,source_history_verified=True,current_user_labels_preserved=True,recipient_report=str(reports[-1]),seconds=time.time()-start,scope='Actual ZIP extraction with Unicode/spaces, same physical developer computer. Recipient must run CHECK_ON_NEW_PC.cmd on the other device; no second-device claim.')
(ROOT/'outputs/full-transfer-verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8');print('FULL TRANSFER VERIFIED',flush=True)
