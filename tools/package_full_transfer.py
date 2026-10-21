"""Create one current handoff: frozen app, source/history, weights and working data."""
import hashlib,json,shutil,subprocess,time,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
APP=ROOT/'dist/CarVision'
STAGE=ROOT/'outputs/full-transfer-21/CarVision'
ARCHIVE=ROOT/'CarVision-2.1-Full-Transfer.zip'
PRIMARY=['classifier.pt','detector.pt','colour_model.pt','vehicle_model.pt','segmenter.pt','masked_colour.pt','visual_references.pt','generic_brand_backbone.pt','generic_detector_backbone.pt']

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def copy_tree(source,destination,exclude=()):
 destination.mkdir(parents=True,exist_ok=True)
 for path in source.rglob('*'):
  rel=path.relative_to(source)
  if not path.is_file() or '__pycache__' in rel.parts or path.suffix.lower() in {'.pyc','.tmp'}|set(exclude):continue
  target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)

def main():
 started=time.time()
 if STAGE.exists() or ARCHIVE.exists():raise RuntimeError('Handoff stage/archive already exists; inspect before replacing it')
 STAGE.mkdir(parents=True)
 print('Copy frozen EXE and runtime libraries',flush=True)
 shutil.copy2(APP/'CarVision.exe',STAGE/'CarVision.exe');copy_tree(APP/'_internal',STAGE/'_internal')
 for folder in ['carvision','tests','tools','handoff','docs','models','samples','data']:
  print('Copy',folder,flush=True);copy_tree(ROOT/folder,STAGE/folder)
 # Original downloads duplicate already expanded images. Preserve source URLs,
 # metadata and working photographs; exclude the duplicate compressed shards.
 copy_tree(ROOT/'datasets',STAGE/'datasets',exclude={'.zip','.parquet'})
 # Preserve current GUI annotations instead of the developer's empty template.
 copy_tree(APP/'data/images',STAGE/'data/images')
 shutil.copy2(APP/'data/labels.jsonl',STAGE/'data/labels.jsonl')
 for path in ROOT.iterdir():
  if path.is_file() and (path.suffix.lower() in {'.py','.ps1','.txt','.md','.json','.spec'} or path.name in {'.gitignore','.gitattributes'}):shutil.copy2(path,STAGE/path.name)
 for name in ['CHECK_ON_NEW_PC.cmd','CHECK_ON_NEW_PC.ps1']:shutil.copy2(ROOT/'handoff'/name,STAGE/name)
 for path in (ROOT/'output/pdf').glob('*.pdf'):shutil.copy2(path,STAGE/'docs'/path.name)
 assignment=Path('C:/Users/Miku/Downloads/Telegram Desktop/CSAI-3210_Midterm_CarBrandClassification.pdf')
 if assignment.exists():shutil.copy2(assignment,STAGE/'docs'/assignment.name)
 development=STAGE/'Development';development.mkdir()
 print('Preserve real Git history',flush=True)
 subprocess.run(['git','bundle','create',str(development/'CarVision-history.bundle'),'--all'],cwd=ROOT,check=True,capture_output=True)
 verified=subprocess.run(['git','bundle','verify',str(development/'CarVision-history.bundle')],cwd=ROOT,check=True,capture_output=True,text=True)
 (development/'GIT_BUNDLE_VERIFY.txt').write_text(verified.stdout+verified.stderr,encoding='utf-8')
 revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 (development/'SOURCE_REVISION.json').write_text(json.dumps(dict(commit=revision,history='Actual local Git history; no remote publishing or reconstructed commits'),indent=2),encoding='utf-8')
 hashes={name:digest(ROOT/'models'/name) for name in PRIMARY}
 for name,sha in hashes.items():assert digest(STAGE/'models'/name)==sha
 (development/'ACTIVE_MODEL_SHA256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
 for name in ['tests-v21.log','training-smoke-v21.log','training-smoke-v21.err','frozen-v21-check.json','portable-v21-check.log','portable-v21-check.err','archive-v21-check.log']:
  shutil.copy2(ROOT/'outputs'/name,development/name)
 (development/'PREVIOUS_CHECKS_SCOPE.txt').write_text('Historical source/frozen engineering checks were run on the developer computer. Current-machine checks must be performed with CHECK_ON_NEW_PC.cmd; do not attribute old results to a different device.\n',encoding='utf-8')
 files=[p for p in STAGE.rglob('*') if p.is_file()]
 info=dict(version='2.1',source_commit=revision,created_local=time.strftime('%Y-%m-%d %H:%M:%S'),files_before_manifest=len(files),uncompressed_bytes_before_manifest=sum(p.stat().st_size for p in files),trained_weights_included=True,raw_working_data_included=True,current_gui_labels_preserved=True,excluded=['Old delivery ZIPs','Development virtual environment','Build directories','Duplicate downloaded ZIP/parquet archives','Derived training cache','Previous extracted test/app copies','Old runtime outputs'],scope='Full current application and project materials. No new training, no change of model quality, no claim of a second-computer test.')
 (STAGE/'TRANSFER_CONTENTS.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
 print('Hash every handoff file',flush=True)
 manifest=[]
 for path in sorted(STAGE.rglob('*')):
  if path.is_file() and path.name!='SHA256SUMS.txt':manifest.append(digest(path)+'  '+path.relative_to(STAGE).as_posix())
 (STAGE/'SHA256SUMS.txt').write_text('\n'.join(manifest)+'\n',encoding='utf-8')
 print('Create full ZIP',len(manifest),'files',flush=True)
 with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
  for path in sorted(STAGE.rglob('*')):
   if path.is_file():archive.write(path,'CarVision/'+path.relative_to(STAGE).as_posix())
 print(json.dumps(dict(archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,seconds=time.time()-started)),flush=True)

if __name__=='__main__':main()
