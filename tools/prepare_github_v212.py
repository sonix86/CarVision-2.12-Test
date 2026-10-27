"""Prepare a new private repository and release without changing sealed 2.12 bytes."""
import os,json,shutil,hashlib,zipfile,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PUB=Path('D:/Projects/CarVision-GitHub-2.12-Submission')
APP=ROOT/'outputs/CarVision-2.12-Test/CarVision'
URL='https://github.com/sonix86/CarVision-2.12-Test'
ASSETS=PUB/'.publication/assets'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,t):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(t,'utf-8')
def copy(a,b):b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
def main():
 ASSETS.mkdir(parents=True,exist_ok=True)
 for folder in ('carvision','tools','tests'):
  for p in (ROOT/folder).glob('*.py'):copy(p,PUB/folder/p.name)
 for p in ROOT.glob('requirements*.txt'):copy(p,PUB/p.name)
 for name in ('launch.py','build.ps1','THIRD_PARTY.md','MODEL_AND_COLOUR.md','INSTANCE_ANALYSIS.md','TARGET_EV_COVERAGE_RU.txt'):
  copy(ROOT/name,PUB/name)
 # Exact source/weight pairing from the delivered app, never an old root checkpoint.
 for folder in ('carvision','models','data','datasets','samples','licenses','Development'):
  for p in (APP/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:
    q=PUB/folder/p.relative_to(APP/folder)
    if q.exists():
     if digest(p)==digest(q):continue
     copy(p,q)
    else:q.parent.mkdir(parents=True,exist_ok=True);os.link(p,q)
 for folder in ('models','data'):
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.txt') and not any(x in p.parts for x in ('images','backups')):
    q=PUB/folder/p.relative_to(ROOT/folder)
    # Runtime metadata remains exactly the frozen delivered version.
    if not q.exists():copy(p,q)
 for name in ('GUIDE_2_12_RU.txt','AUDIT_RU.txt','BUILD_2_12.json','CHECK_ON_NEW_PC.cmd','CHECK_ON_NEW_PC.ps1'):
  copy(APP/name,PUB/'release'/name)
 evidence=PUB/'docs/evidence';evidence.mkdir(parents=True,exist_ok=True)
 for src in ('outputs/seal-v212.json','outputs/exe-v212-verification/verification.json','outputs/tests-audit-v212-final.log','outputs/pipeline-audit-v212/audit.json','models/phone_v32/regression.json','outputs/ev-specialist-v212/baseline.json'):
  p=ROOT/src;name=str(Path(src).parent).replace('\\','_').replace('/','_')+'_'+p.name;copy(p,evidence/name)
 old=(PUB/'.gitignore').read_text('utf-8');write(PUB/'.gitignore',old+'\n# Release payloads are served from GitHub Releases, not ordinary Git.\n.publication/\n.release-downloads/\nDevelopment/\ndatasets/\n*.exe\n*.dll\n*.download\n*.zip.*\n*.env\n.env*\n**/images/\n**/*.jpg\n**/*.jpeg\n**/*.png\n**/*.pt\n**/*.pth\n**/*.parquet\n**/*.bin\n')
 readme='''# CarVision 2.12 TEST

Offline Windows application for detecting visible cars and predicting badge brand
and body colour in Tashkent phone photographs. Project owner: Sofia Rakhmankulova.

Repository: https://github.com/sonix86/CarVision-2.12-Test

## Download the exact portable application

Open [release v2.12-test](https://github.com/sonix86/CarVision-2.12-Test/releases/tag/v2.12-test).
Download both `CarVision-2.12-Test-Full.zip.001` and `.002`, together with
`RESTORE_FULL.ps1` and `RESTORE_FULL.cmd`, into one new folder.
Run `RESTORE_FULL.cmd`. It verifies each part, rebuilds the original ZIP and
extracts it. Open `CarVision/CarVision.exe` and confirm the title is **2.12 TEST**.
Keep the whole folder. No Python, pip, old patches or internet is required for
ordinary inference. The original 2,936,560,562-byte ZIP is reconstructed exactly:
`cd5644e7f6bdd5c046e5130e0db6c9065e802c80c4dd9b7ecdddb0fb9540995c`.

Open photo -> Analyse photo -> select a car row/outline. Copy the assignment
answer as English `brand colour` pairs. Results include JSON, TXT and an
annotated image. For a technical check run `CHECK_ON_NEW_PC.cmd` inside the
portable application. PASS verifies execution, not recognition accuracy.

## Run source on a clean Windows machine

Python 3.12 x64 is the tested source runtime. In PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
# Download release files using the browser, or authenticated GitHub CLI:
gh release download v2.12-test --repo sonix86/CarVision-2.12-Test --dir .release-downloads
.venv\\Scripts\\python.exe tools/restore_release_v212.py --downloads .release-downloads --training
.venv\\Scripts\\python.exe launch.py gui
.venv\\Scripts\\python.exe launch.py predict "C:\\photos\\car.jpg" --mode fast --output outputs\\exam
```

The restorer reads the frozen release manifest, verifies hashes, and restores
`models`, `data`, `datasets` and `samples` from the exact portable archive. With
`--training`, it also restores the frozen feature/crop cache and source manifests.
It refuses path traversal and refuses to replace an existing differing file.
This private repository requires an authorized GitHub account to download assets.

## Architecture, data and measured limits

Generic COCO Mask R-CNN instances -> mask-aware crops -> frozen general DINOv2
ViT-S/14 and CLIP ViT-B/32 features -> own 82-brand cosine head plus calibrated
grouped retrieval, an own 43-family head and a limited own correction head.
Colour uses separately trained project heads and masked-paint/illumination
evidence. No external inference API or ready-made automotive brand classifier.

Version 2.12 fixes model loading, detector-only routing, colour interfaces,
paired encoder integrity and unsafe training output reuse. Its recognition
weights are unchanged from 2.11. It is a tested experimental application, not
a guarantee of correct recognition. All 325 requested model names are not trained.

On the reused 942-crop development benchmark, correct confirmed makes are
626/942 (66.5%) on whole views and 478/942 (50.7%) on artificial phone views.
Confirmed precision 626/649 (96.5%) must not be mistaken for overall accuracy.
On a 174-image target-make subset, correct confirmed makes are 90/174 and 47/174.
These are reused development crop tests, not blind Tashkent photograph accuracy.
AIQAR, Voyah, some Chevrolet Tracker cases, exact model labels and colours remain
problematic. 150 engineering tests passed; see `docs/evidence` and the report.

The frozen brand cache contains 3,851 crop records (2,205 training, 704 validation,
942 reused development-test). Public sources dominate. Known local corrections
are TRAIN/development, not an independent test. Release data include the available
source photographs/labels and frozen RGB/mask/features; missing original files
are documented rather than fabricated. Preserve `group`, source and split fields.
Restoring files does not certify every imported label or recreate missing originals.

## Training and evaluation

Source scripts and fixed experiment policies are in `tools` and `models`.
General encoders remain frozen in the deployed classifier. The own cosine heads
use weighted CE, AdamW, label smoothing and public-teacher distillation.
`tools/audit_ev_baseline_v212.py` measures the deployed route from the frozen cache.
`tools/verify_exe_v212.py` records actual EXE checks from the development environment.
Historical fitting scripts use one-shot guards and original cache paths: do not
delete their history or rerun them claiming an exact independent reproduction.

```powershell
.venv\\Scripts\\python.exe -m unittest discover -s tests
.venv\\Scripts\\python.exe tools/audit_ev_baseline_v212.py
# Optional personal candidate training, separate from the advanced active head:
.venv\\Scripts\\python.exe launch.py train --labels data/reviewed_owner_v212/labels.jsonl --output models/my_new_candidate
```

Personal training writes a separate candidate; it does not automatically replace
the advanced head. Do not train into the shipped `models` directory. Building a
Windows EXE uses `build.ps1`; the frozen release remains the tested reference.

## Submission and provenance

Read `docs/ASSIGNMENT_COMPLIANCE.md`, `docs/CarVision-2.12-Report.pdf`, source
attribution manifests and `THIRD_PARTY.md`. This new repository is private because
the release includes private phone photographs; the instructor must be granted
access separately. No unrelated existing GitHub repository was changed.

The original local Git history is retained. Its author metadata is not rewritten
or backdated. This publication was prepared on 28 October 2026, after the stated
27 October 23:59 deadline; it cannot retroactively meet that deadline. The new
`midterm` tag identifies this snapshot, not a claim of timely submission.

OpenAI Codex assisted with implementation, data preparation/review, experiments,
debugging, verification, packaging, documentation and publication. Student work
and AI assistance must be disclosed honestly. Originality relative to other teams,
student understanding, instructor access and live-test performance require human
verification. Account passwords and authentication tokens are never project files.
'''
 write(PUB/'README.md',readme)
 write(PUB/'docs/ASSIGNMENT_COMPLIANCE.md','''# Assignment compliance - actual 2.12 status

Prepared 28 October 2026. Technical publication is not a claim of complete marks.

| Requirement | Evidence / outstanding limitation |
|---|---|
| Offline per-car brand and 8 colours | GUI/CLI and frozen Windows EXE; prediction errors remain |
| Several cars and camera views | Generic instance masks, optional region and tiled/detail modes |
| Own brand/colour decisions | Own heads; general-purpose COCO/DINO/CLIP initialization disclosed |
| Own Tashkent photos primary | NOT MET: public sources dominate; local set small and reused |
| Data/labels and source licences | Release archives, manifests, frozen crops/features, attribution |
| Separate validation and real phone test | Grouped development evidence; no independent final phone score |
| All Chinese EV models / 80% guarantee | NOT ACHIEVED and not claimed; 82 make labels / 43 families |
| Report, seven sections, at most 6 pages, repository URL | Six-page report included; limits and AI use disclosed |
| Code, trained weights, installation and clean run | Source + checksummed release restore + requirements + README |
| Meaningful history | Existing 33 local commits preserved; author metadata unchanged |
| Work of both members | User reports Sofia as sole student contributor; no second author invented |
| Final midterm tag | New real snapshot tag, created at actual publication time |
| Instructor access | Private repository: instructor invitation still requires their GitHub login |
| Submission by 27 Oct 23:59 | CANNOT BE MET RETROACTIVELY by publication on 28 October |
| Different architecture / explain own code | Must be verified by student and instructor |

The delivered ZIP is byte-for-byte the previously verified 2.12 release.
New documentation and packaging do not constitute retraining or a quality gain.
''')
 write(PUB/'THIRD_PARTY.md','# Current 2.12 additions\n\nThe active route uses general Meta DINOv2 ViT-S/14 (Apache-2.0) and the general LAION CLIP ViT-B/32 visual encoder (MIT), with own supervised automotive make/family heads. The generic COCO Mask R-CNN ResNet50-FPN-v2 supplies instance geometry. Colour decisions use project-trained VCoR heads and original paint/illumination evidence. This header supersedes older architecture counts below. See models/model_version.json, source manifests, README and docs/evidence.\n\n'+(PUB/'THIRD_PARTY.md').read_text('utf-8'))
 # Two native split volumes, preserving the exact already-delivered ZIP bytes.
 original=ROOT/'outputs/CarVision-2.12-Test-Full.zip';expected=json.loads((ROOT/'outputs/seal-v212.json').read_text('utf-8'));assert digest(original)==expected['sha256']
 parts=[]
 with original.open('rb') as f:
  for j in (1,2):
   p=ASSETS/(original.name+f'.{j:03}');left=1_500_000_000
   if not p.exists():
    with p.open('wb') as out:
     while left:
      b=f.read(min(8*1024*1024,left))
      if not b:break
      out.write(b);left-=len(b)
   else:f.seek(p.stat().st_size,1)
   parts.append(dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)));print('SPLIT_READY',parts[-1]['name'],flush=True)
  assert not f.read(1)
 training=ASSETS/'CarVision-2.12-Training-Crops.zip'
 paths=['cache/owner_domain_v29/data.pt','cache/truth_v30/e70_crops.pt','cache/truth_v31/features.pt','cache/truth_v31/rows.json','cache/truth_v31/brands.json','cache/phone_v32/features.pt','cache/audited_v212/rows.json','cache/audited_v212/brands.json','models/owner_domain_v29/proposals.pt']
 if not training.exists():
  with zipfile.ZipFile(training,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
   for name in paths:
    p=ROOT/name
    if p.exists():z.write(p,name)
   for p in (ROOT/'datasets').glob('*/manifest.json'):z.write(p,p.relative_to(ROOT).as_posix())
 train=dict(name=training.name,bytes=training.stat().st_size,sha256=digest(training));assert train['bytes']<2*1024**3
 manifest=dict(repository=URL,release_tag='v2.12-test',version='2.12 TEST',original_zip=dict(name=original.name,bytes=original.stat().st_size,sha256=expected['sha256']),parts=parts,training=train,weights_sha256=expected.get('model_sha256','c1b01c33d55e163815734ba8c37810da97ac02bba0f9d042519635815d9fac14'),exe_sha256=expected['exe_sha256'],scope='Exact delivered portable release; supplemental frozen training crops/features, no recognition weight changes')
 write(PUB/'release_assets.json',json.dumps(manifest,indent=2));copy(PUB/'release_assets.json',ASSETS/'release_assets.json')
 sums=''.join(r['sha256']+'  '+r['name']+'\n' for r in parts+[train]);write(ASSETS/'SHA256SUMS.txt',sums)
 print('PUBLICATION_PREPARED',json.dumps(dict(repository=str(PUB),assets=manifest)),flush=True)
if __name__=='__main__':main()
