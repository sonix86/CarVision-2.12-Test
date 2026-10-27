# CarVision 2.12 TEST

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
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Download release files using the browser, or authenticated GitHub CLI:
gh release download v2.12-test --repo sonix86/CarVision-2.12-Test --dir .release-downloads
.venv\Scripts\python.exe tools/restore_release_v212.py --downloads .release-downloads --training
.venv\Scripts\python.exe launch.py gui
.venv\Scripts\python.exe launch.py predict "C:\photos\car.jpg" --mode fast --output outputs\exam
```

The restorer reads the frozen release manifest, verifies hashes, and restores
`models`, `data`, `datasets` and `samples` from the exact portable archive. With
`--training`, it also restores the frozen feature/crop cache and source manifests.
It refuses path traversal and refuses to replace an existing differing file.
The repository and release are public; instructor access requires no invitation.

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
.venv\Scripts\python.exe -m unittest discover -s tests
.venv\Scripts\python.exe tools/audit_ev_baseline_v212.py
# Optional personal candidate training, separate from the advanced active head:
.venv\Scripts\python.exe launch.py train --labels data/reviewed_owner_v212/labels.jsonl --output models/my_new_candidate
```

Personal training writes a separate candidate; it does not automatically replace
the advanced head. Do not train into the shipped `models` directory. Building a
Windows EXE uses `build.ps1`; the frozen release remains the tested reference.

## Submission and provenance

Read `docs/ASSIGNMENT_COMPLIANCE.md`, `docs/CarVision-2.12-Report.pdf`, source
attribution manifests and `THIRD_PARTY.md`. The owner explicitly authorized public publication of the included phone photographs.
This new repository and its release are public. No unrelated existing GitHub repository was changed.

The original local Git history is retained. Its author metadata is not rewritten
or backdated. This publication was prepared on 28 October 2026, after the stated
27 October 23:59 deadline; it cannot retroactively meet that deadline. The new
`midterm` tag identifies this snapshot, not a claim of timely submission.

OpenAI Codex assisted with implementation, data preparation/review, experiments,
debugging, verification, packaging, documentation and publication. Student work
and AI assistance must be disclosed honestly. Originality relative to other teams,
student understanding and live-test performance require human
verification. Account passwords and authentication tokens are never project files.
