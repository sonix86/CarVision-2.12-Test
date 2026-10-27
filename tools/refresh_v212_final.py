"""Update the unreleased local audit build after the final training preflight fix."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.stage_v211 import sha,replace,write
def main():
    app=ROOT/'outputs/CarVision-2.12-Test/CarVision';build=ROOT/'outputs/build-v212-final/dist/CarVision';assert (build/'CarVision.exe').exists();assert not (app.parent/'SEALED').exists();changes=[]
    for p in build.rglob('*'):
        if p.is_file():
            target=app/p.relative_to(build)
            if not target.exists() or sha(target)!=sha(p):replace(p,target);changes.append(p.relative_to(build).as_posix())
    for folder in ('carvision','tools','tests'):
        for p in (ROOT/folder).glob('*.py'):replace(p,app/folder/p.name)
    for p in (ROOT/'outputs/pipeline-audit-v212').glob('*'):
        if p.is_file():replace(p,app/'Development/FIX_2_12'/p.name)
    replace(ROOT/'outputs/pipeline-audit-v212/AUDIT_RU.txt',app/'AUDIT_RU.txt');replace(ROOT/'outputs/tests-audit-v212-final.log',app/'Development/FIX_2_12/tests-audit-v212-final.log')
    guide=(app/'GUIDE_2_12_RU.txt').read_text(encoding='utf-8-sig').replace('149','150');guide+='\nДополнительно: train --output требует отдельную папку, если в указанной уже есть dino_make/semantic_make/make_expert/vehicle_model. Это предотвращает ситуацию, когда старые головы перекрывают новый результат обучения.\n'
    for name in ('GUIDE_2_12_RU.txt','GUIDE_FULL_2_6_RU.txt'):write(app/name,guide)
    verification=ROOT/'outputs/exe-v212-verification/verification.json'
    if verification.exists():replace(verification,verification.with_name('first-pass-before-training-guard.json'))
    report=json.loads((ROOT/'outputs/stage-v212.json').read_text(encoding='utf-8'));report.update(exe_sha256=sha(app/'CarVision.exe'),final_training_preflight_included=True,changed_build_files=changes);write(ROOT/'outputs/stage-v212.json',json.dumps(report,indent=2));write(app/'BUILD_2_12.json',json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
