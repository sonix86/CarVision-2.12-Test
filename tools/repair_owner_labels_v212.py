"""Align stale raw label exports with already manually reviewed annotations; no model labels."""
import collections,hashlib,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.data import load_records,statistics
def write(p,s):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.v212-tmp');tmp.write_text(s,encoding='utf-8');os.replace(tmp,p)
def main():
    out=ROOT/'outputs/pipeline-audit-v212';annotations=json.loads((ROOT/'models/owner_domain_v29/annotations.json').read_text(encoding='utf-8'));approval=json.loads((ROOT/'models/owner_domain_v29/review_approval.json').read_text(encoding='utf-8'));oldpath=ROOT/'data/owner_oct25/labels.jsonl';old=[json.loads(s) for s in oldpath.read_text(encoding='utf-8').splitlines() if s.strip()];reviewed={r['original_sha256']:r for r in annotations};changes=[]
    backup=out/'historical_owner_labels_before.jsonl'
    if not backup.exists():shutil.copy2(oldpath,backup)
    historical={r['image']:r for r in [json.loads(s) for s in backup.read_text(encoding='utf-8').splitlines() if s.strip()]}
    for r in old:
        if Path(r.get('source','')).name not in ('photo_21_2026-10-25_20-44-40.jpg','photo_22_2026-10-25_20-44-40.jpg'):continue
        approved=reviewed[r['original_sha256']]
        before=historical[r['image']]
        if (before['brand'],before['box'])!=(approved['brand'],approved['box']):
            changes.append(dict(image=r['image'],source=r['source'],before=dict(brand=before['brand'],box=before['box']),after=dict(brand=approved['brand'],box=approved['box']),basis='Existing v29 manual full-original review; not a classifier guess'))
            r['previous_export_label']=dict(brand=before['brand'],box=before['box']);r['brand']=approved['brand'];r['box']=approved['box'];r['label_source']='Manual full-original review v29, stale raw export aligned in audit v2.12';r['review']=approved['review']
    write(oldpath,'\n'.join(json.dumps(r,ensure_ascii=False) for r in old)+'\n')
    # Separate authoritative TRAIN-only export. Historical files are not inference tables.
    labels=[dict(r) for i,r in enumerate(annotations) if str(i) not in approval['exclusions']]
    for i in (15,16):
        r=dict(annotations[i]);r.update(brand='Dongfeng',model='Aeolus E70',colour='white',label_source='Visible emblem/body and independent author-labelled E70 source comparison');labels.append(r)
    for r in json.loads((ROOT/'models/phone_v32/truth.json').read_text(encoding='utf-8')):
        labels.append(dict(image='',source=r['image'],original_sha256=r['original_sha256'],box=r['box'],brand=r['brand'],model=r['family'] or 'unknown',colour=r['colour'],group=r['group'],label_source=r['label_source']))
    aiqar='C:/Users/Miku/Downloads/photo_2026-10-26_11-51-34.jpg';saved=max((ROOT/'outputs/CarVision-2.11-Test/CarVision/outputs').glob('photo_2026-10-26_11-51-34*.json'),key=lambda p:p.stat().st_mtime,default=None)
    if saved is None:saved=max((ROOT/'outputs/CarVision-2.10-Test/CarVision/outputs').glob('photo_2026-10-26_11-51-34*.json'),key=lambda p:p.stat().st_mtime)
    result=json.loads(saved.read_text(encoding='utf-8'));pending=[]
    if Path(aiqar).exists():labels.append(dict(image='',source=aiqar,original_sha256=hashlib.sha256(Path(aiqar).read_bytes()).hexdigest(),box=result['cars'][0]['box'],brand='AIQAR',model='unknown',colour='black',group='owner-night-oct26',label_source='Literal AIQAR badge read from original; exact model not certified. New brand has no trained head.'))
    else:pending.append(dict(source=aiqar,brand='AIQAR',reason='Original photograph unavailable; not exported for training'))
    target=ROOT/'data/reviewed_owner_v212';target.mkdir(exist_ok=True);export=[]
    renamed=list((ROOT/'outputs/exe-v211-verification').glob('*renamed.jpg'));recovered={hashlib.sha256(p.read_bytes()).hexdigest():p for p in renamed}
    for r in labels:
        src=Path(r['source'])
        if not src.exists():src=recovered.get(r['original_sha256'])
        if src is None or not src.exists():pending.append(dict(source=r['source'],brand=r['brand'],reason='Original unavailable, no verified identical copy'));continue
        digest=hashlib.sha256(src.read_bytes()).hexdigest()
        if digest!=r['original_sha256']:pending.append(dict(source=r['source'],brand=r['brand'],reason='Original hash changed; manual review required'));continue
        name='images/'+digest+'.jpg';dest=target/name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copy2(src,dest)
        row=dict(r,image=name,split='train',exclude_from_independent_evaluation=True,role='Reviewed known examples only. TRAIN data, never a blind test.');export.append(row)
    path=target/'labels.jsonl';write(path,'\n'.join(json.dumps(r,ensure_ascii=False) for r in export)+'\n');stats=statistics(load_records(path));report=dict(stale_raw_export_corrections=changes,reviewed_train_export=str(path),statistics=stats,pending_missing_originals=pending,all_known_examples_train_only=True,model_weights_changed=False,source_of_truth='Existing manual v29 review + E70 independent review + user and original visible badges. Never model predictions.')
    write(out/'label_repairs.json',json.dumps(report,ensure_ascii=False,indent=2));write(target/'README_RU.txt','Проверенные известные примеры для обучения. Не независимый тест. Экспортированы только доступные оригиналы с совпавшей контрольной суммой; недостающие перечислены в отчёте label_repairs.json. Не используйте этот набор для заявления точности на незнакомых фотографиях. Сам файл разметки не меняет веса приложения.\n');print(json.dumps(report,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
