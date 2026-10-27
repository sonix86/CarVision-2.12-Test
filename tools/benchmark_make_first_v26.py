"""Alternating warm classifier benchmark against immutable 2.5.3 source."""
import importlib.util,json,sys,time,statistics
from pathlib import Path
import torch
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.core import Pipeline
from carvision.scene import classify
from carvision.segmentation import decode_mask
torch.set_num_threads(3);spec=importlib.util.spec_from_file_location('carvision.scene_v253',ROOT/'outputs/CarVision-Update-2.5.3/payload/carvision/scene.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);p=Pipeline(ROOT/'outputs/Проверка обновления 2.5.3/CarVision/models',mode='fast');records=[]
for number in (2,5,16):
    d=json.loads((ROOT/f'outputs/fallback253-recipient-verification/owner-diagnostic-{number}.stdout.txt').read_text(encoding='utf-8-sig'));r=max(d['cars'],key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]));im=ImageOps.exif_transpose(Image.open(d['image'])).convert('RGB').crop(r['box']);mask=decode_mask(r['segmentation']);times={'old':[],'new':[]};outputs={}
    for run in range(3):
        for name,fn in ([('old',module.classify),('new',classify)] if run%2==0 else [('new',classify),('old',module.classify)]):
            start=time.perf_counter();outputs[name]=fn(p,im,mask);elapsed=time.perf_counter()-start
            if run:times[name].append(elapsed)
    for key in ('brand','brand_confidence','make_evidence','colour','colour_confidence','model','model_alternatives'):assert outputs['old'].get(key)==outputs['new'].get(key),(number,key)
    records.append(dict(photo=number,old_seconds=statistics.mean(times['old']),new_seconds=statistics.mean(times['new']),decision_preserved=True))
report=dict(records=records,scope='Warm classification of three known owner-error crops; no detector/startup time; concurrent preparation may affect timings. Not a quality test.')
(ROOT/'models/ev_views_v26/speed_classification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
