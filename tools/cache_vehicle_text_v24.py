"""General local OCR on training/validation car masks; no filenames or badge lookup."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.vehicle_text import VehicleTextReader,text_features

torch.set_num_threads(2);out=ROOT/'models/multimodal_v24';reader=VehicleTextReader(out/'ocr')
path=ROOT/'cache/vehicle_text_v24.pt';cache=torch.load(path,weights_only=True) if path.exists() else dict(items={})
dataset=ROOT/'cache/multimodal_v24_data.pt';data=torch.load(dataset,weights_only=True)
parser=argparse.ArgumentParser();parser.add_argument('--include-test',action='store_true');args=parser.parse_args()
if args.include_test:assert (out/'training.json').exists(),'Select and freeze weights before test feature extraction'
rows=[r for r in data['rows'] if args.include_test or r['split'] in ('train','validation')]
started=time.time()
for i,r in enumerate(rows):
    key=hashlib.sha256(r['body_rgb'].numpy().tobytes()+r['body_mask'].numpy().tobytes()).hexdigest()
    if key not in cache['items']:
        records=reader.read(Image.fromarray(r['body_rgb'].numpy()),r['body_mask'].numpy())
        cache['items'][key]=dict(records=records,features=torch.from_numpy(text_features(records)))
    if i%100==0:
        torch.save(cache,path);status=dict(completed=i,total=len(rows),text_images=sum(bool(v['records']) for v in cache['items'].values()),seconds=time.time()-started)
        (out/'text_progress.json').write_text(json.dumps(status),encoding='utf-8');print(json.dumps(status),flush=True)
torch.save(cache,path);print('OCR FEATURES COMPLETE',flush=True)
