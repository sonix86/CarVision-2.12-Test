"""Inspect suspected training examples without their own retrieval-bank votes."""
import json,sys
from pathlib import Path
import torch
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from carvision.semantic_make import CosineMakeHead
from carvision.segmentation import isolate
from tools.audit_ev_baseline_v212 import TARGET_MAKES
def main():
    torch.set_num_threads(3);out=ROOT/'outputs/pipeline-audit-v212';rows=json.loads((ROOT/'cache/truth_v31/rows.json').read_text(encoding='utf-8'));ck=torch.load(ROOT/'models/phone_v32/scoped_candidate.pt',weights_only=True);features=torch.load(ROOT/'cache/truth_v31/features.pt',weights_only=True)['x'][:len(rows)];brands=ck['brands'];h=CosineMakeHead(torch.zeros(len(brands),1280)).eval();h.load_state_dict(ck['head'])
    with torch.no_grad():p=(h(features)/ck['temperature']).softmax(1)
    truth=torch.tensor([brands.index(r['brand']) for r in rows]);wrong=p.argmax(1)!=truth;score=p.max(1).values-p.gather(1,truth[:,None])[:,0];order=score.argsort(descending=True).tolist();chosen=[]
    for ev in (True,False):chosen.extend([i for i in order if rows[i]['split']=='train' and bool(wrong[i]) and (rows[i]['brand'] in TARGET_MAKES)==ev][:12])
    data=torch.load(ROOT/'cache/owner_domain_v29/data.pt',weights_only=True)['rows'];data+=torch.load(ROOT/'cache/truth_v30/e70_crops.pt',weights_only=True)['rows'];data+=torch.load(ROOT/'models/owner_domain_v29/proposals.pt',weights_only=True)['rows'][15:16];lookup={r['image']:r for r in data};flags=[]
    for page in range((len(chosen)+11)//12):
        sheet=Image.new('RGB',(1200,900),'white');draw=ImageDraw.Draw(sheet)
        for slot,i in enumerate(chosen[page*12:(page+1)*12]):
            r=rows[i];body=lookup[r['image']];x=slot%4*300;y=slot//4*300;im=ImageOps.contain(isolate(Image.fromarray(body['body_rgb'].numpy()),body['body_mask'].numpy()),(294,245));sheet.paste(im,(x+(300-im.width)//2,y+42));candidate=brands[int(p[i].argmax())];draw.text((x+3,y+3),f"#{i} LABEL: {r['brand']}",fill='black');draw.text((x+3,y+19),f"MODEL GUESS: {candidate} {float(p[i].max()):.0%}",fill='red');flags.append(dict(index=i,image=r['image'],brand=r['brand'],family=r.get('family'),source=r.get('source'),candidate=candidate,confidence=float(p[i].max()),role='TRAIN-only manual review flag. No own retrieval vote used; no automatic label change.'))
        sheet.save(out/f'neural_train_{page}.jpg',quality=95)
    (out/'neural_training_flags.json').write_text(json.dumps(flags,ensure_ascii=False,indent=2),encoding='utf-8');print('TRAIN FLAGS',len(flags),flush=True)
if __name__=='__main__':main()
