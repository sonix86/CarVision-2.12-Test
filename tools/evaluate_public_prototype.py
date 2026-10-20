"""Honest held-out measurements; test records never train or choose model weights."""
import argparse,collections,json,sys,time,random
from pathlib import Path
import numpy as np,torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,Pipeline,read_image,heuristic_colour

def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--detector-images',type=int,default=52);args=p.parse_args()
    root=ROOT/'data/public_prototype';rows=[json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines()];split=json.loads((ROOT/'models/public_split.json').read_text());selected=set(split['validation' if args.split=='validation' else 'official_test']);rows=[r for r in rows if r['image'] in selected]
    pipeline=Pipeline(mode='fast');matrix=np.zeros((len(pipeline.brands),len(pipeline.brands)),dtype=int);preds=[];correct=[];conf=[];started=time.perf_counter()
    for start in range(0,len(rows),32):
        batch=rows[start:start+32];crops=[]
        for r in batch:
            im=read_image(root/r['image']);x,y,x2,y2=r['box'];px=(x2-x)*.04;py=(y2-y)*.04
            crops.append(pipeline.preprocess(im.crop((max(0,x-px),max(0,y-py),min(im.width,x2+px),min(im.height,y2+py)))))
        with torch.inference_mode():
            x=torch.stack(crops);b,_=pipeline.classifier(x);prob=(b/getattr(pipeline,'brand_temperature',1.)).softmax(1)
            if getattr(pipeline,'multiview_brand',False):
                b,_=pipeline.classifier(x.flip(3));prob=(prob+(b/getattr(pipeline,'brand_temperature',1.)).softmax(1))/2
            values,indices=prob.max(1)
        for r,value,index in zip(batch,values.tolist(),indices.tolist()):
            truth=pipeline.brands.index(r['brand']);matrix[truth,index]+=1;correct.append(truth==index);conf.append(value);preds.append(dict(image=r['image'],reference=r['brand'],prediction=pipeline.brands[index],confidence=value))
    result=dict(split=args.split,images=len(rows),brand_top1_accuracy=float(np.mean(correct)),brand_scope=f"{len(set(r['brand'] for r in rows))} brands represented in this Stanford benchmark; human annotated crops, not full-scene/unknown-brand accuracy",brands=pipeline.brands,confusion_matrix=matrix.tolist(),per_brand={b:dict(count=int(matrix[i].sum()),accuracy=float(matrix[i,i]/max(1,matrix[i].sum()))) for i,b in enumerate(pipeline.brands)},thresholds=[],seconds=round(time.perf_counter()-started,2))
    for threshold in [.2,.3,.4,.5,.6,.65,.7,.8,.9]:
        accepted=np.asarray(conf)>=threshold;n=int(accepted.sum());result['thresholds'].append(dict(threshold=threshold,accepted=n,coverage=n/len(rows),accuracy_when_accepted=float(np.asarray(correct)[accepted].mean()) if n else None))
    refs=root/'colour_reference_review.json'
    if refs.exists() and args.split=='test':
        references=json.loads(refs.read_text())['references'];scored=[]
        for r in references:
            if r['reference_colour'] is None:continue
            crop=read_image(root/r['image']).crop(r['box'])
            predicted,confidence=pipeline.colour_classifier.predict(crop) if pipeline.colour_classifier is not None else heuristic_colour(crop)
            scored.append(dict(id=r['id'],reference=r['reference_colour'],prediction=predicted,correct=predicted==r['reference_colour']))
        result['colour_visual_review']=dict(images=len(scored),accuracy=sum(r['correct'] for r in scored)/max(1,len(scored)),annotator='OpenAI Codex visual review, not independent human ground truth; ambiguous examples excluded',details=scored)
    if args.split=='test' and args.detector_images>0:
        from torchvision.ops import box_iou
        selected=[]
        for brand in sorted({r['brand'] for r in rows}):
            pool=[r for r in rows if r['brand']==brand];random.Random(3211).shuffle(pool);selected.extend(pool[:max(1,args.detector_images//len(pipeline.brands))])
        measured=[];scene_started=time.perf_counter()
        for i,r in enumerate(selected):
            prediction=pipeline.predict(root/r['image']);cars=prediction['cars'];boxes=torch.tensor([c['box'] for c in cars],dtype=torch.float32).reshape(-1,4);truth=torch.tensor([r['box']],dtype=torch.float32)
            overlaps=box_iou(boxes,truth).reshape(-1) if len(boxes) else torch.empty(0)
            best=int(overlaps.argmax()) if len(overlaps) else -1;hit=best>=0 and float(overlaps[best])>=.5
            measured.append(dict(image=r['image'],reference=r['brand'],detected=hit,predicted_brand=cars[best]['brand'] if hit else None,correct=hit and cars[best]['brand']==r['brand'],seconds=prediction['seconds'],proposals=len(cars)))
            print('Full-photo check',i+1,'/',len(selected),flush=True)
        result['dominant_car_full_photo_check']=dict(images=len(measured),detected=sum(r['detected'] for r in measured),detection_recall=sum(r['detected'] for r in measured)/len(measured),correct_brand=sum(r['correct'] for r in measured),end_to_end_known_brand_accuracy=sum(r['correct'] for r in measured)/len(measured),threshold=pipeline.brand_threshold,seconds=round(time.perf_counter()-scene_started,2),scope='Fixed stratified official-test photos; dominant labelled car only, IoU>=0.5, fast whole-frame mode. Other cars are unannotated: full-scene precision cannot be inferred. Not a Tashkent phone test.',details=measured)
    output=ROOT/'models/evaluation';output.mkdir(exist_ok=True);(output/f'public_{args.split}.json').write_text(json.dumps(result,indent=2));(output/f'public_{args.split}_predictions.json').write_text(json.dumps(preds,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    if args.split=='test':
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(11,9));ax.imshow(matrix,cmap='Blues');ax.set_xticks(range(len(pipeline.brands)),pipeline.brands,rotation=65,ha='right');ax.set_yticks(range(len(pipeline.brands)),pipeline.brands);ax.set(xlabel='Predicted brand',ylabel='Reference brand',title=f'Public held-out crops: {len(rows)} images, top-1 {result["brand_top1_accuracy"]:.1%}')
        for i in range(len(matrix)):
            for j in range(len(matrix)):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=7,color='white' if matrix[i,j]>matrix.max()*.5 else 'black')
        fig.tight_layout();fig.savefig(output/'public_brand_confusion.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
