"""Fit simple HSV thresholds on a small AI-reviewed DEV set, never the final colour set."""
import itertools,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image,COLOURS
root=ROOT/'data/public_prototype';rows=json.loads((ROOT/'outputs/colour-development/references.pending.json').read_text())
labels=['black','blue','grey','grey','white','white','grey','brown','white','white','brown','grey','grey',None,'grey','grey',None,'black','white','grey','red',None,'green','black','grey','red']
for r,c in zip(rows,labels):r['reference_colour']=c;r['annotator']='OpenAI Codex visual development review, not independent human ground truth'
(root/'colour_development_review.json').write_text(json.dumps(dict(references=rows,purpose='Development only: thresholds selected here; not final evaluation'),indent=2))
data=[]
for top,bottom in itertools.product([.35,.45,.5],[.8,.9]):
    samples=[]
    for r in rows:
        if r['reference_colour'] is None:continue
        im=read_image(root/r['image']).crop(r['box']);w,h=im.size
        a=np.asarray(im.crop((w*.15,h*top,w*.85,h*bottom)).resize((96,64)).convert('HSV')).astype(float)
        samples.append((a[:,:,0]*360/255,a[:,:,1]/255,a[:,:,2]/255,COLOURS.index(r['reference_colour'])))
    data.append((top,bottom,samples))
best=None
for top,bottom,samples in data:
    for saturation,black,white in itertools.product([.18,.22,.26,.3,.34,.38],[.18,.22,.26,.30,.34],[.72,.76,.8,.84]):
        correct=[];per={}
        for hue,s,v,truth in samples:
            chrom=(s>=saturation)&(v>=black)
            masks=[(s<saturation)&(v>white),v<black,(s<saturation)&(v>=black)&(v<=white),((hue<18)|(hue>=345))&chrom,(hue>=185)&(hue<275)&chrom,(hue>=65)&(hue<185)&chrom,(hue>=18)&(hue<65)&chrom&(v<.68),(hue>=18)&(hue<65)&chrom&(v>=.68)]
            prediction=int(np.argmax([m.sum() for m in masks]));hit=prediction==truth;correct.append(hit);per.setdefault(truth,[]).append(hit)
        score=float(np.mean([np.mean(v) for v in per.values()]))+.05*float(np.mean(correct))
        preference=abs(saturation-.22)+abs(black-.22)+abs(white-.76)+abs(top-.4)+abs(bottom-.85)
        candidate=dict(score=score,accuracy=float(np.mean(correct)),macro_accuracy=float(np.mean([np.mean(v) for v in per.values()])),images=len(correct),rules=dict(saturation=saturation,black=black,white=white,top=top,bottom=bottom),preference=preference)
        if best is None or (score,-preference)>(best['score'],-best['preference']):best=candidate
best['scope']='Threshold selection on AI-reviewed development crops, not final accuracy. No test images used.'
(ROOT/'models/colour_rules.json').write_text(json.dumps(best,indent=2));print(json.dumps(best,indent=2))
