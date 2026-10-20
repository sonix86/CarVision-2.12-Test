"""Mine false car windows on TRAINING images only; never mine the city demos/test set."""
import json,random,sys,time
from pathlib import Path
import cv2,numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image
from carvision.hog_detector import descriptor

torch.set_num_threads(2);cv2.setNumThreads(1);random.seed(3210);torch.manual_seed(3210)
root=ROOT/'data/public_prototype';rows=[json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines() if json.loads(s)['official_split']=='train']
random.shuffle(rows);train=rows[:2200];checkpoint=torch.load(ROOT/'models/detector.pt',weights_only=True);started=time.perf_counter()
def overlap(rect,truth):
    x,y,w,h=rect;a,b,c,d=truth;intersection=max(0,min(x+w,c)-max(x,a))*max(0,min(y+h,d)-max(y,b))
    return intersection/max(1,w*h)
for model in checkpoint['models']:
    window=tuple(model['window']);hog=descriptor(window);xs=[];ys=[]
    for row in train:
        im=cv2.cvtColor(np.asarray(read_image(root/row['image'])),cv2.COLOR_RGB2BGR);h,w=im.shape[:2];a,b,c,d=map(round,row['box']);positive=im[b:d,a:c]
        if positive.size:
            positive=cv2.resize(positive,window)
            for patch in [positive,cv2.flip(positive,1)]:xs.append(hog.compute(patch).ravel());ys.append(1.)
        for _ in range(10):
            ww=random.randint(window[0],max(window[0],min(w,round(w*.7))));hh=round(ww*window[1]/window[0])
            if hh>=h or ww>=w:continue
            x=random.randrange(w-ww+1);y=random.randrange(h-hh+1)
            if overlap((x,y,ww,hh),row['box'])>.03:continue
            xs.append(hog.compute(cv2.resize(im[y:y+hh,x:x+ww],window)).ravel());ys.append(0.)
    for round_index in range(2):
        hog.setSVMDetector(model['weights'].numpy().astype(np.float32));mined=0
        for row in train[:400]:
            im=cv2.cvtColor(np.asarray(read_image(root/row['image'])),cv2.COLOR_RGB2BGR)
            rects,scores=hog.detectMultiScale(im,0.,(8,8),(0,0),1.18,0,False)
            if not len(rects):continue
            order=np.argsort(np.asarray(scores).ravel())[::-1];accepted=[]
            for k in order:
                x,y,w,h=map(int,rects[k])
                if overlap((x,y,w,h),row['box'])>.03:continue
                if any(abs(x-q[0])+abs(y-q[1])<20 and abs(w-q[2])<20 for q in accepted):continue
                patch=im[y:y+h,x:x+w]
                if patch.size==0:continue
                xs.append(hog.compute(cv2.resize(patch,window)).ravel());ys.append(0.);mined+=1;accepted.append((x,y,w,h))
                if len(accepted)>=12:break
        x=torch.from_numpy(np.stack(xs));y=torch.tensor(ys).unsqueeze(1);linear=torch.nn.Linear(x.shape[1],1)
        with torch.no_grad():linear.weight.copy_(model['weights'][:-1].reshape(1,-1));linear.bias.copy_(model['weights'][-1:])
        optimizer=torch.optim.LBFGS(linear.parameters(),lr=.8,max_iter=80,history_size=12,line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad();logits=linear(x);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)+.0002*linear.weight.square().sum();loss.backward();return loss
        optimizer.step(closure);model['weights']=torch.cat([linear.weight.detach().ravel(),linear.bias.detach()])
        print(json.dumps(dict(window=window,round=round_index+1,mined=mined,patches=len(y),seconds=round(time.perf_counter()-started,1))),flush=True)
checkpoint['hard_negative_mining']='2 rounds, first 400 train images; random aspect-preserving background windows on 2200 train images'
checkpoint['raw_threshold']=2.
torch.save(checkpoint,ROOT/'models/detector-refined.pt')
print('Refined checkpoint saved separately for validation.',flush=True)
