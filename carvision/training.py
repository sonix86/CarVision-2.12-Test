import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .core import Classifier, COLOURS, PREPROCESS, read_image
from .data import load_records, split_records, statistics

class Cars(Dataset):
    def __init__(self, rows, brands, augment=False):
        self.rows,self.brands=rows,brands
        self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomAffine(8,translate=(.04,.04),scale=(.9,1.1)),transforms.ColorJitter(brightness=.15,contrast=.15),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]) if augment else PREPROCESS
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return self.transform(read_image(r['path']).crop(r['box'])),self.brands.index(r['brand']),COLOURS.index(r['colour'])

def train(args):
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rows=load_records(args.labels); tr,va=split_records(rows,args.seed)
    brands=sorted({r['brand'] for r in tr})
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    model=Classifier(len(brands))
    device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)
    train_loader=DataLoader(Cars(tr,brands,True),batch_size=args.batch,shuffle=True,num_workers=0)
    val_loader=DataLoader(Cars(va,brands),batch_size=args.batch,num_workers=0)
    counts=np.array([sum(r['brand']==b for r in tr) for b in brands]); weights=torch.tensor(1/np.sqrt(counts),dtype=torch.float32,device=device)
    loss_b=nn.CrossEntropyLoss(weight=weights/weights.mean(),label_smoothing=.05)
    loss_c=nn.CrossEntropyLoss(label_smoothing=.05)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    history=[]; best=-1; started=time.perf_counter()
    manifest={r['image']:('train' if r in tr else 'validation') for r in rows}
    (out/'split.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    from .data import dataset_fingerprint
    (out/'dataset.sha256').write_text(dataset_fingerprint(rows),encoding='ascii')
    for epoch in range(args.epochs):
        model.train(); total=0.
        for x,b,c in train_loader:
            x,b,c=x.to(device),b.to(device),c.to(device)
            # Keep running statistics stable even for a final one-item batch.
            if len(x)==1:
                for m in model.modules():
                    if isinstance(m,nn.BatchNorm2d): m.eval()
            optimizer.zero_grad(); pb,pc=model(x); loss=loss_b(pb,b)+.5*loss_c(pc,c)
            loss.backward(); optimizer.step(); total+=float(loss.detach())*len(x)
        scheduler.step(); model.eval(); bc=cc=0
        with torch.inference_mode():
            for x,b,c in val_loader:
                pb,pc=model(x.to(device)); bc+=int((pb.argmax(1).cpu()==b).sum()); cc+=int((pc.argmax(1).cpu()==c).sum())
        row=dict(epoch=epoch+1,loss=total/len(tr),brand_accuracy=bc/len(va),colour_accuracy=cc/len(va)); history.append(row)
        print(json.dumps(row),flush=True)
        score=.6*row['brand_accuracy']+.4*row['colour_accuracy']
        if score>best:
            best=score
            torch.save(dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},brands=brands,architecture='mobilenet_v3_small_avgmax_dualhead_v1',seed=args.seed,origin='team-trained-from-scratch'),out/'classifier.tmp')
            (out/'classifier.tmp').replace(out/'classifier.pt')
    metadata=dict(history=history,data=statistics(rows),train=statistics(tr),validation=statistics(va),parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started,hardware=device,torch_version=str(torch.__version__),epochs=args.epochs,lr=args.lr,batch=args.batch,seed=args.seed,loss='weighted CE brand + 0.5 CE colour; label smoothing 0.05',optimizer='AdamW, weight_decay=0.01, cosine LR',augmentation='resize224, horizontal flip, affine8deg/4percent/scale0.9-1.1, brightness/contrast0.15',pretrained=False)
    (out/'training.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    evaluate(args.labels,out,out/'evaluation',args.seed)
    from .detector_training import train_detector
    train_detector(tr,va,out,getattr(args,'detector_epochs',25),args.seed,args.lr)

def evaluate(labels,models_dir,output,seed=3210):
    rows=load_records(labels); directory=Path(models_dir)
    from .data import dataset_fingerprint
    fingerprint=directory/'dataset.sha256'
    if not fingerprint.exists() or fingerprint.read_text().strip()!=dataset_fingerprint(rows):
        raise ValueError('Dataset changed or fingerprint missing. Retrain before evaluating; labels/images must match the saved split.')
    manifest_path=directory/'split.json'
    if not manifest_path.exists(): raise ValueError('Missing saved split.json. Evaluate the original held-out split.')
    split=json.loads(manifest_path.read_text(encoding='utf-8')); va=[r for r in rows if split.get(r['image'])=='validation']
    if not va: raise ValueError('No held-out validation records.')
    ck=torch.load(directory/'classifier.pt',map_location='cpu',weights_only=True)
    brands=ck['brands']; model=Classifier(len(brands)); model.load_state_dict(ck['state']); model.eval()
    matrix=np.zeros((len(brands),len(brands)),dtype=int); colour_matrix=np.zeros((8,8),dtype=int); errors=[]
    with torch.inference_mode():
        for r in va:
            b,c=model(PREPROCESS(read_image(r['path']).crop(r['box'])).unsqueeze(0)); bi=int(b.argmax()); ci=int(c.argmax())
            matrix[brands.index(r['brand']),bi]+=1; colour_matrix[COLOURS.index(r['colour']),ci]+=1
            if brands[bi]!=r['brand'] or COLOURS[ci]!=r['colour']: errors.append(dict(image=r['image'],box=r['box'],expected=[r['brand'],r['colour']],predicted=[brands[bi],COLOURS[ci]]))
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    metrics=dict(validation_cars=len(va),brand_accuracy=float(matrix.trace()/len(va)),colour_accuracy=float(colour_matrix.trace()/len(va)),brands=brands,brand_confusion=matrix.tolist(),colour_confusion=colour_matrix.tolist(),errors=errors,scope='Ground-truth car crops; NOT end-to-end detection accuracy. Validation used for model selection, not an unseen final test.')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(max(7,len(brands)*.45),max(6,len(brands)*.4)))
    ax.imshow(matrix,cmap='Blues'); ax.set_xticks(range(len(brands)),brands,rotation=90); ax.set_yticks(range(len(brands)),brands); ax.set_xlabel('Predicted'); ax.set_ylabel('Reference'); fig.tight_layout(); fig.savefig(out/'brand_confusion.png',dpi=150); plt.close(fig)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['errors','brand_confusion','colour_confusion']}),flush=True)
    return metrics
