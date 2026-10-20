import sys,torch,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.transforms.functional import to_tensor
from PIL import Image,ImageDraw
import numpy as np,cv2
torch.set_num_threads(3);ck=torch.load(ROOT/'models/segmenter.pt',weights_only=True);net=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=800,max_size=1333,box_score_thresh=.45,box_detections_per_img=40).eval();net.load_state_dict(ck['state']);out=ROOT/'outputs/instance-preview';out.mkdir(exist_ok=True)
with torch.inference_mode():
 for name in ['06','01','03','04','05']:
  im=read_image(ROOT/'samples/images'/f'{name}.jpg');im.thumbnail((1600,1600));start=time.perf_counter();d=net([to_tensor(im)])[0];keep=[i for i in range(len(d['scores'])) if int(d['labels'][i]) in [3,8]];rgb=np.asarray(im).copy();rows=[]
  for j,i in enumerate(keep):
   mask=d['masks'][i,0].numpy()>.5;colour=np.array([(30+j*83)%220,(180+j*47)%240,(160+j*71)%240]);rgb[mask]=(rgb[mask]*.65+colour*.35).astype('uint8');contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,contours,-1,tuple(int(x) for x in colour),3);box=d['boxes'][i].round().int().tolist();rows.append(dict(id=j+1,box=box,score=float(d['scores'][i]),mask_pixels=int(mask.sum())));cv2.putText(rgb,str(j+1),(box[0],box[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),4);cv2.putText(rgb,str(j+1),(box[0],box[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
  Image.fromarray(rgb).save(out/f'{name}.jpg');torch.save(dict(predictions={k:v[keep] for k,v in d.items()},size=im.size),out/f'{name}.pt');(out/f'{name}.json').write_text(json.dumps(rows,indent=2));print(name,time.perf_counter()-start,len(rows),flush=True)
