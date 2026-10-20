"""Download only general COCO instance masks; brand/model/colour decisions remain ours."""
import json,sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2,MaskRCNN_ResNet50_FPN_V2_Weights
torch.hub.set_dir(str(ROOT/'cache/hub'));torch.set_num_threads(3)
weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
model=maskrcnn_resnet50_fpn_v2(weights=weights).eval()
torch.save(dict(origin='generic-coco-instance-segmenter',architecture='maskrcnn_resnet50_fpn_v2',state=model.state_dict(),source=weights.url,brand_classifier=False,model_classifier=False,colour_classifier=False),ROOT/'models/segmenter.pt')
(ROOT/'models/segmenter.json').write_text(json.dumps(dict(origin='generic-coco-instance-segmenter',architecture='Mask R-CNN ResNet50 FPN v2',parameters=sum(p.numel() for p in model.parameters()),source=weights.url,scope='General COCO car/truck visible-object masks only, no brand/model/colour classifier',authorization='General ImageNet/COCO weights explicitly permitted by owner',documentation='https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html'),indent=2))
print('GENERAL SEGMENTER READY',flush=True)
