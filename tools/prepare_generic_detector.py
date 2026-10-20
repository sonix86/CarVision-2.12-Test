"""General COCO boxes only, explicitly authorized by owner; no brand classifier weights."""
import os,sys,torch,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT
os.environ['TORCH_HOME']=str(ROOT/'cache')
from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
p=argparse.ArgumentParser();p.add_argument('--accurate',action='store_true');args=p.parse_args()
if args.accurate:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
    model=fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval();architecture='fasterrcnn_resnet50_fpn_v2_coco_v1'
else:model=ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1).eval();architecture='ssdlite320_mobilenet_v3_large_coco_v1'
checkpoint=dict(state=model.state_dict(),architecture=architecture,origin='generic-coco-detector',source='torchvision official COCO_V1; general object boxes only; car/truck classes 3/8',authorization='Owner explicitly allowed general ImageNet/COCO weights on 2026-10-20. Brand head must be independently trained.',brand_classifier=False)
torch.save(checkpoint,ROOT/'models/detector.tmp');(ROOT/'models/detector.tmp').replace(ROOT/'models/detector.pt')
(ROOT/'models/generic_detector.json').write_text(json.dumps({k:v for k,v in checkpoint.items() if k!='state'},indent=2))
if not args.accurate:torch.save(dict(state=model.backbone.state_dict(),origin='generic-coco-backbone'),ROOT/'models/generic_detector_backbone.pt')
from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
features=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features
torch.save(dict(state=features.state_dict(),origin='generic-imagenet-backbone'),ROOT/'models/generic_brand_backbone.pt')
print('General detector ready. No car-brand weights used.')
