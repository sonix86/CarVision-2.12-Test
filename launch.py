import argparse
import json
import multiprocessing
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Tashkent CarVision — offline multi-car classifier')
    sub=p.add_subparsers(dest='command')
    sub.add_parser('gui')
    sub.add_parser('setup',help='Download permitted generic weights once; never called by inference')
    pred=sub.add_parser('predict'); pred.add_argument('image'); pred.add_argument('--output',default='outputs'); pred.add_argument('--models'); pred.add_argument('--detection',type=float,default=.6); pred.add_argument('--threshold',type=float,default=.65)
    tr=sub.add_parser('train'); tr.add_argument('--labels',required=True); tr.add_argument('--output',default='models'); tr.add_argument('--epochs',type=int,default=25); tr.add_argument('--batch',type=int,default=16); tr.add_argument('--lr',type=float,default=.0003); tr.add_argument('--seed',type=int,default=3210); tr.add_argument('--scratch',action='store_true')
    ev=sub.add_parser('evaluate'); ev.add_argument('--labels',required=True); ev.add_argument('--models',default='models'); ev.add_argument('--output',default='outputs/evaluation')
    audit=sub.add_parser('audit'); audit.add_argument('--labels',required=True)
    report=sub.add_parser('report'); report.add_argument('--team',required=True); report.add_argument('--repo',required=True); report.add_argument('--contributions',required=True); report.add_argument('--analysis',required=True,help='Text file with actual error analysis, Chinese-brand discussion, phone tests and reflection'); report.add_argument('--output',default='outputs/report.pdf')
    args=p.parse_args()
    from carvision.core import ROOT
    if args.command in [None,'gui']:
        from carvision.gui import run
        run()
    elif args.command=='setup':
        import os
        os.environ['TORCH_HOME']=str(ROOT/'cache')
        import torch
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights
        from torchvision.models import mobilenet_v3_small,MobileNet_V3_Small_Weights
        (ROOT/'models').mkdir(exist_ok=True)
        model=ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
        torch.save(model.state_dict(),ROOT/'models/detector.pth')
        mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        print('Generic detector and ImageNet initialization ready. Brand classifier still requires your training data.')
    elif args.command=='predict':
        from carvision.core import Pipeline,save_prediction
        result=Pipeline(args.models,args.detection,args.threshold).predict(args.image)
        save_prediction(args.image,result,args.output); print(json.dumps(result,indent=2))
    elif args.command=='train':
        import os
        os.environ['TORCH_HOME']=str(ROOT/'cache')
        if args.epochs<1 or args.batch<1: p.error('epochs and batch must be positive')
        from carvision.training import train
        train(args)
    elif args.command=='evaluate':
        from carvision.training import evaluate
        evaluate(args.labels,args.models,args.output)
    elif args.command=='audit':
        from carvision.data import load_records,statistics,split_records
        records=load_records(args.labels); tr,va=split_records(records)
        print(json.dumps(dict(all=statistics(records),train=statistics(tr),validation=statistics(va)),indent=2))
    elif args.command=='report':
        from carvision.reporting import report
        report(args)

if __name__=='__main__':
    multiprocessing.freeze_support()
    main()
