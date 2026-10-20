"""Exercise training/checkpoint/evaluation in an isolated synthetic fixture directory."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from carvision.training import train

root=Path(__file__).resolve().parents[1]/'outputs/smoke-data';root.mkdir(parents=True,exist_ok=True)
rows=[]
for i in range(20):
    Image.new('RGB',(96,96),(20+i*12,50+i*6,150-i*5)).save(root/f'{i}.png')
    rows.append(dict(image=f'{i}.png',box=[2,2,94,94],brand='BYD' if i%2 else 'Kia',colour='blue',source='synthetic engineering check ONLY',group=f'smoke-{i//2}'))
(root/'labels.jsonl').write_text('\n'.join(json.dumps(r) for r in rows),encoding='utf-8')
train(argparse.Namespace(labels=str(root/'labels.jsonl'),output=str(root.parent/'smoke-models'),seed=3210,epochs=1,batch=8,lr=.0003,scratch=True))
print('Synthetic training/checkpoint/evaluation completed. This is NOT car recognition accuracy.')
