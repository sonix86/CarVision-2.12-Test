"""Create stable-ID contact sheets for remaining make-source visual review."""
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'models/make_scope_v23'
info=json.loads((OUT/'dataset.json').read_text(encoding='utf-8'))
reviewed=json.loads((OUT/'reviewed_images.json').read_text(encoding='utf-8'))
data=torch.load(ROOT/'cache/make_scope_v23.pt',weights_only=True)
lookup={r['image']:r for r in data['rows']}
pending=[r for r in info['new_rows'] if r['image'] not in reviewed and r['image'] in lookup]
(OUT/'pending_final_index.json').write_text(json.dumps(pending,ensure_ascii=False,indent=2),encoding='utf-8')
for first in range(0,len(pending),40):
    sheet=Image.new('RGB',(1000,1120),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(pending[first:first+40],first):
        im=Image.fromarray(lookup[r['image']]['body_rgb'].numpy());im.thumbnail((190,103))
        x=(i-first)%5*200;y=(i-first)//5*140
        sheet.paste(im,(x,y));draw.text((x,y+104),f"{i} {r['brand']}",fill='black')
        draw.text((x,y+119),r['family'],fill='black')
    sheet.save(OUT/f'pending_final_{first//40}.jpg')
print('Pending',len(pending),'sheets',(len(pending)+39)//40)
