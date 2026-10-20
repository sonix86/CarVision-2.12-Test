"""Separate dev images for refining colour rules; final colour sheet remains held out."""
import json,random,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image
root=ROOT/'data/public_prototype';validation=set(json.loads((ROOT/'models/public_split.json').read_text())['validation']);rows=[json.loads(s) for s in (root/'labels.jsonl').read_text().splitlines() if json.loads(s)['image'] in validation]
chosen=[]
for brand in sorted({r['brand'] for r in rows}):
    pool=sorted([r for r in rows if r['brand']==brand],key=lambda r:r['image']);random.Random(532).shuffle(pool)
    for r in pool[:2]:chosen.append(dict(id=len(chosen)+1,image=r['image'],box=r['box'],brand=brand,reference_colour=None))
out=ROOT/'outputs/colour-development';out.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
for start in range(0,len(chosen),16):
    sheet=Image.new('RGB',(1280,960),'#eef2f6');draw=ImageDraw.Draw(sheet)
    for offset,r in enumerate(chosen[start:start+16]):
        x=offset%4*320;y=offset//4*240;im=read_image(root/r['image']).crop(r['box']);im.thumbnail((308,200));sheet.paste(im,(x+(320-im.width)//2,y+32+(200-im.height)//2));draw.text((x+8,y+7),f"{r['id']:02d} {r['brand']}",fill='#102b46',font=font)
    sheet.save(out/f'sheet-{start//16+1}.jpg',quality=93)
(out/'references.pending.json').write_text(json.dumps(chosen,indent=2))
