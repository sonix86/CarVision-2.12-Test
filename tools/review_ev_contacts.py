"""Contact sheets of real source photos for annotation review, not model input."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw,ImageFont,ImageOps
from carvision.core import ROOT,read_image
src=ROOT/'datasets/ev_commons_v19';rows=[]
for f in [src/'manifest.json',src/'thumb_data/manifest.json']:
 if f.exists():rows+=json.loads(f.read_text(encoding='utf-8'))
rows.sort(key=lambda r:(r['brand'],r['model'],r['image']))
out=ROOT/'outputs/ev-contact-review';out.mkdir(exist_ok=True);font=ImageFont.truetype('segoeui.ttf',15)
index=[]
for start in range(0,len(rows),25):
 canvas=Image.new('RGB',(1500,1150),'#f4f7fb');draw=ImageDraw.Draw(canvas)
 for j,r in enumerate(rows[start:start+25]):
  x=j%5*300;y=j//5*230;im=ImageOps.contain(read_image(src/r['image']),(290,190));canvas.paste(im,(x+(300-im.width)//2,y+25));label=f"{start+j}: {r['brand']} {r['model']}" if r['brand'] else f'{start+j}: unlabelled street scene';draw.text((x+5,y+3),label,font=font,fill='#203249');draw.text((x+5,y+215),r['image'].split('/')[-1],font=font,fill='#203249');index.append(dict(index=start+j,**r))
 canvas.save(out/f'page-{start//25+1:02}.jpg',quality=92)
(out/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
print('Photos',len(rows),'sheets',(len(rows)+24)//25,flush=True)
