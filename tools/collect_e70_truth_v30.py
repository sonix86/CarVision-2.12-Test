"""Bounded licensed source collection for manual truth review, no prediction labels."""
import hashlib,html,json,re,time,sys,urllib.parse,urllib.request,urllib.error
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/e70_truth_v30';OUT.mkdir(exist_ok=True)
HEAD={'User-Agent':'CarVision-Education/0.30 (local student research; per-file Commons attribution retained)'}
def fetch(url):
    blocked=OUT/'deferred.json'
    if blocked.exists() and time.time()<json.loads(blocked.read_text())['not_before']:raise RuntimeError('Honour Retry-After; collection deferred')
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers=HEAD),timeout=35) as r:return r.read()
    except urllib.error.HTTPError as e:
        if e.code==429:
            delay=max(600,int(e.headers.get('Retry-After','600')));blocked.write_text(json.dumps(dict(not_before=time.time()+delay)));raise RuntimeError('HTTP429: entire collector stopped') from e
        raise
def api(**params):
    url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',format='json',**params));p=OUT/('api_'+hashlib.sha256(url.encode()).hexdigest()[:20]+'.json')
    if not p.exists():p.write_bytes(fetch(url))
    return json.loads(p.read_text())
def main():
    members=api(list='categorymembers',cmtitle='Category:Dongfeng Aeolus E70',cmlimit=50)['query']['categorymembers'];titles=sorted(r['title'] for r in members if r['ns']==6 and r['title'].lower().endswith(('.jpg','.jpeg')));assert len(titles)<=20
    pages=api(prop='imageinfo',titles='|'.join(titles),iiprop='url|size|extmetadata',iiurlwidth=960)['query']['pages'];rows=[]
    for page in sorted(pages.values(),key=lambda p:p['title']):
        info=page['imageinfo'][0];meta=info['extmetadata'];value=lambda k:html.unescape(re.sub('<[^>]*>','',meta.get(k,{}).get('value','')))
        license=value('LicenseShortName')
        if not license.startswith(('CC BY','CC0','Public domain')):continue
        if info['width']<400:continue
        url=info.get('thumburl',info['url']);name=hashlib.sha256(page['title'].encode()).hexdigest()[:20]+'.jpg';p=OUT/name
        if not p.exists():p.write_bytes(fetch(url))
        row=dict(image=name,title=page['title'],source='https://commons.wikimedia.org/wiki/'+urllib.parse.quote(page['title']),original_url=info['url'],download_url=url,author=value('Artist'),date=value('DateTimeOriginal'),license=license,license_url=value('LicenseUrl'),description=value('ImageDescription'),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),label_status='UNREVIEWED: category/title not sufficient alone',proposed_make='Dongfeng',proposed_family='Aeolus E70')
        rows.append(row);print('SOURCE',len(rows),row['title'],flush=True)
    (OUT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    from PIL import Image,ImageOps,ImageDraw
    sheet=Image.new('RGB',(1200,300*((len(rows)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows):
        with Image.open(OUT/r['image']) as im:im=ImageOps.contain(im.convert('RGB'),(295,255));x=i%4*300;y=i//4*300;sheet.paste(im,(x,y+25));draw.text((x+3,y+3),str(i)+': '+r['title'][:31],fill='black')
    sheet.save(OUT/'review.jpg',quality=95);print('COLLECTION COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
