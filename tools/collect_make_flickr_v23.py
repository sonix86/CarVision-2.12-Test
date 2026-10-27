"""Bounded complementary street-photo research; source-confirmed free Flickr licences."""
import hashlib,json,re,sys,time,urllib.parse,urllib.request,urllib.error
from pathlib import Path
from PIL import Image,ImageOps
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/make_flickr_v23';OUT.mkdir(exist_ok=True)
CACHE=OUT/'metadata';CACHE.mkdir(exist_ok=True);deadline=time.time()+3600;last={}
queries=['Weltmeister','WM Motor EX5','威马 EX5','BAIC EU5','Beijing EU5','Li Auto car','Lixiang car','Zeekr car',
         'BYD car night','BYD Seagull','GAC Aion car','Aion S car','Neta car','Nammi car','Leapmotor car','Wuling car',
         'Geometry car','Changan car','Ora electric car','Arcfox car','Pocco car','Letin car','Hanteng car']
patterns={'Weltmeister':r'Weltmeister|WM\s*Motor|威马','BAIC':r'BAIC|Beijing\s+EU[357]',
          'Li Auto':r'Li\s+Auto|Lixiang|理想','Zeekr':r'Zeekr|极氪','BYD':r'\bBYD\b|比亚迪',
          'GAC Aion':r'\bAion\b|广汽埃安','Neta':r'\bNeta\b|哪吒','Dongfeng':r'\bNammi\b',
          'Leapmotor':r'Leapmotor|零跑','Wuling':r'Wuling|五菱','Geometry':r'Geometry|几何',
          'Changan':r"Chang['’]?an|长安",'Ora':r'\bORA\b|欧拉','Arcfox':r'Arcfox|极狐',
          'Pocco':r'\bPocco\b','Letin':r'\bLetin\b|雷丁','Hanteng':r'\bHanteng\b|汉腾'}
def fetch(url):
    if time.time()>deadline:raise SystemExit('One-hour Flickr collection bound reached; no retries.')
    host=urllib.parse.urlparse(url).hostname;delay=8 if host=='api.openverse.org' else 2
    time.sleep(max(0,delay-(time.monotonic()-last.get(host,0))));last[host]=time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TashkentCarVision/2.3 educational licence-preserving photo research'}),timeout=30) as response:return response.read()
    except urllib.error.HTTPError as error:
        if error.code==429:raise SystemExit('Flickr/Openverse HTTP429; stopped without retries.')
        raise
found={}
for query in queries:
    name=hashlib.sha256(query.encode()).hexdigest()[:20];path=CACHE/('search_'+name+'.json')
    try:
        if not path.exists():path.write_bytes(fetch('https://api.openverse.org/v1/images/?'+urllib.parse.urlencode(dict(q=query,source='flickr',license='by,by-sa,cc0,pdm',page_size=20))))
        for r in json.loads(path.read_text(encoding='utf-8')).get('results',[]):
            if r.get('source')=='flickr' and r.get('license') in ['by','by-sa','cc0','pdm']:found[r['foreign_landing_url']]=r
        print('SEARCH',query,'unique',len(found),flush=True)
    except Exception as error:print('SEARCH_ERROR',query,str(error),flush=True)
manifest=OUT/'manifest.json';rows=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else [];seen={r['source'] for r in rows}
for p in (ROOT/'datasets').glob('*/manifest.json'):
    if p==manifest:continue
    try:seen.update(r.get('source','') for r in json.loads(p.read_text(encoding='utf-8')))
    except (ValueError,TypeError):pass
counts={}
for r in rows:counts[r['brand']]=counts.get(r['brand'],0)+1
def licence_key(url):
    value=urllib.parse.urlparse(url)
    return value.netloc.lower().removeprefix('www.')+value.path.rstrip('/')
for source,r in found.items():
    if len(rows)>=180:break
    title=r.get('title','');brands=[b for b,pattern in patterns.items() if re.search(pattern,title,re.I)]
    if len(brands)!=1 or source in seen or re.search(r'interior|engine|logo|badge|\btoy\b|scale model|render|illustration|diecast',title,re.I):continue
    brand=brands[0]
    if counts.get(brand,0)>=24:continue
    key=hashlib.sha256(source.encode()).hexdigest()[:20];page=CACHE/(key+'.html');photo=OUT/(key+'.jpg')
    try:
        if not page.exists():page.write_bytes(fetch(source))
        text=page.read_text(encoding='utf-8');license=re.search(r'"license"\s*:\s*"([^"]+)"',text)
        if not license or licence_key(license[1])!=licence_key(r['license_url']):print('UNVERIFIED_LICENCE',source,flush=True);continue
        date=re.search(r'"dateTaken"\s*:\s*"([^"]+)"',text)
        if not photo.exists():
            raw=OUT/(key+'.download');raw.write_bytes(fetch(r['url']))
            try:
                with Image.open(raw) as im:
                    im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1024,1024))
                    if min(im.size)<140:continue
                    im.save(photo,quality=94)
            finally:raw.unlink(missing_ok=True)
        rows.append(dict(image=photo.name,brand=brand,requested_family='(make-only)',title=title,source=source,
                         original_url=r['url'],author=r.get('creator') or '',date=date[1] if date else '',
                         license='CC '+r['license'].upper()+' '+r.get('license_version',''),license_url=r['license_url'],
                         description=r.get('attribution',''),sha256=hashlib.sha256(photo.read_bytes()).hexdigest(),
                         changes='Source Flickr photograph resized and converted to JPEG',provisional_label=True,
                         model_label_available=False,license_verified_on_source=True,annotation='Explicit make in author title; exterior and source make must pass visual review'))
        temporary=manifest.with_suffix('.tmp');temporary.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(manifest)
        seen.add(source);counts[brand]=counts.get(brand,0)+1;print('SAVED',len(rows),brand,title,flush=True)
    except Exception as error:print('SKIP',source,str(error),flush=True)
(OUT/'collection_report.json').write_text(json.dumps(dict(images=len(rows),brand_counts=counts,reviewed=False,trained=False,limits=dict(queries=len(queries),images=180,seconds=3600)),indent=2),encoding='utf-8')
print('COMPLETE',len(rows),flush=True)
