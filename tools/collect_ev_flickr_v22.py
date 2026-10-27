"""Bounded photo research through Openverse's public search API, Flickr only.

No catalog scraping/pagination, account, key or paid service. Each photograph's
licence is checked on its original Flickr landing page before downloading it.
Wikimedia rate limits remain respected; no Wikimedia mirror/proxy is used.
"""
import hashlib,json,re,sys,time,urllib.request,urllib.parse,urllib.error
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image
OUT=ROOT/'datasets/ev_patch_flickr_v22';OUT.mkdir(exist_ok=True);CACHE=OUT/'flickr_metadata';CACHE.mkdir(exist_ok=True)
HEAD={'User-Agent':'TashkentCarVision/2.2 educational research (licence-preserving photo collection)'}
QUERIES=['Weltmeister EX5', 'Hongqi E-QM5', 'BYD Dolphin', 'BYD Han', 'BYD Seagull', 'BYD Seal', 'BYD Atto 3', 'Nissan Ariya', 'Toyota bZ4X', 'Nissan Leaf', 'XPeng G6', 'Zeekr X']
def fetch(url):
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers=HEAD),timeout=30) as r:return r.read()
 except urllib.error.HTTPError as e:
  if e.code==429:raise SystemExit('Flickr/Openverse HTTP429: collector stopped; no retries')
  raise
found={}
for f in CACHE.glob('initial-*.json'):
 for r in json.loads(f.read_text(encoding='utf-8')).get('results',[]):
  if r.get('source')=='flickr' and r.get('license') in ['by','by-sa','cc0','pdm']:found[r['id']]=r
for query in QUERIES:
 key=hashlib.sha256(query.encode()).hexdigest()[:20];f=CACHE/('search_'+key+'.json')
 if not f.exists():
  time.sleep(8)
  url='https://api.openverse.org/v1/images/?'+urllib.parse.urlencode(dict(q=query,source='flickr',license='by,by-sa,cc0,pdm',page_size=20))
  try:f.write_bytes(fetch(url))
  except Exception as e:print('SEARCH ERROR',query,str(e),flush=True);continue
 for r in json.loads(f.read_text(encoding='utf-8')).get('results',[]):
  if r.get('source')=='flickr' and r.get('license') in ['by','by-sa','cc0','pdm']:found[r['id']]=r
 print('SEARCH',query,'unique',len(found),flush=True)
manifest=OUT/'manifest.json';rows=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else [];seen={r['source'] for r in rows}
patterns=[('Weltmeister', 'EX5', 'Weltmeister.*EX5\\b'), ('Hongqi', 'E-QM5', 'Hongqi.*E-QM5\\b'), ('BYD', 'Dolphin', 'BYD.*Dolphin\\b'), ('BYD', 'Han', 'BYD.*Han\\b'), ('BYD', 'Seagull', 'BYD.*Seagull\\b'), ('BYD', 'Seal', 'BYD.*Seal\\b'), ('BYD', 'Yuan Plus', 'BYD.*Atto 3\\b'), ('Nissan', 'Ariya', 'Nissan.*Ariya\\b'), ('Toyota', 'bZ4X', 'Toyota.*bZ4X\\b'), ('Nissan', 'Leaf', 'Nissan.*Leaf\\b'), ('XPeng', 'G6', 'XPeng.*G6\\b'), ('Zeekr', 'X', 'Zeekr.*X\\b')]
counts={};jobs=[]
for r in found.values():
 title=r['title'];pair=next(((b,m) for b,m,pattern in patterns if re.search(pattern,title,re.I)),None)
 if not pair:
  continue
  pair=('','')
 if re.search(r'interior|badge|logo|toy|model car|render|3d|illustration|scale model',title,re.I):continue
 if r['foreign_landing_url'] in seen or counts.get(pair,0)>=50:continue
 counts[pair]=counts.get(pair,0)+1;jobs.append((r,pair))
print('JOBS',len(jobs),counts,flush=True)
for n,(r,(brand,model)) in enumerate(jobs,1):
 source=r['foreign_landing_url'];key=hashlib.sha256(source.encode()).hexdigest()[:20];page=CACHE/(key+'.html');path=OUT/(key+'.jpg')
 try:
  if not page.exists():time.sleep(2);page.write_bytes(fetch(source))
  html=page.read_text(encoding='utf-8');licence=re.search(r'"license"\s*:\s*"([^"]+)"',html)
  if not licence or licence[1].rstrip('/')!=r['license_url'].rstrip('/'):print('SKIP UNVERIFIED LICENCE',source,flush=True);continue
  date=re.search(r'"dateTaken"\s*:\s*"([^"]+)"',html)
  if not path.exists():
   time.sleep(2);temp=OUT/(key+'.download');temp.write_bytes(fetch(r['url']))
   try:
    im=read_image(temp);im.thumbnail((1024,1024))
    if min(im.size)<140:continue
    im.save(path,quality=94)
   finally:temp.unlink(missing_ok=True)
  row=dict(image=path.name,brand=brand,model=model,vehicle_category=(brand+' '+model).strip(),title=r['title'],source=source,original_url=r['url'],author=r['creator'],license='CC '+r['license'].upper()+' '+r['license_version'],license_url=r['license_url'],date=date[1] if date else '',description=r['attribution'],changes='Original Flickr photo resized to max1024px; JPEG conversion',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),annotation='Explicit author title, not a classifier guess; dominant-car crop reviewed separately' if brand else 'Unlabelled parking/traffic scene; not model-name ground truth',license_verified_on_source=True,search_source='Openverse bounded first-page search, Flickr source only')
  rows.append(row);tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(manifest)
  print('SAVED',len(rows),n,'/',len(jobs),source,flush=True)
 except Exception as e:print('SKIP',source,str(e),flush=True)
print('COMPLETE',len(rows),flush=True)
