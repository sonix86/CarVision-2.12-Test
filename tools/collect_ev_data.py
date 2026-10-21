"""Bounded, resumable Commons collection. Real photographs; per-file credits.

Scenes are unlabelled, never assigned the model name of a foreground vehicle.
Stop the whole collector on HTTP429, honoring the server's Retry-After.
"""
import hashlib,html,json,re,sys,time,urllib.request,urllib.parse,urllib.error
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image
OUT=ROOT/'datasets/ev_commons_v19';OUT.mkdir(exist_ok=True)
SEEDS=[('BYD','Yuan Plus','BYD Yuan Plus'),('BYD','Seagull','BYD Seagull'),('BYD','Qin Plus','BYD Qin Plus'),('BYD','Seal','BYD Seal (BEV)'),('Geely','Galaxy E5','Geely Galaxy E5'),('Zeekr','001','Zeekr 001'),('Chery','Tiggo 8','Chery Tiggo 8'),('Haval','H6 III','Haval H6 (3rd generation)'),('','', 'Car parks in China'),('','','Intersections in China')]
HEAD={'User-Agent':'TashkentCarVision/1.9 (educational research; licence-preserving sequential downloader)'}
deferred=OUT/'deferred.json';old_deferred=ROOT/'datasets/regional_commons/deferred.json'
for f in [deferred,old_deferred]:
 if f.exists() and json.loads(f.read_text()).get('not_before_epoch',0)>time.time():raise SystemExit('Source embargo still active: '+str(f))
last=0
def fetch(url):
 global last
 time.sleep(max(0,2.5-(time.monotonic()-last)));last=time.monotonic()
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers=HEAD),timeout=35) as r:return r.read()
 except urllib.error.HTTPError as e:
  if e.code==429:
   try:delay=int(e.headers.get('Retry-After','600'))
   except ValueError:delay=600
   deferred.write_text(json.dumps(dict(host=urllib.parse.urlparse(url).hostname,not_before_epoch=time.time()+max(600,delay),reason='HTTP429; entire collector stopped, no early retry')))
   raise SystemExit('HTTP429: deferred all downloads; rerun only after Retry-After')
  raise
def api(**params):
 url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',format='json',**params));f=OUT/('api_'+hashlib.sha256(url.encode()).hexdigest()[:20]+'.json')
 previous=ROOT/'datasets/regional_commons'/f.name
 if not f.exists() and previous.exists():f.write_bytes(previous.read_bytes())
 if not f.exists():f.write_bytes(fetch(url))
 return json.loads(f.read_text(encoding='utf-8'))
bad=re.compile('interior|engine|dashboard|logo|badge|wheel|detail|brochure|crash|toy|scale model|diagram',re.I)
excluded={r['title'] for r in json.loads((ROOT/'samples/sources.json').read_text(encoding='utf-8'))}
old=json.loads((ROOT/'datasets/regional_commons/manifest.json').read_text(encoding='utf-8'));seen={r['title'] for r in old}
manifest=OUT/'manifest.json';rows=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else [];seen|={r['title'] for r in rows}
def members(category,depth=0):
 out=[]
 for r in api(list='categorymembers',cmtitle='Category:'+category,cmlimit=500).get('query',{}).get('categorymembers',[]):
  if r['ns']==6 and not bad.search(r['title']):out.append(r['title'])
  elif r['ns']==14 and depth<1 and category not in ['Car parks in China','Intersections in China'] and not bad.search(r['title']):out+=members(r['title'][9:],depth+1)
 return list(dict.fromkeys(out))
jobs=[]
for brand,model,category in SEEDS:
 titles=members(category)
 eligible=[t for t in titles if t not in seen and t[5:] not in excluded and t.lower().endswith(('.jpg','.jpeg','.png')) and (not brand or brand.lower() in t.lower())]
 eligible=sorted(eligible,key=lambda t:hashlib.sha256(t.encode()).hexdigest())[:40 if brand else 12]
 jobs.extend((brand,model,category,t) for t in eligible);seen.update(eligible);print('CATEGORY',category,len(eligible),flush=True)
info={}
for folder in [ROOT/'datasets/regional_commons',OUT]:
 for f in folder.glob('api_*.json'):
  for p in json.loads(f.read_text(encoding='utf-8')).get('query',{}).get('pages',{}).values():
   if p.get('imageinfo'):info[p['title']]=p['imageinfo'][0]
missing=[r for r in jobs if r[3] not in info]
for first in range(0,len(missing),40):
 j=api(prop='imageinfo',titles='|'.join(r[3] for r in missing[first:first+40]),iiprop='url|extmetadata')
 for p in j.get('query',{}).get('pages',{}).values():
  if p.get('imageinfo'):info[p['title']]=p['imageinfo'][0]
for n,(brand,model,category,title) in enumerate(jobs):
 i=info.get(title,{});meta=i.get('extmetadata',{})
 def m(k):return html.unescape(re.sub('<[^>]*>','',meta.get(k,{}).get('value','')))
 licence=m('LicenseShortName')
 if not licence.startswith(('CC BY','CC0','Public domain')):continue
 key=hashlib.sha256(title.encode()).hexdigest()[:20];path=OUT/(key+'.jpg')
 try:
  if not path.exists():
   raw=OUT/(key+'.download');raw.write_bytes(fetch(i['url']))
   try:
    im=read_image(raw);im.thumbnail((1024,1024))
    if min(im.size)<140:continue
    im.save(path,quality=94)
   finally:raw.unlink(missing_ok=True)
  rows.append(dict(image=path.name,brand=brand,model=model,vehicle_category=category,title=title,source=i['descriptionurl'],original_url=i['url'],author=m('Artist'),license=licence,license_url=m('LicenseUrl'),date=m('DateTimeOriginal'),description=m('ImageDescription'),changes='Resized to max1024px; JPEG conversion',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),annotation='category and title; dominant-car crop must pass review' if brand else 'unlabelled traffic/parking scene; NOT model training labels'))
  temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(manifest)
  print('SAVED',len(rows),n+1,'/',len(jobs),title,flush=True)
 except Exception as e:print('SKIP',title,str(e),flush=True)
print('COMPLETE',len(rows),flush=True)
