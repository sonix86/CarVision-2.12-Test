"""Supplementary public photographs, never attributed to the student.

Labels come from explicit Commons vehicle categories + matching file titles.
Only reusable licences are accepted. Six development examples are excluded.
Run preparation/training separately; downloads are not runtime dependencies.
"""
import concurrent.futures,hashlib,html,json,re,sys,time,urllib.parse,urllib.request,threading,urllib.error,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image

OUT=ROOT/'datasets/regional_commons';OUT.mkdir(parents=True,exist_ok=True)
HEADERS={'User-Agent':'TashkentCarVision/1.3 (educational computer vision research)'}
API_LOCK=threading.Lock();API_LAST=0.
DOWNLOAD_LOCK=threading.Lock();DOWNLOAD_LAST=0.
EMBARGO={}
deferred_file=OUT/'deferred.json'
if deferred_file.exists():
 deferred=json.loads(deferred_file.read_text())
 if deferred.get('not_before_epoch',0)>time.time():raise SystemExit('Downloads deferred until '+time.ctime(deferred['not_before_epoch'])+'; already downloaded data remains available.')
SEEDS={
 'Chevrolet':['Chevrolet Tracker (4th generation)','Chevrolet Cobalt (Gamma II)','Chevrolet Onix (GEM) sedan','Chevrolet Nexia','Chevrolet Spark (M300)','Chevrolet Spark (M200)','Chevrolet Captiva','Chevrolet Damas','Chevrolet Lacetti','Chevrolet Malibu (E2XX)'],
 'Daewoo':['Daewoo Nexia','Daewoo Matiz'],
 'BYD':['BYD Song Plus','BYD Yuan Plus','BYD Seagull','BYD Qin Plus'],
 'Kia':['Kia K5 (DL3)','Kia Seltos','Kia Sportage (NQ5)'],
 'Lada':['VAZ-2101','VAZ-2107','Lada Vesta'],
 'Chery':['Chery Tiggo 7','Chery Tiggo 8'],
 'Haval':['Haval Jolion','Haval H6 (3rd generation)']}
EXCLUDE=re.compile(r'interior|engine|dashboard|logo|badge|wheel|detail|advert|brochure|presskit|crash|toy|scale model|police|competition|racing',re.I)
def fetch(url):
 global DOWNLOAD_LAST
 for attempt in range(4):
  try:
   host=urllib.parse.urlparse(url).hostname
   if EMBARGO.get(host,0)>time.time():raise RuntimeError('Host deferred after HTTP429; no early retry')
   if 'upload.wikimedia.org' in url:
    with DOWNLOAD_LOCK:
     time.sleep(max(0,1.2-(time.monotonic()-DOWNLOAD_LAST)));DOWNLOAD_LAST=time.monotonic()
   with urllib.request.urlopen(urllib.request.Request(url,headers=HEADERS),timeout=30) as r:return r.read()
  except urllib.error.HTTPError as e:
   if attempt==3:raise
   delay=max(int(e.headers.get('Retry-After','30')),3)
   print('HTTP',e.code,'retry-after',delay,'attempt',attempt+1,flush=True)
   # A file requiring a long embargo is deferred; never retry it early or
   # let it occupy all download workers while other public files are available.
   if e.code==429 and delay>60:
    EMBARGO[host]=time.time()+delay;deferred_file.write_text(json.dumps(dict(host=host,not_before_epoch=EMBARGO[host],reason='HTTP429 Retry-After; no early retries',remaining_downloads_deferred=True),indent=2));raise
   time.sleep(delay)
  except Exception:
   if attempt==3:raise
   time.sleep(2+attempt*3)
def api(**params):
 global API_LAST
 url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',format='json',**params))
 cache=OUT/('api_'+hashlib.sha256(url.encode()).hexdigest()[:20]+'.json')
 if cache.exists():return json.loads(cache.read_text(encoding='utf-8'))
 with API_LOCK:
  time.sleep(max(0,3-(time.monotonic()-API_LAST)))
  raw=fetch(url);API_LAST=time.monotonic();cache.write_bytes(raw)
 return json.loads(raw)
def members(category,depth=0):
 j=api(list='categorymembers',cmtitle='Category:'+category,cmlimit=500)
 out=[]
 for r in j.get('query',{}).get('categorymembers',[]):
  if r['ns']==6 and not EXCLUDE.search(r['title']):out.append(r['title'])
  elif r['ns']==14 and depth<1 and not EXCLUDE.search(r['title']):out+=members(r['title'][9:],depth+1)
 return list(dict.fromkeys(out))
excluded={r['title'] for r in json.loads((ROOT/'samples/sources.json').read_text(encoding='utf-8'))}
manifest=OUT/'manifest.json';rows=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
seen={r['title'] for r in rows};jobs=[]
for brand,categories in SEEDS.items():
 for category in categories:
  try:titles=members(category)
  except Exception as e:print('CATEGORY ERROR',category,str(e),flush=True);continue
  eligible=[t for t in titles if t[5:] not in excluded and t.lower().endswith(('.jpg','.jpeg','.png')) and (brand.lower() in t.lower() or (brand=='Lada' and 'vaz' in t.lower()))]
  # Deterministic, independent of download order and model predictions.
  eligible=[t for t in sorted(eligible,key=lambda t:hashlib.sha256(t.encode()).hexdigest())[:65] if t not in seen]
  for title in eligible:jobs.append((brand,category,title));seen.add(title)
  print('CATEGORY',brand,category,len(eligible),flush=True)
INFO={}
for file in OUT.glob('api_*.json'):
 cached=json.loads(file.read_text(encoding='utf-8'))
 for p in cached.get('query',{}).get('pages',{}).values():
  if p.get('imageinfo'):INFO[p['title']]=p['imageinfo'][0]
missing=[job for job in jobs if job[2] not in INFO]
for start in range(0,len(missing),40):
 batch=missing[start:start+40]
 j=api(prop='imageinfo',titles='|'.join(job[2] for job in batch),iiprop='url|extmetadata',iiurlwidth=960)
 for p in j.get('query',{}).get('pages',{}).values():
  if p.get('imageinfo'):INFO[p['title']]=p['imageinfo'][0]
 print('METADATA',start+len(batch),'/',len(missing),flush=True)
# Round-robin brands so a resumed collection does not starve later brands.
print('METADATA READY',len(INFO),'jobs',len(jobs),flush=True)
queues={b:collections.deque(job for job in jobs if job[0]==b) for b in SEEDS};jobs=[]
while any(queues.values()):
 for queue in queues.values():
  if queue:jobs.append(queue.popleft())
def download(job):
 brand,category,title=job
 info=INFO.get(title,{})
 if not info:return None
 meta=info.get('extmetadata',{})
 def m(k):return html.unescape(re.sub('<[^>]*>','',meta.get(k,{}).get('value','')))
 license=m('LicenseShortName')
 if not (license.startswith(('CC BY','CC0','Public domain'))):return None
 # Original files are served successfully even when thumbnail rendering fails.
 # Resize locally rather than repeatedly asking Thumbor to regenerate previews.
 url=info['url'];key=hashlib.sha256(title.encode()).hexdigest()[:20];path=OUT/(key+'.jpg')
 if not path.exists():
  raw=fetch(url);tmp=OUT/(key+'.download');tmp.write_bytes(raw)
  try:
   image=read_image(tmp);image.thumbnail((960,960))
   if min(image.size)<140:return None
   image.save(path,quality=94)
  finally:tmp.unlink(missing_ok=True)
 return dict(image=path.name,brand=brand,vehicle_category=category,title=title,source=info['descriptionurl'],original_url=info['url'],author=m('Artist'),license=license,license_url=m('LicenseUrl'),date=m('DateTimeOriginal'),description=m('ImageDescription'),changes='Original Commons image resized locally to max960px and converted to JPEG; detector crop prepared separately',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures={pool.submit(download,job):job for job in jobs}
 for n,f in enumerate(concurrent.futures.as_completed(futures),1):
  try:
   row=f.result()
   if row:rows.append(row)
  except Exception as e:print('DOWNLOAD ERROR',futures[f][2],str(e),flush=True)
  if n%20==0 or n==len(jobs):
   temporary=manifest.with_suffix('.tmp');temporary.write_text(json.dumps(sorted(rows,key=lambda r:r['title']),ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(manifest);print('SAVED',len(rows),'processed',n,'/',len(jobs),flush=True)
print('COMPLETE',len(rows),flush=True)
