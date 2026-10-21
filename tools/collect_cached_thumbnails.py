"""One final bounded source attempt, after Retry-After, using standard500px previews.

At most24 photos per EV family, at most2 per author/date session; no recursive
discovery. Reuses cached licence metadata. Independent manifest prevents races
with the Flickr collector. Stops all requests on the first HTTP429.
"""
import collections,hashlib,json,re,sys,time,urllib.request,urllib.parse,urllib.error,html
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image
BASE=ROOT/'datasets/ev_commons_v19';OUT=BASE/'thumb_data';OUT.mkdir(exist_ok=True);HEAD={'User-Agent':'TashkentCarVision/1.9 educational research; standard low-resolution previews'}
f=BASE/'deferred.json';delay=max(0,json.loads(f.read_text()).get('not_before_epoch',0)-time.time()+3)
if delay>700:raise SystemExit('Source embargo exceeds bounded wait')
print('WAIT',round(delay),flush=True);time.sleep(delay)
info={}
for folder in [ROOT/'datasets/regional_commons',BASE]:
 for f in folder.glob('api_*.json'):
  for p in json.loads(f.read_text(encoding='utf-8')).get('query',{}).get('pages',{}).values():
   if p.get('imageinfo'):info[p['title']]=p['imageinfo'][0]
def m(i,k):return html.unescape(re.sub('<[^>]*>','',i.get('extmetadata',{}).get(k,{}).get('value','')))
seeds=[('BYD','Yuan Plus',r'byd.*(?:yuan plus|atto.?3)'),('BYD','Seagull',r'byd.*(?:seagull|dolphin mini)'),('BYD','Qin Plus',r'byd.*qin plus'),('BYD','Seal',r'byd.*seal\b'),('Geely','Galaxy E5',r'geely.*(?:galaxy e5|ex5)'),('Zeekr','001',r'zeekr.*001'),('Chery','Tiggo 8',r'chery.*tiggo.?8'),('Haval','H6',r'haval.*h6')]
seen={r['source'] for r in json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))};jobs=[]
for brand,model,pattern in seeds:
 candidates=[(t,i) for t,i in info.items() if re.search(pattern,t,re.I) and t.lower().endswith(('.jpg','.jpeg','.png')) and not re.search('interior|engine|dashboard|logo|badge|wheel|detail|brochure|toy|scale|concept|facelift',t,re.I) and m(i,'LicenseShortName').startswith(('CC BY','CC0','Public domain')) and i.get('descriptionurl') not in seen]
 counts=collections.Counter();selected=[]
 for t,i in sorted(candidates,key=lambda r:hashlib.sha256(r[0].encode()).hexdigest()):
  session=(m(i,'Artist'),re.sub('[^0-9]','',m(i,'DateTimeOriginal'))[:8])
  if counts[session]>=2:continue
  counts[session]+=1;selected.append((brand,model,t,i))
  if len(selected)>=24:break
 jobs+=selected;print('CANDIDATES',brand,model,len(selected),'sessions',len(counts),flush=True)
manifest=OUT/'manifest.json';rows=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else [];seen|={r['source'] for r in rows};jobs=[r for r in jobs if r[3]['descriptionurl'] not in seen]
last=0
def fetch(url):
 global last
 time.sleep(max(0,4-(time.monotonic()-last)));last=time.monotonic()
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers=HEAD),timeout=35) as r:return r.read()
 except urllib.error.HTTPError as e:
  if e.code==429:
   (BASE/'deferred.json').write_text(json.dumps(dict(host=urllib.parse.urlparse(url).hostname,not_before_epoch=time.time()+max(600,int(e.headers.get('Retry-After','600'))),reason='Final bounded thumbnail attempt throttled; stop, no further automatic collection')))
   raise SystemExit('HTTP429: final source attempt stopped; no further retries')
  raise
thumbs={}
for first in range(0,len(jobs),40):
 titles='|'.join(r[2] for r in jobs[first:first+40]);url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',format='json',prop='imageinfo',titles=titles,iiprop='url',iiurlwidth=500));f=OUT/('api_'+hashlib.sha256(url.encode()).hexdigest()[:20]+'.json')
 if not f.exists():f.write_bytes(fetch(url))
 for p in json.loads(f.read_text(encoding='utf-8')).get('query',{}).get('pages',{}).values():
  if p.get('imageinfo'):thumbs[p['title']]=p['imageinfo'][0].get('thumburl')
for n,(brand,model,title,i) in enumerate(jobs,1):
 url=thumbs.get(title)
 if not url:continue
 try:
  key=hashlib.sha256(title.encode()).hexdigest()[:20];path=OUT/(key+'.jpg');temp=OUT/(key+'.download')
  temp.write_bytes(fetch(url))
  try:
   im=read_image(temp)
   if min(im.size)<140:continue
   im.save(path,quality=94)
  finally:temp.unlink(missing_ok=True)
  rows.append(dict(image='thumb_data/'+path.name,brand=brand,model=model,vehicle_category=brand+' '+model,title=title,source=i['descriptionurl'],original_url=i['url'],download_url=url,author=m(i,'Artist'),license=m(i,'LicenseShortName'),license_url=m(i,'LicenseUrl'),date=m(i,'DateTimeOriginal'),description=m(i,'ImageDescription'),changes='Standard500px Wikimedia thumbnail; JPEG conversion',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),annotation='Explicit author title and cached category, no classifier guesses'))
  tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(manifest);print('SAVED',len(rows),n,'/',len(jobs),flush=True)
 except Exception as e:print('SKIP',title,str(e),flush=True)
print('COMPLETE',len(rows),flush=True)
