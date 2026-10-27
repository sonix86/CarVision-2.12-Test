"""Finite, licence-preserving collection across the user's requested scope.

Category/title labels are provisional and must pass exterior review before training.
No generated photographs, automotive weights, payments, or retry loops.
"""
import argparse,hashlib,html,json,re,sys,time,urllib.request,urllib.parse,urllib.error
from pathlib import Path
from PIL import Image,ImageOps

sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
parser=argparse.ArgumentParser();parser.add_argument('--breadth-only',action='store_true');args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/make_scope_v23';OUT.mkdir(exist_ok=True)
budget_file=OUT/'run_budget.json'
budget=json.loads(budget_file.read_text(encoding='utf-8-sig')) if budget_file.exists() else dict(started_epoch=time.time(),seconds=4*3600)
if not budget_file.exists():budget_file.write_text(json.dumps(budget),encoding='utf-8')
started=budget['started_epoch'];deadline=started+budget['seconds'];last=0;LIMIT=2000
for embargo in (ROOT/'datasets').glob('*/deferred.json'):
    try:state=json.loads(embargo.read_text(encoding='utf-8'))
    except (ValueError,OSError):continue
    if state.get('not_before_epoch',0)>time.time():raise SystemExit('Source embargo active: '+str(embargo))
manifest=OUT/'manifest.json';rows=json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
inventory=json.loads((ROOT/'datasets/ev_catalog_inventory/inventory.json').read_text(encoding='utf-8'))['entries']
targets=json.loads((ROOT/'data/catalogue/ev_target_catalog.json').read_text(encoding='utf-8'))['entries']
order={(r['brand'],r['family']):min(r['requested_ranges']) for r in targets}
family_ranks={}
for brand in {r['brand'] for r in inventory}:
    entries=[r for r in inventory if r['brand']==brand and r['category_candidates']]
    entries.sort(key=lambda r:(order[(r['brand'],r['family'])],-sum(c.get('files',0)+c.get('subcats',0)*15 for c in r['category_candidates']),r['family']))
    for rank,r in enumerate(entries):family_ranks[(brand,r['family'])]=rank
inventory.sort(key=lambda r:(family_ranks.get((r['brand'],r['family']),1000),order[(r['brand'],r['family'])],r['brand'],r['family']))
bad=re.compile(r'interior|engine|dashboard|logo|badge|wheel|brochure|crash|concept|scale model|diagram|\btoy\b|eAxle|chassis|door handle|retouched|changed numberplates|IMG\d+crop|\bmod\b|bearb\s*Sp',re.I)
seen={r['title'] for r in rows};seen_sources={r['source'] for r in rows}
for folder in ['regional_commons','ev_commons_v19','ev_patch_v22','ev_patch_flickr_v22','chevrolet_v20']:
    p=ROOT/'datasets'/folder/'manifest.json'
    if p.exists():
        for r in json.loads(p.read_text(encoding='utf-8')):
            seen.add(r.get('title',''));seen_sources.add(r.get('source',''))
excluded=json.loads((ROOT/'samples/sources.json').read_text(encoding='utf-8'))
seen.update(r['title'] if r['title'].startswith('File:') else 'File:'+r['title'] for r in excluded)

def fetch(url):
    global last
    if time.time()>deadline:raise SystemExit('Four-hour collection bound reached; state saved; no automatic retry.')
    time.sleep(max(0,(10 if '/w/api.php?' in url else 4)-(time.monotonic()-last)));last=time.monotonic()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TashkentCarVision/2.3 educational licence-preserving sequential collection'}),timeout=35) as response:return response.read()
    except urllib.error.HTTPError as error:
        if error.code==429:
            try:delay=int(error.headers.get('Retry-After','600'))
            except ValueError:delay=600
            (OUT/'deferred.json').write_text(json.dumps(dict(not_before_epoch=time.time()+max(600,delay),reason='HTTP429, entire collector stopped without retry')))
            raise SystemExit('HTTP429; collector stopped and saved its state.')
        raise

def api(**params):
    url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',format='json',**params))
    name='api_'+hashlib.sha256(url.encode()).hexdigest()[:20]+'.json';p=OUT/name
    if not p.exists():
        for folder in ['regional_commons','ev_patch_v22','ev_commons_v19']:
            old=ROOT/'datasets'/folder/name
            if old.exists():p.write_bytes(old.read_bytes());break
    if not p.exists():p.write_bytes(fetch(url))
    return json.loads(p.read_text(encoding='utf-8'))

def members(category,depth=0):
    values=api(list='categorymembers',cmtitle=category,cmlimit=500).get('query',{}).get('categorymembers',[]);names=[]
    for r in values:
        if bad.search(r['title']):continue
        if r['ns']==6 and r['title'].lower().endswith(('.jpg','.jpeg','.png')):names.append(r['title'])
        elif r['ns']==14 and depth<1:names+=members(r['title'],depth+1)
    return list(dict.fromkeys(names))

counts={};sessions={}
for r in rows:
    counts[r['brand']]=counts.get(r['brand'],0)+1
    sessions[(r['brand'],r['author'],re.sub(r'[^0-9]','',r['date'])[:8])]=1
for index,entry in enumerate(inventory):
    brand,family=entry['brand'],entry['family']
    if len(rows)>=LIMIT or time.time()>deadline:break
    if args.breadth_only and family_ranks.get((brand,family))!=0:continue
    if counts.get(brand,0)>=96:continue
    categories=entry['category_candidates']
    if not categories:continue
    saved=sum(r['brand']==brand and r['requested_family']==family for r in rows)
    family_limit=24 if family_ranks[(brand,family)]==0 else 8
    if saved>=family_limit:continue
    titles=[]
    try:
        for category in categories:titles+=members(category['category'])
        titles=sorted({t for t in titles if t not in seen},key=lambda t:hashlib.sha256(t.encode()).hexdigest())[:48]
        info={}
        for first in range(0,len(titles),40):
            result=api(prop='imageinfo',titles='|'.join(titles[first:first+40]),iiprop='url|extmetadata',iiurlwidth=640)
            for page in result.get('query',{}).get('pages',{}).values():
                if page.get('imageinfo'):info[page['title']]=page['imageinfo'][0]
        for title in titles:
            if saved>=family_limit or len(rows)>=LIMIT or counts.get(brand,0)>=96:break
            item=info.get(title,{});meta=item.get('extmetadata',{})
            def m(k):return html.unescape(re.sub('<[^>]*>','',meta.get(k,{}).get('value','')))
            license=m('LicenseShortName');source=item.get('descriptionurl','')
            if not license.startswith(('CC BY','CC0','Public domain')) or not source or source in seen_sources:continue
            author,date=m('Artist'),m('DateTimeOriginal');session=(brand,author,re.sub(r'[^0-9]','',date)[:8])
            if sessions.get(session,0)>=2:continue
            key=hashlib.sha256(title.encode()).hexdigest()[:20];path=OUT/(key+'.jpg')
            if not path.exists():
                raw=OUT/(key+'.download');raw.write_bytes(fetch(item.get('thumburl',item['url'])))
                try:
                    with Image.open(raw) as im:
                        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1024,1024))
                        if min(im.size)<140:continue
                        im.save(path,quality=94)
                finally:raw.unlink(missing_ok=True)
            row=dict(image=path.name,brand=brand,requested_family=family,title=title,source=source,original_url=item['url'],
                     author=author,date=date,license=license,license_url=m('LicenseUrl'),description=m('ImageDescription'),
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest(),changes='Commons thumbnail resized and converted to JPEG',
                     provisional_label=True,annotation='Category-derived make; must pass vehicle/exterior/manual source-label review. Model not certified.')
            rows.append(row);seen.add(title);seen_sources.add(source);saved+=1;counts[brand]=counts.get(brand,0)+1;sessions[session]=sessions.get(session,0)+1
            temporary=manifest.with_suffix('.tmp');temporary.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8');temporary.replace(manifest)
            print('SAVED',len(rows),brand,family,title,flush=True)
        print('CATEGORY',index+1,len(inventory),brand,family,'images',saved,flush=True)
    except (urllib.error.URLError,OSError,ValueError) as error:
        print('SOURCE_ERROR',brand,family,type(error).__name__,str(error),flush=True)
summary=dict(completed=True,total=len(rows),brand_counts=counts,seconds=time.time()-started,breadth_only=args.breadth_only,
             limits=dict(seconds=4*3600,images=LIMIT,per_make=96,primary_family=24,other_family=8,per_author_date=2),
             reviewed=False,trained=False)
(OUT/'collection_report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',json.dumps(summary),flush=True)
