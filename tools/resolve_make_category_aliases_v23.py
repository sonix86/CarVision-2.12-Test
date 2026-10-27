"""Resolve case/region/category redirects; these remain provisional source labels."""
import hashlib,json,time,urllib.request,urllib.parse,urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'datasets/ev_catalog_inventory'
inventory=json.loads((OUT/'inventory.json').read_text(encoding='utf-8'));rows=inventory['entries'];requests={}
def candidates(r):
    b,m=r['brand'],r['family'];out=[]
    if b=='Changan':out += ["Chang'an "+m,"Chang'an "+m.replace('BenBen','Benben'),"Changan "+m.replace('BenBen','Benben')]
    if b=='Ora':out += ['ORA '+m]
    if b=='Nio':out += ['NIO '+m]
    if b=='YangWang':out += ['Yangwang '+m]
    if b=='IM Motors':out += ['IM '+m]
    if b=='FangChengBao':out += ['Fangchengbao '+m,'Fangchengbao '+m.replace('Bao ','Leopard ')]
    if b=='Smart':out += ['Smart '+m.replace('#',''),'Smart Hashtag '+m.replace('#','')]
    if b=='Wey':out += ['WEY '+m]
    if b=='iCAR':out += ['ICAR '+m,'iCar '+m,'iCAR '+m]
    if b=='Aito':out += ['AITO '+m]
    if b=='Jiyue':out += ['Ji Yue '+m]
    if b=='Rising Auto':out += ['Rising '+m,'Feifan '+m]
    if b=='MG':out += [m,'MG '+m.replace(' EV','')]
    special={('Baojun','Kiwi EV'):['Baojun KiWi EV','Baojun E300'],('Baojun','Yep Plus'):['Baojun Yep (5-door)'],
             ('Geely','Panda Mini'):['Geely Panda Mini EV'],('Geely','Starwish'):['Geely Xingyuan'],
             ('Changan','Lumin'):['Changan Lumin Corn'],('Changan','Nevo A07'):['Changan Qiyuan A07',"Chang'an Qiyuan A07"],
             ('Changan','Eado EV'):['Changan Eado EV460'],('Chery','Fulwin A8'):['Chery Fengyun A8'],
             ('BYD','Qin L'):['BYD Qin L DM-i'],('Forthing','Friday'):['Forthing Leiting'],
             ('Hongqi','EHS3'):['Hongqi E-HS3'],('Arcfox','Kaola S'):['Arcfox Kaola'],
             ('Arcfox','Alpha T5'):['Arcfox αT5'],('Arcfox','Alpha S5'):['Arcfox αS5'],
             ('Haval','Raptor'):['Haval Menglong'],('Exlantix','ES'):['Exeed Sterra ES'],('Exlantix','ET'):['Exeed Sterra ET'],
             ('Chevrolet','Menlo'):['Chevrolet Changxun'],('BMW','i3 Sedan'):['BMW i3 (G28)'],
             ('Porsche','Macan Electric'):['Porsche Macan (XAB)'],('Zeekr','009 Grand'):['Zeekr 009'],
             ('Tank','400 Hi4-T'):['Tank 400'],('Tank','700 Hi4-T'):['Tank 700']}
    out+=special.get((b,m),[])
    return ['Category:'+n for n in dict.fromkeys(out)]
for i,r in enumerate(rows):
    r['category_candidates']=[c for c in r['category_candidates'] if c.get('files',0)+c.get('subcats',0)>0]
    if not r['category_candidates']:
        for name in candidates(r):requests.setdefault(name,[]).append(i)
names=sorted(requests);found={};last=0
for first in range(0,len(names),40):
    url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(dict(action='query',format='json',prop='categoryinfo',redirects=1,titles='|'.join(names[first:first+40])))
    p=OUT/('aliases_'+hashlib.sha256(url.encode()).hexdigest()[:20]+'.json')
    if not p.exists():
        time.sleep(max(0,10-(time.monotonic()-last)));last=time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'TashkentCarVision/2.3 educational source alias audit'}),timeout=35) as response:p.write_bytes(response.read())
        except urllib.error.HTTPError as error:
            if error.code==429:
                (ROOT/'datasets/make_scope_v23/deferred.json').write_text(json.dumps(dict(not_before_epoch=time.time()+600,reason='HTTP429 alias inventory; no retry')))
            raise
    query=json.loads(p.read_text(encoding='utf-8'))['query'];mapping={r['from']:r['to'] for r in query.get('normalized',[])+query.get('redirects',[])}
    pages={r['title']:r for r in query.get('pages',{}).values()}
    for name in names[first:first+40]:
        resolved=name
        for _ in range(5):
            if resolved not in mapping:break
            resolved=mapping[resolved]
        page=pages.get(resolved,{})
        if 'categoryinfo' in page and page['categoryinfo'].get('files',0)+page['categoryinfo'].get('subcats',0)>0:found[name]=dict(category=resolved,requested_alias=name,**page['categoryinfo'],url='https://commons.wikimedia.org/wiki/'+urllib.parse.quote(resolved.replace(' ','_')))
    print('ALIASES',first,len(names),flush=True)
for name,entry in found.items():
    for i in requests[name]:
        if not any(r['category']==entry['category'] for r in rows[i]['category_candidates']):rows[i]['category_candidates'].append(entry)
inventory.update(families_with_category=sum(bool(r['category_candidates']) for r in rows),alias_queries=len(names),alias_source_labels_review_required=True)
(OUT/'inventory_before_aliases.json').write_text((OUT/'inventory.json').read_text(encoding='utf-8'),encoding='utf-8')
(OUT/'inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
print('COMPLETE',inventory['families_with_category'],len(rows),flush=True)
