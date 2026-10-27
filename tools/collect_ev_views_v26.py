"""A separate newly authorized finite collection; never reset v23's budget."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'tools/collect_make_scope_v23.py').read_text(encoding='utf-8')
source=source.replace("datasets/make_scope_v23","datasets/ev_views_v26").replace('seconds=4*3600','seconds=1800').replace('LIMIT=2000','LIMIT=240').replace('Four-hour collection bound','Thirty-minute collection bound')
source=source.replace("['regional_commons','ev_commons_v19','ev_patch_v22','ev_patch_flickr_v22','chevrolet_v20']", "['regional_commons','ev_commons_v19','ev_patch_v22','ev_patch_flickr_v22','chevrolet_v20','make_scope_v23','make_flickr_v23']")
source=source.replace("['regional_commons','ev_patch_v22','ev_commons_v19']", "['regional_commons','ev_patch_v22','ev_commons_v19','make_scope_v23']")
source=source.replace('counts.get(brand,0)>=96','counts.get(brand,0)>=28').replace('family_limit=24 if family_ranks[(brand,family)]==0 else 8','family_limit=4')
source=source.replace("limits=dict(seconds=4*3600,images=LIMIT,per_make=96,primary_family=24,other_family=8,per_author_date=2)","limits=dict(seconds=1800,images=LIMIT,per_make=28,per_family=4,per_author_date=2)")
source=source.replace('TashkentCarVision/2.3','TashkentCarVision/2.6')
marker="bad=re.compile("
insert="""# Target body/angle breadth, rather than resampling only the first family.
priority={'BYD':['Song Plus','Seagull','Yuan Plus','Han','Seal','Dolphin','Qin Plus','e2'],
'Weltmeister':['EX5','W6'],'Li Auto':['L6','L7','L8','L9','One'],
'Zeekr':['001','007','7X','X','009'],'GAC Aion':['S','S Plus','Y','Y Plus','V'],
'Geely':['Galaxy E5','Galaxy E8','Galaxy L7','Starwish'],
'Geometry':['A','C','E','G6'],'Deepal':['S07','SL03','S05'],
'Leapmotor':['T03','C11','C10','C01'],'Neta':['V','U','X','S'],
'XPeng':['Mona M03','G6','P7'],'Changan':['Lumin','Nevo A07','Qiyuan A05'],
'Toyota':['bZ4X','bZ3'],'Honda':['e:NS1','e:NP1'],
'Nio':['ET5','ES6'],'Avatr':['11','12'],'Voyah':['Free'],
'Hongqi':['E-QM5','E-HS9'],'Wuling':['Bingo','Hongguang Mini EV']}
inventory=[r for r in inventory if r['family'] in priority.get(r['brand'],[])]
inventory.sort(key=lambda r:(priority[r['brand']].index(r['family']),list(priority).index(r['brand'])))
"""
source=source.replace(marker,insert+marker)
exec(compile(source,str(ROOT/'tools/collect_make_scope_v23.py'),'exec'),dict(__file__=str(ROOT/'tools/collect_make_scope_v23.py'),__name__='__main__'))
