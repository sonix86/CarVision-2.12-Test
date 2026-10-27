"""Actual separate-copy install and offline EXE verification; seal once."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'tools/verify_semantic_v25.py').read_text(encoding='utf-8')
source=source.replace('outputs/CarVision-Update-2.5','outputs/CarVision-Update-2.5.3').replace('outputs/Проверка обновления 2.5/CarVision','outputs/Проверка обновления 2.5.3/CarVision').replace('outputs/semantic25-recipient-verification','outputs/fallback253-recipient-verification').replace('APPLY_UPDATE_2_5.ps1','APPLY_UPDATE_2_5_3.ps1').replace("ROOT/'CarVision-Update-2.5.zip'","ROOT/'CarVision-Update-2.5.3.zip'")
source=source.replace("run('selfcheck'", "run('install-again',[str(ps),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(stage/'APPLY_UPDATE_2_5_3.ps1'),'-Target',str(target)],300)\nrun('selfcheck'")
source=source.replace("(stage/'VERIFICATION.json').write_text", "verification['quality_scope']='Technical checks only; owner BYD/Toyota recognition remains unresolved. Experimental comparison build.'\n(stage/'VERIFICATION.json').write_text")
exec(compile(source,str(ROOT/'tools/verify_semantic_v25.py'),'exec'),dict(__file__=str(ROOT/'tools/verify_semantic_v25.py'),__name__='__main__'))
