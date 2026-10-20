"""Freeze a balanced downloaded snapshot, then prepare and train it.

Later downloads are not silently incorporated after the split is frozen.
"""
import collections,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
last=None
while True:
    try:
        rows=json.loads((ROOT/'datasets/regional_commons/manifest.json').read_text(encoding='utf-8'));counts=collections.Counter(r['brand'] for r in rows)
        if counts!=last:print('DOWNLOAD COUNTS',dict(counts),flush=True);last=counts
        complete=all(counts[b]>=70 for b in ['Daewoo','BYD','Kia','Lada','Chery','Haval'])
        done='COMPLETE' in (ROOT/'outputs/regional-download.log').read_text(encoding='utf-8',errors='replace')
        if complete or done:break
    except (OSError,json.JSONDecodeError):pass
    time.sleep(10)
print('Freezing downloaded snapshot for this experiment.',flush=True)
subprocess.run([sys.executable,str(ROOT/'tools/prepare_regional_data.py')],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(ROOT/'tools/train_regional_brand.py')],cwd=ROOT,check=True)
