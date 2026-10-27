"""Bind the candidate executable to the reviewed local application source."""
import json,types,marshal,hashlib
from pathlib import Path
from PyInstaller.loader.pyimod01_archive import ZlibArchiveReader
from PyInstaller.archive.readers import CArchiveReader
ROOT=Path(__file__).resolve().parents[1];build=ROOT/'outputs/build-v24'
reader=ZlibArchiveReader(str(build/'work/CarVision-v24/PYZ-00.pyz'))
def signature(code):return (code.co_code,tuple(signature(v) if isinstance(v,types.CodeType) else v for v in code.co_consts),code.co_names,code.co_varnames,code.co_freevars,code.co_cellvars)
checked=[]
for path in sorted((ROOT/'carvision').glob('*.py')):
    name='carvision' if path.stem=='__init__' else 'carvision.'+path.stem
    if name not in reader.toc:continue
    assert signature(reader.extract(name))==signature(compile(path.read_text(encoding='utf-8'),str(path),'exec')),name
    checked.append(name)
for name in ('carvision.multimodal_make','carvision.illumination_colour','carvision.vehicle_text','rapidocr_onnxruntime','onnxruntime'):assert name in reader.toc,name
exe=build/'dist/CarVision/CarVision.exe';entry=CArchiveReader(str(exe))
assert signature(marshal.loads(entry.extract('launch')))==signature(compile((ROOT/'launch.py').read_text(encoding='utf-8'),str(ROOT/'launch.py'),'exec'))
report=dict(passed=True,modules=checked,entry_script_matches=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest());(ROOT/'outputs/frozen-v24-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
