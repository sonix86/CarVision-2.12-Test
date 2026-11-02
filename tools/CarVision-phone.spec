# Build from the repository root with PyInstaller 6.16 / Python 3.12 x64.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files
project=Path(SPECPATH).resolve().parent
datas=collect_data_files('torchvision')+collect_data_files('matplotlib')
a=Analysis([str(project/'launch.py')],pathex=[str(project)],binaries=[],datas=datas,hiddenimports=['PIL._tkinter_finder'],hookspath=[],hooksconfig={},runtime_hooks=[],excludes=['pyarrow','sklearn','open_clip','timm','onnxruntime','rapidocr_onnxruntime','shapely'],noarchive=False,optimize=0)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='CarVision',debug=False,bootloader_ignore_signals=False,strip=False,upx=True,console=True,disable_windowed_traceback=False,argv_emulation=False,target_arch=None,codesign_identity=None,entitlements_file=None)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=True,upx_exclude=[],name='CarVision')
