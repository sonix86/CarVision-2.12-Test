"""Validate contents, CRC and embedded SHA256 manifest of the actual delivery ZIP."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root/'TashkentCarVision-Portable.zip') as archive:
    names=archive.namelist();assert archive.testzip() is None
    required=['CarVision/CarVision.exe','CarVision/models/classifier.pt','CarVision/models/detector.pt','CarVision/models/colour_model.pt','CarVision/models/colour_training.json','CarVision/PROTOTYPE_VALIDATION.md','CarVision/source/carvision/core.py','CarVision/source/tools/train_public_brand.py','CarVision/models/evaluation/public_test.json']
    assert all(n in names for n in required)
    assert not any('/datasets/' in n or '/cache/' in n or '/data/public_prototype/images/' in n or n.endswith('.parquet') or n.endswith('.tmp') for n in names)
    for line in archive.read('CarVision/SHA256SUMS.txt').decode('utf-8').splitlines():
        digest,name=line.split('  ',1);assert hashlib.sha256(archive.read('CarVision/'+name)).hexdigest()==digest,name
    card=json.loads(archive.read('CarVision/models/model_card.json'));assert len(card['supported_brands'])==13
    assert archive.getinfo('CarVision/models/classifier.pt').file_size>1000000
with zipfile.ZipFile(root/'Tashkent-JPG-Photos.zip') as photos:
    assert photos.testzip() is None
    assert len([n for n in photos.namelist() if n.lower().endswith('.jpg')])==6
    assert not any(n.lower().endswith('.pdf') for n in photos.namelist())
print('Both delivery archives passed CRC, contents and model checks. Program SHA256 manifest fully verified.')
