"""Validate contents, CRC and embedded SHA256 manifest of the actual delivery ZIP."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root/'TashkentCarVision-v2.1-Portable.zip') as archive:
    names=archive.namelist();assert archive.testzip() is None
    required=['CarVision/source/carvision/body_expert.py','CarVision/models/body_expert_evaluation.json','CarVision/models/body_expert_training.json','CarVision/samples/CHEVROLET_PHOTO_SOURCES.json','CarVision/docs/history-v19/README.md','CarVision/models/visual_references.pt','CarVision/source/carvision/visual_parts.py','CarVision/models/segmenter.pt','CarVision/models/masked_colour.pt','CarVision/INSTANCE_ANALYSIS.md','CarVision/source/carvision/scene.py','CarVision/models/vehicle_model.pt','CarVision/MODEL_AND_COLOUR.md','CarVision/source/carvision/vehicle_model.py','CarVision/CarVision.exe','CarVision/models/classifier.pt','CarVision/models/detector.pt','CarVision/models/colour_model.pt','CarVision/models/colour_training.json','CarVision/PROTOTYPE_VALIDATION.md','CarVision/source/carvision/core.py','CarVision/source/carvision/overlays.py','CarVision/source/tools/train_public_brand.py','CarVision/models/evaluation/public_test.json']
    required.extend(['CarVision/docs/CarVision_Report_v2.1_DRAFT.pdf','CarVision/models/release_v21_validation.json','CarVision/models/release_v21_parameters.json','CarVision/source/carvision/badge_evidence.py','CarVision/data/own_oct21/labels.jsonl'])
    assert all(n in names for n in required)
    assert not any(n.startswith(('CarVision/datasets/','CarVision/cache/','CarVision/models/backups/','CarVision/data/regional/images/','CarVision/source/data/regional/images/','CarVision/data/public_prototype/images/','CarVision/source/data/public_prototype/images/')) or n.endswith('.parquet') or n.endswith('.tmp') for n in names)
    for line in archive.read('CarVision/SHA256SUMS.txt').decode('utf-8').splitlines():
        digest,name=line.split('  ',1);assert hashlib.sha256(archive.read('CarVision/'+name)).hexdigest()==digest,name
    card=json.loads(archive.read('CarVision/models/model_card.json'));assert len(card['supported_brands'])>=13
    assert archive.getinfo('CarVision/models/classifier.pt').file_size>1000000
    for name in ['classifier.pt','detector.pt','colour_model.pt','vehicle_model.pt','segmenter.pt','masked_colour.pt','visual_references.pt']:
        assert hashlib.sha256(archive.read('CarVision/models/'+name)).hexdigest()==hashlib.sha256((root/'models'/name).read_bytes()).hexdigest(),'ZIP differs from the actually tested model: '+name
with zipfile.ZipFile(root/'Tashkent-JPG-Photos.zip') as photos:
    assert photos.testzip() is None
    assert len([n for n in photos.namelist() if n.lower().endswith('.jpg')])==6
    assert not any(n.lower().endswith('.pdf') for n in photos.namelist())
print('Both delivery archives passed CRC, contents and model checks. Program SHA256 manifest fully verified.')

