"""Layout fixture only; output is not an exam submission."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import carvision.reporting as reporting
from carvision.core import ROOT

fixture=ROOT/'outputs/report-fixture';(fixture/'models/evaluation').mkdir(parents=True,exist_ok=True);(fixture/'data').mkdir(exist_ok=True)
import shutil
for name in ['training.json','detector_training.json']:
    shutil.copy2(ROOT/'outputs/smoke-models'/name,fixture/'models'/name)
for name in ['metrics.json','brand_confusion.png']:
    shutil.copy2(ROOT/'outputs/smoke-models/evaluation'/name,fixture/'models/evaluation'/name)
shutil.copytree(ROOT/'outputs/smoke-data',fixture/'data',dirs_exist_ok=True)
analysis={key:'SYNTHETIC ENGINEERING FIXTURE ONLY. No real car training or phone evaluation is claimed. Кириллица: проверка шрифта.' for key in ['brand_choice','collection_labelling','architecture_rationale','errors','chinese_brands','phone_tests','reflection']}
(fixture/'analysis.json').write_text(json.dumps(analysis),encoding='utf-8')
reporting.ROOT=fixture
reporting.report(argparse.Namespace(team='SYNTHETIC TEST — not student names',repo='https://github.com/EXAMPLE/NOT-A-SUBMISSION',contributions='Automated engineering test only',analysis=str(fixture/'analysis.json'),output=str(fixture/'LAYOUT-TEST-NOT-FOR-SUBMISSION.pdf')))
