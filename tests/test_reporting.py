import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from carvision import reporting

class ReportRequirements(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'models/evaluation').mkdir(parents=True);(self.root/'data').mkdir()
        counts=dict(brands={'BYD':1,'Kia':1},colours={'white':2},sources={'synthetic test':2})
        training=dict(data=counts,train=counts,validation=counts,seed=1,parameters=50,loss='test',optimizer='test',lr=.001,epochs=1,batch=2,hardware='test',seconds=1,augmentation='test')
        (self.root/'models/training.json').write_text(json.dumps(training),encoding='utf-8')
        (self.root/'models/detector_training.json').write_text(json.dumps(dict(parameters=20,epochs=1,seconds=1,history=[{}])),encoding='utf-8')
        (self.root/'models/evaluation/metrics.json').write_text(json.dumps(dict(validation_cars=2,brand_accuracy=.5,colour_accuracy=.5,scope='synthetic test')),encoding='utf-8')
        (self.root/'data/labels.jsonl').write_text('',encoding='utf-8')
        self.analysis={key:'synthetic engineering test only' for key in ['brand_choice','collection_labelling','architecture_rationale','errors','chinese_brands','phone_tests','reflection']}
        self.args=SimpleNamespace(team='Test',repo='https://github.com/EXAMPLE/TEST',contributions='Test only',analysis=str(self.root/'analysis.json'),output=str(self.root/'report.pdf'))
    def tearDown(self):self.temp.cleanup()
    def test_more_than_six_pages_is_rejected_without_publishing_final_file(self):
        (self.root/'analysis.json').write_text(json.dumps(self.analysis),encoding='utf-8')
        document=Mock();document.page=7
        with patch.object(reporting,'ROOT',self.root),patch.object(reporting,'SimpleDocTemplate',return_value=document):
            with self.assertRaisesRegex(ValueError,'limit is 6'):reporting.report(self.args)
        self.assertFalse((self.root/'report.pdf').exists())
    def test_report_requires_actual_architecture_rationale(self):
        self.analysis.pop('architecture_rationale');(self.root/'analysis.json').write_text(json.dumps(self.analysis),encoding='utf-8')
        with patch.object(reporting,'ROOT',self.root):
            with self.assertRaisesRegex(ValueError,'architecture_rationale'):reporting.report(self.args)

if __name__=='__main__':unittest.main()
