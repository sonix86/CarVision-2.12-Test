"""Reproduce real initialization and fallback bugs found during the full audit."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,torch
from PIL import Image
from carvision.core import Pipeline,COLOURS
from carvision.colour_model import JointColourClassifier

class AuditRegressions(unittest.TestCase):
    def test_personal_training_cannot_be_hidden_by_old_specialist_weights(self):
        from carvision.training import train
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'dino_make.pt').write_bytes(b'previous specialist')
            with patch('carvision.training.load_records') as loader:
                with self.assertRaisesRegex(ValueError,'отдельную папку'):train(SimpleNamespace(output=str(p)))
                loader.assert_not_called()
            self.assertEqual((p/'dino_make.pt').read_bytes(),b'previous specialist')

    def test_mixed_encoder_weights_are_rejected_before_inference(self):
        import hashlib
        from carvision.model_integrity import verify_clip_pair
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'semantic_make.pt';p.write_bytes(b'paired encoder checkpoint')
            ck=dict(kind='joint',clip_checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            self.assertEqual(verify_clip_pair(ck,p),ck['clip_checkpoint_sha256'])
            p.write_bytes(b'another version')
            with self.assertRaisesRegex(ValueError,'разных наборов'):verify_clip_pair(ck,p)
            p.unlink()
            with self.assertRaisesRegex(ValueError,'Отсутствует'):verify_clip_pair(ck,p)

    def test_non_joint_and_legacy_heads_do_not_require_unused_weights(self):
        from carvision.model_integrity import verify_clip_pair
        self.assertIsNone(verify_clip_pair(dict(kind='dino'),'missing.pt'))
        self.assertIsNone(verify_clip_pair(dict(kind='joint'),'missing.pt'))

    def test_default_loads_separate_colour_without_legacy_mode_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);torch.save(dict(origin='team-trained-from-scratch',state={}),p/'detector.pt');torch.save(dict(dummy=True),p/'colour_model.pt');marker=object()
            with patch('carvision.core.ssdlite320_mobilenet_v3_large',return_value=torch.nn.Identity()),patch('carvision.colour_model.ColourClassifier',return_value=marker) as ctor:
                pipeline=Pipeline(p)
            self.assertIs(pipeline.colour_classifier,marker);ctor.assert_called_once()

    def test_joint_colour_supports_the_predict_api_used_by_box_pipeline(self):
        model=JointColourClassifier(lambda x:(torch.zeros(len(x),2),torch.tensor([[0.,0.,0.,0.,0.,9.,0.,0.]]).expand(len(x),-1)),lambda im:torch.ones(3,16,16))
        colour,score=model.predict(Image.new('RGB',(50,40),'green'))
        self.assertEqual(colour,'green');self.assertGreater(score,.99)

    def test_box_detector_uses_new_make_expert_instead_of_old_classifier(self):
        with tempfile.TemporaryDirectory() as folder:
            photo=Path(folder)/'photo.png';Image.new('RGB',(100,80),'white').save(photo)
            p=object.__new__(Pipeline);p.mode='fast';p.detection=.5;p.brand_threshold=.5;p.colour_mode='trained';p.brands=['Chevrolet'];p.colour_classifier=None;p.vehicle_classifier=None;p.masked_colour=None;p.colour_reliability_threshold=.5
            p.classifier=lambda x:(_ for _ in ()).throw(AssertionError('Old classifier must not replace active make expert'))
            evidence=dict(brand='BYD',candidate='BYD',display_name='BYD',confidence=.99,accepted=True,ranking=[dict(brand='BYD',confidence=.99)],alternatives=[dict(brand='BYD',confidence=.99)],source='own test make head')
            p.make_expert=SimpleNamespace(predict=lambda *a:dict(evidence));p.detector=lambda images:[dict(boxes=torch.tensor([[5.,5.,90.,75.]]),labels=torch.tensor([1]),scores=torch.tensor([.99]))]
            p.runtime_identity=dict(active_head_file='dino_make.pt')
            r=p.predict(photo);self.assertEqual(r['cars'][0]['brand'],'BYD');self.assertTrue(r['cars'][0]['make_first']);self.assertEqual(r['runtime_identity']['active_head_file'],'dino_make.pt')

    def test_identical_pixels_with_different_metadata_stay_in_one_split(self):
        from carvision.data import load_records,split_records
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);records=[]
            from PIL.PngImagePlugin import PngInfo
            for i in range(12):
                im=Image.new('RGB',(24,24),(i//2*35,0,0));info=PngInfo();info.add_text('note',str(i));im.save(p/f'{i}.png',pnginfo=info)
                records.append(dict(image=f'{i}.png',box=[0,0,24,24],brand='BYD' if i//2%2 else 'Kia',colour='red',source='synthetic regression',group='group-'+str(i)))
            labels=p/'labels.jsonl';labels.write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8');rows=load_records(labels);tr,va=split_records(rows)
            self.assertFalse({r['pixel_hash'] for r in tr}&{r['pixel_hash'] for r in va})

    def test_conflicting_labels_on_same_visible_box_are_rejected(self):
        from carvision.data import load_records
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);Image.new('RGB',(24,24),'white').save(p/'a.png');r=dict(image='a.png',box=[0,0,24,24],brand='BYD',colour='white',source='synthetic regression',group='one');labels=p/'labels.jsonl';labels.write_text(json.dumps(r)+'\n'+json.dumps(dict(r,brand='Kia')),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'[Cc]onflicting'):load_records(labels)
