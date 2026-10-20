"""Engineering checks on synthetic images, never presented as model accuracy."""
import json
import tempfile
import unittest
import socket
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import torch
from carvision.core import Pipeline, heuristic_colour, Classifier, ROOT,detection_tiles,restore_tile_boxes,save_prediction,merge_vehicle_boxes
from carvision.data import load_records,split_records
from carvision.scoring import photo_score

class Checks(unittest.TestCase):
    def test_single_region_bypasses_detector_and_keeps_original_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'photo.png';Image.new('RGB',(600,400),'white').save(path)
            p=object.__new__(Pipeline);p.mode='detail';p.classifier=None;p.colour_mode='heuristic';p.detection=.5
            p.detector=lambda x: self.fail('Single-car selection must not depend on a detector finding it')
            result=p.predict(path,region=[100,80,420,350],single=True)
            self.assertEqual(result['cars'][0]['box'],[100,80,420,350]);self.assertEqual(result['detector_views'],0);self.assertIsNone(result['cars'][0]['detection_confidence'])
            self.assertEqual(result['region'],[100,80,420,350])
    def test_region_detection_uses_crop_and_restores_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'photo.png';Image.new('RGB',(600,400),'white').save(path)
            p=object.__new__(Pipeline);p.mode='fast';p.classifier=None;p.colour_mode='heuristic';p.detection=.5
            def detect(images):
                self.assertEqual(tuple(images[0].shape), (3,270,320))
                return [dict(boxes=torch.tensor([[10.,20.,200.,220.]]),scores=torch.tensor([.9]),labels=torch.tensor([1]))]
            p.detector=detect;result=p.predict(path,region=[100,80,420,350]);self.assertEqual(result['cars'][0]['box'],[110,100,300,300])
            for region in [[0,0,8,200],[200,0,100,100],[0,0,float('nan'),100]]:
                with self.assertRaises(ValueError):p.predict(path,region=region)
    def test_cross_view_fragments_removed_but_nested_full_view_cars_preserved(self):
        boxes=torch.tensor([[10.,10,300,250],[200.,20,280,100],[40.,40,140,150],[200.,20,280,100]])
        scores=torch.tensor([.99,.94,.96,.92]);keep=merge_vehicle_boxes(boxes,scores,2)
        self.assertEqual(keep.tolist(),[0,1])
        # Two genuinely overlapping complete cars are not removed by containment.
        keep=merge_vehicle_boxes(boxes[:2],scores[:2],2);self.assertEqual(keep.tolist(),[0,1])
    def test_dense_numeric_markers_stay_inside_canvas_and_do_not_overlap(self):
        from carvision.overlays import marker_positions
        positions=marker_positions([[100,100,200,180]]*18,640,480)
        for i,(x,y,x2,y2) in enumerate(positions):
            self.assertTrue(0<=x<x2<=640 and 0<=y<y2<=480)
            for a,b,a2,b2 in positions[:i]:self.assertFalse(x<a2 and x2>a and y<b2 and y2>b)
    def test_export_keeps_names_in_separate_legend(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'photo.png';Image.new('RGB',(600,400),'white').save(path)
            cars=[dict(id=i+1,box=[10,10,400,300],brand='Chevrolet',colour='white') for i in range(10)]
            output=save_prediction(path,dict(cars=cars),Path(temp)/'out')
            with Image.open(output) as image:self.assertGreater(image.height,400);self.assertEqual(image.width,600)
    def test_assignment_scoring(self):
        self.assertEqual(photo_score([('Chevrolet','white')],[('Chevrolet','white')]),5)
        self.assertEqual(photo_score([('Chevrolet','white')],[('Chevrolet','black')]),3)
        self.assertEqual(photo_score([('Chevrolet','white')],[('Kia','white')]),2)
        self.assertEqual(photo_score([('Chevrolet','white')],[('Chevrolet','white'),('Kia','grey')]),4)
        self.assertEqual(photo_score([('Chevrolet','white'),('Chevrolet','white'),('BYD','blue')],[('Chevrolet','white'),('Chevrolet','white'),('BYD','black')]),4.3)
        self.assertEqual(photo_score([('BYD','blue'),('Kia','red')],[('Kia','red'),('BYD','blue')]),5)
        self.assertEqual(photo_score([('Mercedes-Benz','white'),('Volkswagen','grey'),('Li Auto','black')],[('Mercedes','white'),('VW','grey'),('Lixiang','black')]),5)

    def test_brand_and_colour_are_inferred_without_manual_brand_input(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'new-photo.jpg';Image.new('RGB',(400,300),'white').save(path)
            pipeline=object.__new__(Pipeline);pipeline.mode='fast';pipeline.detection=.5;pipeline.brand_threshold=.65;pipeline.colour_mode='trained';pipeline.brands=['Kia','BYD']
            pipeline.detector=lambda images:[dict(boxes=torch.tensor([[20.,30,320,260]]),scores=torch.tensor([.95]),labels=torch.tensor([1]))]
            pipeline.classifier=lambda x:(torch.tensor([[0.,10.]]),torch.tensor([[0.,0.,0.,0.,10.,0.,0.,0.]]))
            result=pipeline.predict(path)
            self.assertEqual([(r['brand'],r['colour']) for r in result['cars']],[('BYD','blue')]);self.assertEqual(result['cars'][0]['colour_source'],'trained')

    def test_colour_palette(self):
        for name,rgb in [('white',(245,245,245)),('black',(10,10,10)),('grey',(110,110,110)),('red',(220,20,20)),('blue',(20,40,230)),('green',(20,160,30)),('brown',(120,70,30)),('yellow',(240,200,20))]:
            self.assertEqual(heuristic_colour(Image.new('RGB',(100,100),rgb))[0],name)

    def test_separate_trained_colour_head_is_used_for_public_prototype(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'new-photo.png';Image.new('RGB',(320,240),'white').save(path)
            p=object.__new__(Pipeline);p.mode='fast';p.detection=.5;p.brand_threshold=.5;p.colour_mode='heuristic';p.brands=['Chevrolet'];p.classifier=lambda x:(torch.tensor([[10.]]),torch.zeros((1,8)))
            p.detector=lambda images:[dict(boxes=torch.tensor([[20.,20,280,210]]),scores=torch.tensor([.95]),labels=torch.tensor([1]))]
            p.colour_classifier=SimpleNamespace(predict=lambda image:('green',.93))
            result=p.predict(path);self.assertEqual([(r['brand'],r['colour']) for r in result['cars']],[('Chevrolet','green')]);self.assertEqual(result['cars'][0]['colour_source'],'project-trained VCoR colour head')

    def test_colour_features_are_finite_for_greyscale_and_small_crops(self):
        from carvision.colour_model import colour_statistics,ColourClassifier
        for image in [Image.new('L',(16,16),100),Image.new('RGB',(400,200),'red')]:
            features=colour_statistics(image);self.assertEqual(features.shape,(261,));self.assertTrue(torch.isfinite(features).all())
        with self.assertRaisesRegex(ValueError,'project-trained'):ColourClassifier(dict(origin='ready-made-colour-classifier'))
    def test_paint_evidence_retains_joint_features_without_a_pretrained_classifier(self):
        from carvision.colour_model import paint_statistics,colour_head,ColourClassifier
        from carvision.core import COLOURS
        x=paint_statistics(Image.new('RGB',(25,20),'black'));self.assertEqual(x.shape,(929,));self.assertTrue(torch.isfinite(x).all())
        checkpoint=dict(origin='team-trained-colour-statistics',classes=COLOURS,feature_mode='paint_statistics',mean=torch.zeros(929),std=torch.ones(929),head=colour_head(929).state_dict())
        model=ColourClassifier(checkpoint);self.assertIsNone(model.features);colour,probability=model.predict(Image.new('RGB',(50,70),'blue'));self.assertIn(colour,COLOURS);self.assertTrue(0<=probability<=1)
        mixture=ColourClassifier(dict(origin='team-trained-colour-ensemble',architecture='colour_evidence_ensemble_v3',classes=COLOURS,components=[checkpoint],mixture=[dict(component=0,region=None,weight=.5),dict(component=0,region=[.08,.4,.92,.9],weight=.5)]))
        probabilities=mixture.probabilities(Image.new('RGB',(120,90),'grey'));self.assertAlmostEqual(float(probabilities.sum()),1.,places=5);self.assertTrue(torch.isfinite(probabilities).all())

    def test_inference_has_no_network_and_no_untrained_brands(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'blank.png';Image.new('RGB',(640,480),'white').save(path)
            pipeline=object.__new__(Pipeline);pipeline.mode='balanced';pipeline.detection=.5;pipeline.classifier=None;pipeline.brand_threshold=.65;pipeline.colour_mode='trained'
            pipeline.detector=lambda images:[dict(boxes=torch.empty((0,4)),scores=torch.empty(0),labels=torch.empty(0,dtype=torch.long)) for _ in images]
            with patch.object(socket.socket,'connect',side_effect=AssertionError('Network forbidden')):result=pipeline.predict(path)
            self.assertFalse(result['classifier_trained']);self.assertEqual(result['cars'],[])

    def test_team_detector_required(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):Pipeline(temp)

    def test_tile_mapping_and_seam_rejection(self):
        d=dict(boxes=torch.tensor([[30.,40,160,140],[0.,50,60,110]]),scores=torch.tensor([.8,.9]),labels=torch.tensor([1,1]))
        boxes,scores=restore_tile_boxes(d,(500,0,1268,768),1600,1000,.5)
        self.assertEqual(boxes.tolist(),[[530,40,660,140]])

    def test_generic_coco_car_and_truck_include_vans_but_exclude_bus(self):
        detection=dict(boxes=torch.tensor([[10.,10,80,80],[90.,10,160,80],[170.,10,240,80]]),scores=torch.tensor([.9,.8,.95]),labels=torch.tensor([3,8,6]))
        boxes,scores=restore_tile_boxes(detection,(0,0,300,200),300,200,.5,(3,8))
        self.assertEqual(boxes.tolist(),[[10,10,80,80],[90,10,160,80]])

    def test_generic_class_mapping_is_used_in_photo_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'van.jpg';Image.new('RGB',(320,240),'white').save(path)
            p=object.__new__(Pipeline);p.mode='fast';p.detection=.5;p.brand_threshold=.65;p.classifier=None;p.colour_mode='heuristic';p.car_label=(3,8)
            p.detector=lambda images:[dict(boxes=torch.tensor([[20.,20,250,200],[10.,10,300,230]]),scores=torch.tensor([.8,.9]),labels=torch.tensor([8,6]))]
            result=p.predict(path);self.assertEqual(len(result['cars']),1);self.assertEqual(result['cars'][0]['box'],[20,20,250,200])

    def test_tiled_predictions_deduplicate_and_keep_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'street.png';Image.new('RGB',(1600,900),'white').save(path);tiles=detection_tiles(1600,900);remaining=iter(tiles)
            def detect(images):
                results=[]
                for image in images:
                    x,y,r,b=next(remaining);global_box=[800,300,1100,500]
                    if x<=800 and y<=300 and r>=1100 and b>=500:results.append(dict(boxes=torch.tensor([[800-x,300-y,1100-x,500-y]],dtype=torch.float32),scores=torch.tensor([.85]),labels=torch.tensor([1])))
                    else:results.append(dict(boxes=torch.empty((0,4)),scores=torch.empty(0),labels=torch.empty(0,dtype=torch.long)))
                return results
            p=object.__new__(Pipeline);p.mode='balanced';p.detection=.5;p.brand_threshold=.65;p.classifier=None;p.colour_mode='trained';p.detector=detect
            result=p.predict(path);self.assertEqual(len(result['cars']),1);self.assertEqual(result['cars'][0]['box'],[800,300,1100,500]);self.assertGreater(result['detector_views'],1)

    def test_exports_do_not_overwrite_same_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);photo=root/'same.png';Image.new('RGB',(32,32),'white').save(photo)
            result=dict(cars=[],classifier_trained=False)
            a=save_prediction(photo,result,root/'out');b=save_prediction(photo,result,root/'out')
            self.assertNotEqual(a,b);self.assertTrue(a.exists() and b.exists())

    def test_duplicate_images_stay_together(self):
        rows=[dict(brand='BYD' if i%2 else 'Kia',group=str(i//2),hash=str(i//2),image=str(i)) for i in range(20)]
        tr,va=split_records(rows)
        self.assertFalse({r['hash'] for r in tr}&{r['hash'] for r in va})

    def test_invalid_box_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            d=Path(temp);Image.new('RGB',(10,10)).save(d/'car.png')
            row=dict(image='car.png',box=[0,0,20,20],brand='BYD',colour='white',source='synthetic test',group='one')
            (d/'labels.jsonl').write_text(json.dumps(row),encoding='utf-8')
            with self.assertRaises(ValueError):load_records(d/'labels.jsonl')

if __name__=='__main__':unittest.main()
