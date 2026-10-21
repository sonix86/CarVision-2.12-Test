import unittest
import numpy as np
import torch
from PIL import Image
from carvision.segmentation import encode_mask,decode_mask,contains_point,isolate,InstanceSegmenter

class InstanceChecks(unittest.TestCase):
    def test_roundtrip_holes_empty_and_solid(self):
        for mask in [np.zeros((20,30),bool),np.ones((20,30),bool),np.eye(20,dtype=bool)]:
            self.assertTrue(np.array_equal(decode_mask(encode_mask(mask)),mask))
    def test_scaled_hit_testing_uses_shape_not_rectangle(self):
        mask=np.zeros((20,30),bool);mask[2:12,3:14]=True;encoded=encode_mask(mask,(100,200));encoded['scale']=[2,3]
        self.assertTrue(contains_point(encoded,110,215));self.assertFalse(contains_point(encoded,145,215));self.assertFalse(contains_point(encoded,99,200))
    def test_background_pixels_do_not_enter_isolated_evidence(self):
        mask=np.zeros((50,80),bool);mask[10:40,20:60]=True
        a=np.zeros((50,80,3),np.uint8);a[:]=[0,255,0];a[mask]=[240,240,240]
        b=a.copy();b[~mask]=[255,0,0]
        self.assertTrue(np.array_equal(isolate(Image.fromarray(a),mask),isolate(Image.fromarray(b),mask)))
    def test_mask_merge_preserves_overlapping_boxes_and_excludes_duplicate_pixels(self):
        segmenter=object.__new__(InstanceSegmenter);mask=torch.zeros(3,1,60,80)
        mask[0,0,10:50,10:40]=.95;mask[1,0,20:55,35:70]=.95;mask[2]=mask[0]
        segmenter.raw=lambda image:dict(scores=torch.tensor([.99,.98,.7]),labels=torch.tensor([3,3,3]),boxes=torch.tensor([[10.,10,70,55],[10.,10,70,55],[10.,10,70,55]]),masks=mask)
        rows,_=segmenter.instances(Image.new('RGB',(80,60)),.5)
        self.assertEqual(len(rows),2);self.assertFalse(np.any(rows[0]['mask']&rows[1]['mask']))
    def test_corrupt_mask_is_rejected(self):
        data=encode_mask(np.ones((5,5),bool));data['counts']=[-1,26]
        with self.assertRaises(ValueError):decode_mask(data)

    def test_one_instance_proposal_can_bridge_occluded_parts(self):
        from carvision.segmentation import resolve_instances
        a=np.zeros((80,160),bool);a[20:65,10:50]=True
        b=np.zeros_like(a);b[20:65,80:140]=True
        bridge=a|b
        records=[dict(mask=m,probability=m.astype('float32')*.95,score=s,label=3) for m,s in [(a,.95),(b,.94),(bridge,.8)]]
        rows=resolve_instances(records)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['proposals'],3)
        self.assertTrue(np.array_equal(rows[0]['mask'],bridge));self.assertFalse(rows[0]['mask'][40,65])

    def test_close_disconnected_neighbours_have_no_bridge_evidence(self):
        from carvision.segmentation import resolve_instances
        a=np.zeros((80,160),bool);a[20:65,10:70]=True
        b=np.zeros_like(a);b[20:65,71:140]=True
        rows=resolve_instances([dict(mask=m,probability=m.astype('float32')*.95,score=.95,label=3) for m in [a,b]])
        self.assertEqual(len(rows),2)

    def test_more_confident_person_evidence_excludes_occluder_pixels(self):
        segmenter=object.__new__(InstanceSegmenter);masks=torch.zeros(2,1,80,100)
        masks[0,0,10:70,10:90]=.85;masks[1,0,20:65,45:60]=.99
        segmenter.raw=lambda image:dict(scores=torch.tensor([.9,.99]),labels=torch.tensor([3,1]),boxes=torch.tensor([[10.,10,90,70],[45.,20,60,65]]),masks=masks)
        rows,_=segmenter.instances(Image.new('RGB',(100,80)))
        self.assertEqual(len(rows),1);self.assertFalse(rows[0]['mask'][40,50]);self.assertTrue(rows[0]['mask'][40,30])

    def test_user_union_reclassifies_visible_pixels_and_preserves_the_gap(self):
        from types import SimpleNamespace
        from carvision.scene import merge_prediction
        image=Image.new('RGB',(320,160),'green');image.paste('white',(20,30,100,130));image.paste('white',(180,30,260,130))
        rows=[dict(id=i,brand='BMW',colour='green',box=box,detection_confidence=.9,segmentation=encode_mask(np.ones((100,80),bool),box[:2])) for i,box in enumerate([[20,30,100,130],[180,30,260,130]],1)]
        prediction=dict(image='synthetic.png',cars=rows,seconds=0)
        p=SimpleNamespace(detection=.5,mode='fast',classifier=None,vehicle_classifier=None,colour_classifier=None,masked_colour=None,colour_reliability_threshold=.5,brands=[])
        result=merge_prediction(p,image,prediction,[1,2]);row=result['cars'][0]
        self.assertEqual(len(result['cars']),1);self.assertEqual(row['colour'],'white');self.assertEqual(row['brand'],'unknown')
        self.assertTrue(contains_point(row['segmentation'],50,80));self.assertTrue(contains_point(row['segmentation'],220,80));self.assertFalse(contains_point(row['segmentation'],140,80))
        self.assertEqual(len(prediction['cars']),2)

    def test_subthreshold_neighbour_cannot_steal_valid_foreground_pixels(self):
        segmenter=object.__new__(InstanceSegmenter);masks=torch.zeros(2,1,40,80)
        masks[0,0,5:35,5:30]=.9;masks[0,0,5:35,40:70]=.49
        masks[1,0,5:35,40:70]=.6
        segmenter.raw=lambda image:dict(scores=torch.tensor([.99,.55]),labels=torch.tensor([3,3]),boxes=torch.tensor([[5.,5,30,35],[40.,5,70,35]]),masks=masks)
        rows,_=segmenter.instances(Image.new('RGB',(80,40)),.5)
        self.assertEqual(len(rows),2)
        self.assertEqual(sum(int(row['mask'].sum()) for row in rows),1650)
        self.assertTrue(any(row['mask'][20,55] for row in rows))

    def test_last_image_cache_is_pixel_based_and_does_not_share_mutable_masks(self):
        from unittest.mock import Mock
        segmenter=object.__new__(InstanceSegmenter);mask=torch.ones(1,1,40,80)*.9
        segmenter.raw=Mock(return_value=dict(scores=torch.tensor([.99]),labels=torch.tensor([3]),boxes=torch.tensor([[0.,0,80,40]]),masks=mask))
        first,_=segmenter.instances(Image.new('RGB',(80,40),'white'));first[0]['mask'][:]=False
        second,_=segmenter.instances(Image.new('RGB',(80,40),'white'));self.assertTrue(second[0]['mask'].all());self.assertEqual(segmenter.raw.call_count,1)
        segmenter.instances(Image.new('RGB',(80,40),'blue'));self.assertEqual(segmenter.raw.call_count,2)
        segmenter.instances(Image.new('RGB',(80,40),'blue'),threshold=.6);self.assertEqual(segmenter.raw.call_count,3)

    def test_single_car_prompt_preserves_visible_mask_and_excludes_foreground_car(self):
        from types import SimpleNamespace
        from carvision.scene import predict_instances
        import time
        green=np.zeros((400,600),bool);green[100:370,100:500]=True
        white=np.zeros((400,600),bool);white[40:100,300:580]=True;white[100:280,500:580]=True
        rgb=np.full((400,600,3),240,dtype=np.uint8);rgb[green]=[20,190,40]
        class Segmenter:
            def instances(self,*args):return [dict(mask=green,score=.99),dict(mask=white,score=.98)],1
        p=SimpleNamespace(segmenter=Segmenter(),detection=.5,mode='fast',classifier=None,vehicle_classifier=None,colour_classifier=None,masked_colour=None,colour_reliability_threshold=.5,brands=[])
        result=predict_instances(p,Image.fromarray(rgb),'synthetic.png',[300,40,580,280],True,time.perf_counter())
        self.assertEqual(len(result['cars']),1);row=result['cars'][0];self.assertEqual(row['colour'],'white');self.assertFalse(contains_point(row['segmentation'],400,180));self.assertTrue(contains_point(row['segmentation'],550,180))
