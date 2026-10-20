"""Own GUI behaviour with isolated files and mocked OS/dialog boundaries."""
import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace
from PIL import Image
from carvision import gui

class GuiChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.patch=patch.object(gui,'ROOT',self.folder);self.patch.start()
        self.root=tk.Tk();self.app=gui.App(self.root);self.root.update()
        self.photo=self.folder/'photo.png';Image.new('RGB',(800,500),'blue').save(self.photo)
    def tearDown(self):self.app.close();self.patch.stop();self.temp.cleanup()
    def test_all_views_and_busy_states(self):
        for view in ['infer','labels','training','help']:self.app.show(view);self.root.update();self.assertTrue(self.app.views[view].winfo_ismapped())
        self.root.geometry('1120x760');self.app.show('infer');self.root.update()
        for key in ['copy','export','output']:
            button=self.app.buttons[key];self.assertLessEqual(button.winfo_y()+button.winfo_height(),button.master.winfo_height())
        self.app.set_busy(True,'test');self.assertTrue(self.app.buttons['open'].instate(['disabled']));self.app.set_busy(False,'done')
    def test_analysis_region_coordinates_survive_resize_and_clear_on_next_photo(self):
        app=self.app;app.load_photo(self.photo);self.root.update();app.redraw();app.scope.set(gui.SCOPES[2]);app.scope_changed();sx,sy,ox,oy=app.transforms[app.preview]
        app.start_region(SimpleNamespace(x=ox+400*sx,y=oy+350*sy));app.end_region(SimpleNamespace(x=ox+100*sx,y=oy+80*sy));self.assertEqual(app.region,[100,80,400,350])
        self.root.geometry('1120x760');self.root.update();app.redraw();self.assertEqual(app.region,[100,80,400,350]);self.assertTrue(app.preview.find_withtag('region'))
        app.clear_region();self.assertIsNone(app.region);self.assertEqual(app.scope.get(),gui.SCOPES[0]);app.region=[20,20,200,200];app.load_photo(self.photo);self.assertIsNone(app.region)
    def test_small_analysis_region_is_rejected(self):
        app=self.app;app.load_photo(self.photo);self.root.update();app.redraw();app.scope.set(gui.SCOPES[2]);sx,sy,ox,oy=app.transforms[app.preview]
        app.start_region(SimpleNamespace(x=ox+20*sx,y=oy+20*sy));app.end_region(SimpleNamespace(x=ox+25*sx,y=oy+25*sy));self.assertIsNone(app.region)
    def test_annotation_resizes_save_undo_and_group_reset(self):
        app=self.app;app.load_label_photo(self.photo);self.root.update();app.redraw();first_group=app.group.get()
        sx,sy,ox,oy=app.transforms[app.canvas]
        app.start_box(SimpleNamespace(x=ox+100*sx,y=oy+80*sy));app.end_box(SimpleNamespace(x=ox+500*sx,y=oy+350*sy));self.assertEqual(app.box,[100,80,500,350])
        app.save_label();self.assertEqual(len(app.all_labels()),1);self.assertEqual(app.all_labels()[0]['box'],[100,80,500,350])
        self.root.geometry('1160x780');self.root.update();app.redraw();sx,sy,ox,oy=app.transforms[app.canvas]
        app.start_box(SimpleNamespace(x=ox+20*sx,y=oy+20*sy));app.end_box(SimpleNamespace(x=ox+180*sx,y=oy+200*sy));self.assertEqual(app.box,[20,20,180,200])
        app.undo_label();self.assertEqual(app.all_labels(),[])
        other=self.folder/'other.png';Image.new('RGB',(800,500),'green').save(other);app.load_label_photo(other);self.assertNotEqual(app.group.get(),first_group)
    def test_inference_copy_export_and_file_buttons(self):
        app=self.app;app.load_photo(self.photo);self.root.update()
        result=dict(cars=[dict(id=1,box=[50,50,350,300],brand='BYD',colour='blue',brand_confidence=.8,detection_confidence=.9,colour_source='trained')],seconds=.2,detector_views=3,classifier_trained=True)
        app.events.put(('prediction',result));app.poll();app.copy();self.assertEqual(self.root.clipboard_get(),'BYD blue')
        with patch.object(gui.filedialog,'askdirectory',return_value=str(self.folder/'export')):app.export()
        self.assertEqual(len(list((self.folder/'export').glob('*.json'))),1)
        (self.folder/'samples').mkdir(exist_ok=True);pdf=self.folder/'samples/Tashkent_Photo_Examples.pdf';pdf.write_bytes(b'%PDF-test-placeholder')
        with patch.object(gui.os,'startfile',create=True) as opened:app.open_pdf();app.open_folder(self.folder/'outputs');self.assertEqual(opened.call_count,2)
    def test_sample_and_own_open_buttons(self):
        with patch.object(gui.filedialog,'askopenfilename',return_value=str(self.photo)) as dialog:
            self.app.buttons['samples'].invoke();self.assertEqual(self.app.path,str(self.photo));self.assertIn('samples',dialog.call_args.kwargs['initialdir'])
            self.app.buttons['label_open'].invoke();self.assertIsNotNone(self.app.label_image)
    def test_training_audit_evaluate_commands(self):
        with patch.object(self.app,'command') as command:
            self.app.buttons['audit'].invoke();self.assertEqual(command.call_args.args[0][0],'audit')
            self.app.buttons['evaluate'].invoke();self.assertIn(str(self.folder/'models/evaluation'),command.call_args.args[0])
            self.app.buttons['train'].invoke();self.assertIn('--detector-epochs',command.call_args.args[0])
    def test_stop_button_terminates_active_child(self):
        process=Mock();process.poll.return_value=None;self.app.process=process;self.app.set_busy(True,'running');self.app.buttons['cancel'].invoke();process.terminate.assert_called_once();self.assertTrue(self.app.cancelled);self.app.process=None;self.app.set_busy(False,'stopped')

if __name__=='__main__':unittest.main()
