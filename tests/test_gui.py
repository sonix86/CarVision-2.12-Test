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
            self.assertLessEqual(button.winfo_x()+button.winfo_width(),button.master.winfo_width())
        self.app.set_busy(True,'test');self.assertTrue(self.app.buttons['open'].instate(['disabled']));self.app.set_busy(False,'done')
    def test_analysis_region_coordinates_survive_resize_and_clear_on_next_photo(self):
        app=self.app;app.load_photo(self.photo);self.root.update();app.redraw();app.scope.set(gui.SCOPES[2]);app.scope_changed();sx,sy,ox,oy=app.transforms[app.preview]
        app.start_region(SimpleNamespace(x=ox+400*sx,y=oy+350*sy));app.end_region(SimpleNamespace(x=ox+100*sx,y=oy+80*sy));self.assertEqual(app.region,[100,80,400,350])
        self.root.geometry('1120x760');self.root.update();app.redraw();self.assertEqual(app.region,[100,80,400,350]);self.assertTrue(app.preview.find_withtag('region'))
        app.clear_region();self.assertIsNone(app.region);self.assertEqual(app.scope.get(),gui.SCOPES[0]);app.region=[20,20,200,200];app.load_photo(self.photo);self.assertIsNone(app.region)
    def test_visible_zone_button_and_hidden_advanced_controls(self):
        app=self.app;app.load_photo(self.photo);self.root.update()
        self.assertFalse(app.advanced_controls.winfo_ismapped());app.buttons['choose_region'].invoke();self.assertEqual(app.scope.get(),gui.SCOPES[2]);self.assertEqual(app.preview.cget('cursor'),'crosshair')
        app.advanced.set(True);app.toggle_advanced();self.root.update();self.assertTrue(app.advanced_controls.winfo_ismapped())
    def test_unknown_manufacturer_still_displays_ranked_models(self):
        text=self.app.result_details(dict(brand='unknown',brand_confidence=.3,colour='white',colour_confidence=.4,colour_reliable=False,model_alternatives=[dict(brand='Chevrolet',model='Spark',confidence=.45),dict(brand='Daewoo',model='Matiz',confidence=.3)]))
        self.assertIn('Spark',text);self.assertIn('Matiz',text);self.assertIn('45%',text);self.assertIn('проверьте',text)
    def test_training_guide_explains_scope_and_current_own_dataset(self):
        self.app.show('training');self.root.update()
        self.assertIn('сама по себе ничего не обучает',self.app.training_guide)
        self.assertIn('не добавляет имена новых моделей',self.app.training_guide)
        self.assertIn('0 машин',self.app.training_data_summary.get())
        self.root.geometry('1120x760');self.root.update();button=self.app.buttons['train'];self.assertLessEqual(button.master.winfo_y()+button.master.winfo_height(),self.app.views['training'].winfo_height())
    def test_full_model_ranking_can_be_opened_for_a_selected_car(self):
        self.app.result=dict(cars=[dict(id=1,brand='unknown',model_ranking=[dict(brand='Chevrolet',model='Cobalt',confidence=.4)])])
        self.app.table.insert('','end',iid='1',values=(1,'Фрагмент','Белый'));self.app.table.selection_set('1');self.app.show_model_ranking()
        windows=[w for w in self.root.winfo_children() if isinstance(w,tk.Toplevel)]
        self.assertEqual(len(windows),1);self.assertIn('Сравнение',windows[0].title());windows[0].destroy()
    def test_evidence_card_resets_when_loading_another_photo(self):
        row=dict(brand='Chevrolet',model='Spark',model_confidence=.8,brand_confidence=.9,colour='white',colour_confidence=.6,colour_reliable=True,model_alternatives=[dict(brand='Chevrolet',model='Spark',confidence=.8)])
        self.app.show_evidence(row);self.assertIn('Spark',self.app.selected_title.get());self.assertAlmostEqual(float(self.app.variant_rows[0][1]['value']),.8)
        self.app.load_photo(self.photo);self.assertEqual(float(self.app.variant_rows[0][1]['value']),0);self.assertEqual(self.app.selected_title.get(),'Выберите машину')
    def test_preview_tint_preserves_holes_in_the_actual_instance_mask(self):
        import numpy as np
        from carvision.segmentation import encode_mask
        mask=np.zeros((100,100),bool);mask[10:90,10:90]=True;mask[40:60,40:60]=False
        row=dict(id=1,box=[10,10,90,90],segmentation=encode_mask(mask),contours=[[[10,10],[90,10],[90,90],[10,90]]])
        self.app.draw(self.app.preview,Image.new('RGB',(100,100),'white'),[row])
        shown=gui.ImageTk.getimage(self.app.preview.photo).convert('RGB')
        self.assertEqual(shown.getpixel((50,50)),(255,255,255));self.assertNotEqual(shown.getpixel((30,30)),(255,255,255))
    def test_small_analysis_region_is_rejected(self):
        app=self.app;app.load_photo(self.photo);self.root.update();app.redraw();app.scope.set(gui.SCOPES[2]);sx,sy,ox,oy=app.transforms[app.preview]
        app.start_region(SimpleNamespace(x=ox+20*sx,y=oy+20*sy));app.end_region(SimpleNamespace(x=ox+25*sx,y=oy+25*sy));self.assertIsNone(app.region)
    def test_annotation_resizes_save_undo_and_group_reset(self):
        app=self.app;app.load_label_photo(self.photo);self.root.update();app.redraw();first_group=app.group.get()
        sx,sy,ox,oy=app.transforms[app.canvas]
        app.start_box(SimpleNamespace(x=ox+100*sx,y=oy+80*sy));app.end_box(SimpleNamespace(x=ox+500*sx,y=oy+350*sy));self.assertEqual(app.box,[100,80,500,350])
        app.save_label();self.assertEqual(len(app.all_labels()),1);self.assertEqual(app.all_labels()[0]['box'],[100,80,500,350])
        app.redraw();marker=app.canvas.find_withtag('car-1')[0];x,y,x2,y2=app.canvas.bbox(marker);app.hover_label(SimpleNamespace(x=(x+x2)/2,y=(y+y2)/2));self.assertIn(app.label_rows[0]['brand'],app.label_status.get())
        self.root.geometry('1160x780');self.root.update();app.redraw();sx,sy,ox,oy=app.transforms[app.canvas]
        app.start_box(SimpleNamespace(x=ox+20*sx,y=oy+20*sy));app.end_box(SimpleNamespace(x=ox+180*sx,y=oy+200*sy));self.assertEqual(app.box,[20,20,180,200])
        app.undo_label();self.assertEqual(app.all_labels(),[])
        other=self.folder/'other.png';Image.new('RGB',(800,500),'green').save(other);app.load_label_photo(other);self.assertNotEqual(app.group.get(),first_group)
    def test_inference_copy_export_and_file_buttons(self):
        app=self.app;app.load_photo(self.photo);self.root.update()
        result=dict(cars=[dict(id=1,box=[50,50,350,300],brand='BYD',model='Song Plus',model_confidence=.85,colour='blue',brand_confidence=.8,detection_confidence=.9,colour_source='trained')],seconds=.2,detector_views=3,classifier_trained=True)
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
            with patch.object(gui.messagebox,'showinfo') as info:
                self.app.buttons['evaluate'].invoke();self.assertTrue(info.called)
            candidate=self.folder/'models/own_training_candidate';candidate.mkdir(parents=True,exist_ok=True);(candidate/'training.json').write_text('{}')
            self.app.buttons['evaluate'].invoke();self.assertIn(str(candidate/'evaluation'),command.call_args.args[0])
            self.app.buttons['train'].invoke();self.assertIn('--detector-epochs',command.call_args.args[0])
            self.assertIn(str(candidate),command.call_args.args[0]);self.assertNotIn(str(self.folder/'models'),command.call_args.args[0])

    def test_merge_controls_disable_when_busy_and_clear_for_a_new_photo(self):
        from carvision.segmentation import encode_mask
        import numpy as np
        app=self.app;app.load_photo(self.photo);app.pipeline=Mock()
        row=dict(brand='unknown',model_candidate='unknown',brand_confidence=.1,colour='blue',box=[10,10,90,90],segmentation=encode_mask(np.ones((80,80),bool),(10,10)))
        result=dict(cars=[dict(row,id=1),dict(row,id=2)],seconds=.1)
        app.events.put(('prediction',result));app.poll();app.refresh_buttons()
        self.assertFalse(app.buttons['merge_parts'].instate(['disabled']))
        app.buttons['merge_parts'].invoke();self.root.update()
        windows=[w for w in self.root.winfo_children() if isinstance(w,tk.Toplevel)]
        self.assertEqual(len(windows),1);windows[0].destroy()
        app.before_merge=result;app.set_busy(True,'busy');self.assertTrue(app.buttons['merge_parts'].instate(['disabled']));self.assertTrue(app.buttons['undo_merge'].instate(['disabled']))
        app.set_busy(False,'ready');app.load_photo(self.photo);self.assertIsNone(app.before_merge);self.assertTrue(app.buttons['undo_merge'].instate(['disabled']))
    def test_stop_button_terminates_active_child(self):
        process=Mock();process.poll.return_value=None;self.app.process=process;self.app.set_busy(True,'running');self.app.buttons['cancel'].invoke();process.terminate.assert_called_once();self.assertTrue(self.app.cancelled);self.app.process=None;self.app.set_busy(False,'stopped')

if __name__=='__main__':unittest.main()
