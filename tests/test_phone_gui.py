"""Exercise the HTTP -> GUI -> worker -> phone response boundary."""
import io
import gc
import json
import tempfile
import time
import tkinter as tk
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch, Mock
from PIL import Image
from carvision import gui
from carvision.phone_input import PhoneReceiver


class PhoneGuiChecks(unittest.TestCase):
    def setUp(self):
        gc.collect()
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.patches=[patch.object(gui,'ROOT',self.folder),patch('carvision.phone_input.PhoneReceiver',side_effect=lambda *a,**kw:PhoneReceiver(*a,host='127.0.0.1',port=0))]
        for p in self.patches:p.start()
        self.root=tk.Tk();self.app=gui.App(self.root)
        (self.folder/'models').mkdir();(self.folder/'models/detector.pt').touch()
        self.app.open_phone();self.root.update();self.receiver=self.app.phone_receiver
        self.base=f'http://127.0.0.1:{self.receiver.port}/{self.receiver.token}/'
        buf=io.BytesIO();Image.new('RGB',(120,100),'blue').save(buf,format='PNG');self.raw=buf.getvalue()
    def tearDown(self):
        self.app.close()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()
    def post(self):
        req=urllib.request.Request(self.base+'upload',data=self.raw)
        with urllib.request.urlopen(req,timeout=5) as r:return json.load(r)['id']
    def wait_job(self,ident):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            self.root.update()
            if self.receiver.jobs[ident]['status']!='processing':return self.receiver.jobs[ident]
            time.sleep(.02)
        self.fail('Phone job did not finish')
    def test_upload_triggers_gui_analysis_and_returns_same_unknown(self):
        result=dict(cars=[dict(id=1,brand='unknown',colour='blue',box=[10,10,80,80],model='unknown')],seconds=.2)
        pipeline=Mock();pipeline.predict.return_value=result
        with patch.object(gui,'Pipeline',return_value=pipeline):
            ident=self.post();job=self.wait_job(ident)
            self.assertEqual(job['status'],'done');self.assertEqual(job['cars'][0]['brand'],'unknown')
            self.assertEqual(pipeline.predict.call_count,1)
            self.assertEqual(pipeline.predict.call_args.kwargs,dict(region=None,single=False))
            self.assertTrue(list((self.folder/'outputs/phone_results'/ident).glob('*.json')))
            deadline=time.monotonic()+2
            while self.app.busy and time.monotonic()<deadline:self.root.update();time.sleep(.02)
            self.assertIs(self.app.result,result)
        self.app.stop_phone();self.assertTrue(self.receiver.stopped)
    def test_upload_while_gui_busy_fails_without_changing_current_photo(self):
        self.app.busy=True
        ident=self.post();job=self.wait_job(ident)
        self.assertEqual(job['status'],'error');self.assertIsNone(self.app.path)
