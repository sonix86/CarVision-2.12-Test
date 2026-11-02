import io
import gc
import json
import tempfile
import unittest
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image
from carvision.phone_input import PhoneReceiver, MAX_BYTES, PAGE


class PhoneChecks(unittest.TestCase):
    def setUp(self):
        gc.collect()  # Dispose earlier destroyed Tk fixtures on the main thread.
        self.temp=tempfile.TemporaryDirectory()
        self.submitted=[]
        self.receiver=PhoneReceiver(self.temp.name,lambda *x:self.submitted.append(x),host='127.0.0.1',port=0)
        self.base=f'http://127.0.0.1:{self.receiver.port}/{self.receiver.token}/'
        image=Image.new('RGB',(100,80),'blue');buffer=io.BytesIO();image.save(buffer,format='PNG');self.raw=buffer.getvalue()
    def tearDown(self):
        self.receiver.stop();self.temp.cleanup()
    def request(self,path,data=None,headers=None):
        req=urllib.request.Request(self.base+path,data=data,headers=headers or {})
        try:
            with urllib.request.urlopen(req,timeout=5) as r:return r.status,r.read(),r.headers
        except urllib.error.HTTPError as r:return r.code,r.read(),r.headers
    def test_page_has_no_external_dependencies_and_camera_fallback(self):
        status,body,headers=self.request('')
        self.assertEqual(status,200);self.assertEqual(headers['Cache-Control'],'no-store')
        self.assertIn(b'capture="environment"',body);self.assertIn(b'getUserMedia',body)
        self.assertNotIn('https://',PAGE);self.assertIn('isSecureContext',PAGE)
    def test_uploaded_image_auto_submits_and_same_result_is_returned(self):
        status,raw,_=self.request('upload',self.raw)
        self.assertEqual(status,202);ident=json.loads(raw)['id']
        self.assertEqual(self.submitted[0][0],ident)
        path=Path(self.submitted[0][1]);self.assertTrue(path.exists())
        with Image.open(path) as im:self.assertEqual(im.format,'JPEG');self.assertEqual(im.size,(100,80))
        preview=path.parent/'output.jpg';Image.new('RGB',(100,100)).save(preview)
        self.receiver.complete(ident,dict(cars=[dict(id=1,brand='unknown',colour='grey',model='unknown')],seconds=2),preview)
        status,raw,_=self.request('jobs/'+ident);result=json.loads(raw)
        self.assertEqual(result['cars'][0]['brand'],'unknown');self.assertEqual(result['status'],'done')
        self.assertNotIn('path',result);self.assertNotIn('preview',result)
        self.assertEqual(self.request(result['preview_url'])[1],preview.read_bytes())
    def test_second_upload_rejected_until_previous_finishes(self):
        _,raw,_=self.request('upload',self.raw);ident=json.loads(raw)['id']
        self.assertEqual(self.request('upload',self.raw)[0],409)
        self.receiver.fail(ident,'busy')
        self.assertEqual(self.request('upload',self.raw)[0],202)
    def test_malformed_image_does_not_block_next_upload(self):
        self.assertEqual(self.request('upload',b'not a photo')[0],400)
        self.assertIsNone(self.receiver.active)
        self.assertEqual(self.request('upload',self.raw)[0],202)
    def test_cross_origin_and_missing_token_rejected(self):
        self.assertEqual(self.request('upload',self.raw,{'Origin':'https://evil.example'})[0],403)
        old=self.base;self.base=old.replace(self.receiver.token,'wrong')
        self.assertEqual(self.request('')[0],404)
        self.assertEqual(self.request('upload',self.raw)[0],404)
        self.base=old
    def test_unknown_routes_and_traversal_do_not_serve_disk(self):
        self.assertEqual(self.request('../models/classifier.pt')[0],404)
        self.assertEqual(self.request('not-upload',self.raw)[0],404)
        self.assertEqual(self.request('jobs/no-such-job')[0],404)
    def test_size_limit_checked_before_reading_body(self):
        self.assertEqual(self.request('upload',b'x',{'Content-Length':str(MAX_BYTES+1)})[0],413)
    def test_orientation_and_resize(self):
        im=Image.new('RGB',(3000,80));exif=Image.Exif();exif[274]=6
        buffer=io.BytesIO();im.save(buffer,format='JPEG',exif=exif)
        path=self.receiver.normalise(buffer.getvalue(),'test')
        with Image.open(path) as ready:self.assertEqual(ready.size,(68,2560));self.assertIsNone(ready.getexif().get(274))
    def test_history_is_bounded_and_stop_cancels(self):
        for _ in range(25):
            ident=self.receiver.reserve();self.receiver.fail(ident,'test')
        self.assertEqual(len(self.receiver.jobs),20)
        ident=self.receiver.reserve();self.receiver.stop()
        self.assertIsNone(self.receiver.reserve());self.assertEqual(self.receiver.jobs[ident]['status'],'error')


if __name__=='__main__':unittest.main()
