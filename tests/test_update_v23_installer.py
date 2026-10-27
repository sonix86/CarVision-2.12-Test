"""Actual native Windows installation/rollback on byte fixtures; not ML accuracy."""
import hashlib,json,os,shutil,subprocess,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda raw:hashlib.sha256(raw).hexdigest()

@unittest.skipUnless(os.name=='nt','Native Windows installer')
class UpdateInstallerChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='installer23-',dir=ROOT/'outputs')
        self.root=Path(self.temp.name);self.patch=self.root/'update';self.target=self.root/'Проверка папка'
        (self.patch/'payload').mkdir(parents=True);self.target.mkdir()
        shutil.copy2(ROOT/'handoff/APPLY_UPDATE_2_3.ps1',self.patch/'APPLY_UPDATE_2_3.ps1')
        self.entries=[]
        self.add('CarVision.exe',b'new executable',b'original executable')
        self.add('_internal/python312.dll',b'new runtime',b'old runtime')
        self.add('models/vehicle_model.pt',b'new model',b'baseline22',previous=[sha(b'baseline21'),sha(b'baseline22')])
        self.add('models/make_expert.pt',b'new make classifier',None)
        (self.target/'data').mkdir();(self.target/'data/labels.jsonl').write_text('OWNER LABELS',encoding='utf-8')
        self.manifest=dict(exe_sha256=sha(b'new executable'),previous_exe_sha256=[sha(b'original executable')],files=self.entries)
        self.save()
    def tearDown(self):self.temp.cleanup()
    def add(self,path,new,old,previous=None):
        p=self.patch/'payload'/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(new)
        if old is not None:
            p=self.target/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(old)
        self.entries.append(dict(path=path,sha256=sha(new),previous_sha256=previous if previous is not None else [sha(old)] if old is not None else [],require_existing=old is not None))
    def save(self):(self.patch/'UPDATE_MANIFEST.json').write_text(json.dumps(self.manifest),encoding='utf-8')
    def run_installer(self):
        system=next(v for k,v in os.environ.items() if k.lower()=='systemroot')
        return subprocess.run([str(Path(system)/'System32/WindowsPowerShell/v1.0/powershell.exe'),'-NoProfile','-ExecutionPolicy','Bypass','-File',str(self.patch/'APPLY_UPDATE_2_3.ps1'),'-Target',str(self.target)],capture_output=True,encoding='utf-8',errors='replace',timeout=45)
    def test_installs_over22_backs_up_preserves_owner_and_is_idempotent(self):
        p=self.run_installer();self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertEqual((self.target/'CarVision.exe').read_bytes(),b'new executable')
        self.assertEqual((self.target/'data/labels.jsonl').read_text(encoding='utf-8'),'OWNER LABELS')
        backups=list((self.target/'patch-backups').iterdir());self.assertEqual(len(backups),1)
        self.assertEqual((backups[0]/'CarVision.exe').read_bytes(),b'original executable')
        self.assertEqual(self.run_installer().returncode,0)
        self.assertEqual(len(list((self.target/'patch-backups').iterdir())),1)
    def test_corrupt_payload_rejected_before_changes(self):
        (self.patch/'payload/models/make_expert.pt').write_bytes(b'CORRUPT')
        self.assertNotEqual(self.run_installer().returncode,0)
        self.assertEqual((self.target/'CarVision.exe').read_bytes(),b'original executable')
        self.assertFalse((self.target/'patch-backups').exists())
    def test_installs_over_original21_weight_pair(self):
        (self.target/'models/vehicle_model.pt').write_bytes(b'baseline21')
        p=self.run_installer();self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        self.assertEqual((self.target/'models/vehicle_model.pt').read_bytes(),b'new model')
    def test_missing_unchanged_dependency_rejected_before_changes(self):
        self.manifest['required_files']=[dict(path='_internal/torch_core.dll',sha256=sha(b'required library'))];self.save()
        p=self.run_installer();self.assertNotEqual(p.returncode,0)
        self.assertEqual((self.target/'CarVision.exe').read_bytes(),b'original executable')
        self.assertFalse((self.target/'patch-backups').exists())
    def test_modified_owner_model_rejected_before_changes(self):
        (self.target/'models/vehicle_model.pt').write_bytes(b'OWN CUSTOM MODEL')
        self.assertNotEqual(self.run_installer().returncode,0)
        self.assertEqual((self.target/'models/vehicle_model.pt').read_bytes(),b'OWN CUSTOM MODEL')
        self.assertEqual((self.target/'CarVision.exe').read_bytes(),b'original executable')
    def test_copy_failure_restores_exe_runtime_models_and_removes_created_files(self):
        (self.target/'models/obstruction').write_bytes(b'file blocking a directory')
        self.add('models/obstruction/new.pt',b'cannot install',None);self.save()
        p=self.run_installer();self.assertNotEqual(p.returncode,0)
        self.assertEqual((self.target/'CarVision.exe').read_bytes(),b'original executable')
        self.assertEqual((self.target/'_internal/python312.dll').read_bytes(),b'old runtime')
        self.assertEqual((self.target/'models/vehicle_model.pt').read_bytes(),b'baseline22')
        self.assertFalse((self.target/'models/make_expert.pt').exists())
