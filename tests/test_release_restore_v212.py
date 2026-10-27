import importlib.util,io,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('release_restore',Path(__file__).resolve().parents[1]/'tools/restore_release_v212.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RestoreTests(unittest.TestCase):
 def test_split_reads_and_random_seeks(self):
  with tempfile.TemporaryDirectory() as t:
   ps=[Path(t)/str(i) for i in range(3)]
   for p,b in zip(ps,[b'ab',b'cdef',b'g']):p.write_bytes(b)
   r=m.SplitReader(ps)
   try:
    self.assertEqual(r.read(4),b'abcd');self.assertEqual(r.seek(-2,2),5);self.assertEqual(r.read(),b'fg');r.seek(1);self.assertEqual(r.read(5),b'bcdef');r.seek(0);self.assertEqual(r.read(),b'abcdefg')
   finally:r.close()
 def test_traversal_is_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   with self.assertRaises(ValueError):m.safe_target(Path(t),'../escape')
   self.assertEqual(m.safe_target(Path(t),'models/x'),(Path(t)/'models/x').resolve())
if __name__=='__main__':unittest.main()
