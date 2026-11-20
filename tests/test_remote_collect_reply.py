"""Collection replies remain bounded without dropping metadata or altering outputs."""
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zlib

from aka.light.execution.remote_pool import COLLECT_METADATA,decode_collect_reply
from aka.light.execution.remote_resilience import Failure


class CollectionReplyTests(unittest.TestCase):
    def test_many_inline_reports_are_replaced_by_exact_metadata_in_one_small_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            agent=Path(directory)/'collector.py'
            agent.write_text('''import base64,hashlib
def rpc(root,operation,value):
 assert operation=='collect'
 files={}
 for i in range(20):
  data=bytes([i])*90000
  files['workspace/report-'+str(i)]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
 return {'status':{'state':'completed','attempt':7},'files':files,'collection_token':'unchanged-proof'}
''')
            process=subprocess.run([sys.executable,'-c',COLLECT_METADATA,str(agent),
                hashlib.sha256(agent.read_bytes()).hexdigest(),directory,'{}'],
                capture_output=True,text=True,check=True)
            self.assertLess(len(process.stdout),90000)
            wire=json.loads(process.stdout.removeprefix('AKA_REMOTE='))
            self.assertEqual(wire['original_inline_chars'],20*120000)
            self.assertLessEqual(wire['kept_inline_chars'],40000)
            value=decode_collect_reply(wire)
            self.assertEqual(value['collection_token'],'unchanged-proof')
            self.assertEqual(value['status'],{'state':'completed','attempt':7})
            for i,entry in enumerate(value['files'].values()):
                self.assertEqual(entry['bytes'],90000)
                self.assertEqual(entry['sha256'],hashlib.sha256(bytes([i])*90000).hexdigest())

    def test_corrupt_or_trailing_compressed_metadata_is_rejected(self):
        for encoded in [b'not-zlib',zlib.compress(b'{}')+b'ignored tail']:
            with self.assertRaises(Failure):
                decode_collect_reply({'pool_collect_zlib':base64.b64encode(encoded).decode()})


if __name__=='__main__':unittest.main()
