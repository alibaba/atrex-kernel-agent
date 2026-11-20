"""Optional private OSS data plane; credentials stay in the parent's CLI profile.

Signed URLs are ephemeral arguments, never persisted or included in exceptions.
Objects are backend-owned temporary transfers with a cleanup ledger.
"""
import base64
import configparser
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.parse
import uuid
import zlib

from .remote_resilience import Failure

FETCH='''import base64,hashlib,importlib.util,json,pathlib,sys,urllib.request
try:
 module_path,module_sha,root,sha,size,url=sys.argv[1:];size=int(size)
 p=pathlib.Path(module_path);assert hashlib.sha256(p.read_bytes()).hexdigest()==module_sha
 s=importlib.util.spec_from_file_location('aka_transfer_cache',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 if not url.startswith('https://'):raise ValueError('Object transfer requires HTTPS')
 with urllib.request.urlopen(url,timeout=30) as response:raw=response.read(size+1)
 if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Object transfer checksum mismatch')
 count=(len(raw)+89999)//90000;store=m.Store(root)
 for i,offset in enumerate(range(0,len(raw),90000)):result=store.put_pack(sha,i,count,size,raw[offset:offset+90000])
 print('AKA_REMOTE='+json.dumps(result))
except Exception as error:print('AKA_REMOTE='+json.dumps({'object_failure':type(error).__name__,'http_status':getattr(error,'code',None)}))
'''

EXPORT='''import hashlib,http.client,json,pathlib,sys,urllib.parse
try:
 root,name,sha,size,url=sys.argv[1:];size=int(size);root=pathlib.Path(root).resolve();p=(root/name).resolve()
 if not p.is_relative_to(root):raise ValueError('Artifact escapes attempt')
 raw=p.read_bytes()
 if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('Artifact changed')
 target=urllib.parse.urlsplit(url)
 if target.scheme!='https':raise ValueError('Object transfer requires HTTPS')
 connection=http.client.HTTPSConnection(target.hostname,target.port,timeout=30)
 connection.request('PUT',target.path+'?'+target.query,raw,{'Content-Length':str(len(raw))})
 response=connection.getresponse();response.read()
 if response.status//100!=2:raise ValueError('Object upload rejected')
 connection.close();print('AKA_REMOTE='+json.dumps({'bytes':len(raw),'sha256':sha}))
except Exception as error:print('AKA_REMOTE='+json.dumps({'object_failure':type(error).__name__,'http_status':getattr(error,'code',None)}))
'''


class Channel:
    def __init__(self,prefix,state):
        if not re.fullmatch(r'oss://[a-z0-9][a-z0-9.-]+/[A-Za-z0-9_./-]+',prefix) or '..' in prefix.split('/'):
            raise ValueError('A private OSS transfer prefix is required')
        self.prefix=prefix.rstrip('/')+'/aka-transfer-v2/'
        self.state=Path(state);self.state.mkdir(parents=True,exist_ok=True)
        self.cli=shutil.which('ossutil')
        self.profile=Path(os.environ.get('AKA_OSSUTIL_CONFIG_FILE',str(Path.home()/'.ossutilconfig')))
        if not self.cli:raise Failure('object_transport_cli_unavailable')
        self.gc()

    def call(self,args):
        for attempt in range(3):
            try:
                p=subprocess.run([self.cli,*args,'-c',str(self.profile)],capture_output=True,text=True,timeout=180)
                if p.returncode==0:return p.stdout
            except (OSError,subprocess.TimeoutExpired):pass
            if attempt<2:time.sleep(attempt+1)
        raise Failure('object_transport_unavailable',transient=True)

    def sign(self,key,method='GET'):
        if method=='PUT':
            # ossutil 1.x signs GET only. The standard V1 PUT signature needs
            # no Pod credentials and sends neither Content-Type nor MD5.
            config=configparser.ConfigParser();config.read(self.profile)
            try:credentials=config['Credentials'];ak=credentials['accesskeyid'];sk=credentials['accesskeysecret']
            except KeyError:raise Failure('object_put_profile_unavailable') from None
            if credentials.get('ststoken'):raise Failure('object_put_profile_requires_sdk')
            endpoint=urllib.parse.urlsplit(credentials['endpoint']);target=urllib.parse.urlsplit(key)
            if endpoint.scheme!='https':raise Failure('object_transport_requires_https')
            expires=str(int(time.time())+3600)
            canonical='PUT\n\n\n'+expires+'\n/'+target.netloc+target.path
            signature=base64.b64encode(hmac.new(sk.encode(),canonical.encode(),hashlib.sha1).digest()).decode()
            query=urllib.parse.urlencode({'OSSAccessKeyId':ak,'Expires':expires,'Signature':signature})
            return 'https://'+target.netloc+'.'+endpoint.netloc+urllib.parse.quote(target.path,safe='/')+'?'+query
        if method!='GET':raise ValueError('Unsupported object signature method')
        text=self.call(['sign',key,'--timeout','3600'])
        urls=re.findall(r'https://[^\s\x1b]+',text)
        if not urls:raise Failure('object_signature_unavailable',transient=True)
        return urls[-1]

    def reserve(self,sha):
        name=uuid.uuid4().hex;key=self.prefix+name+'/'+sha
        path=self.state/(name+'.json')
        from .remote_cache import atomic,document
        atomic(path,document({'key':key,'created':time.time()}))
        return key,path

    def remove(self,key,ledger):
        try:
            self.call(['rm',key,'-f']);ledger.unlink(missing_ok=True)
        except Failure:pass  # Next transfer's GC retries only our recorded objects.

    def gc(self):
        for p in self.state.glob('*.json'):
            try:
                v=json.loads(p.read_text())
                if v['created']<time.time()-86400 and v['key'].startswith(self.prefix):self.remove(v['key'],p)
            except (OSError,ValueError,KeyError):continue

    def import_pack(self,executor,engine,root,sha,raw):
        key,ledger=self.reserve(sha)
        try:
            from .remote_cache import Store
            local=Store(self.state.parent/'content-v2');local.put(raw)
            self.call(['cp',str(local.path(sha)),key,'-f'])
            url=self.sign(key)
            result=executor._exec([executor.config.get('python','python3'),'-c',FETCH,
                engine['path'],engine['sha256'],root,sha,str(len(raw)),url],retry_safe=True,operation='content_object_import')
            if not result.get('complete'):raise Failure('object_import_incomplete',transient=True)
            return result
        finally:self.remove(key,ledger)

    def download(self,executor,name,expected):
        from .remote import COMPRESS_FILE
        meta=executor._exec([executor.config.get('python','python3'),'-c',COMPRESS_FILE,executor.root,name],
                            retry_safe=True,operation='artifact_object_prepare')
        if meta['sha256']!=expected['sha256'] or meta['raw_bytes']!=expected['bytes']:raise Failure('artifact_changed')
        key,ledger=self.reserve(meta['encoded_sha256'])
        try:
            result=executor._exec([executor.config.get('python','python3'),'-c',EXPORT,executor.root,meta['name'],
                meta['encoded_sha256'],str(meta['bytes']),self.sign(key,'PUT')],retry_safe=True,operation='artifact_object_export')
            if result.get('sha256')!=meta['encoded_sha256']:raise Failure('artifact_object_identity_mismatch')
            try:
                with urllib.request.urlopen(self.sign(key),timeout=60) as response:encoded=response.read(meta['bytes']+1)
            except Exception:raise Failure('object_download_unavailable',transient=True) from None
            if len(encoded)!=meta['bytes'] or hashlib.sha256(encoded).hexdigest()!=meta['encoded_sha256']:
                raise Failure('artifact_object_checksum_mismatch')
            decoder=zlib.decompressobj();data=decoder.decompress(encoded,expected['bytes']+1)
            if not decoder.eof or decoder.unused_data or len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:
                raise Failure('artifact_object_checksum_mismatch')
            return data
        finally:self.remove(key,ledger)
