"""Versioned content store and build cache; standalone stdlib on a GPU worker.

Inputs are complete manifests. Transport batches are disposable and never
become experiment identity. Materializations are private copies/reflinks, never
writable hard links into the store.
"""
from contextlib import contextmanager
import base64
import fcntl
import hashlib
import inspect
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
import zlib

VERSION=2
MAX_BYTES=64*1024*1024
MAX_FILES=4096


def digest(data):return hashlib.sha256(data).hexdigest()
def document(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def valid_hash(value):
    if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{64}',value):raise ValueError('Invalid content digest')
    return value


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise ValueError('Cache path is a symlink')
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


@contextmanager
def lock(path,exclusive=True,blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise ValueError('Cache lock is a symlink')
    with path.open('a') as stream:
        fcntl.flock(stream,(fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)|(0 if blocking else fcntl.LOCK_NB))
        yield


def manifest(files):
    value={'version':VERSION,'files':{name:{'sha256':digest(data),'bytes':len(data),'mode':420}
                                     for name,data in sorted(files.items())}}
    validate_manifest(value);return value


def validate_manifest(value):
    if not isinstance(value,dict) or set(value)!={'version','files'} or value['version']!=VERSION:raise ValueError('Invalid content manifest version')
    files=value['files']
    if not isinstance(files,dict) or not 0<=len(files)<=MAX_FILES:raise ValueError('Invalid content manifest file count')
    total=0
    for name,entry in files.items():
        p=PurePosixPath(name)
        if (not isinstance(name,str) or not name or p.is_absolute() or '..' in p.parts or '.git' in p.parts
                or str(p)!=name or '\\' in name or '\x00' in name):raise ValueError('Invalid manifest path')
        if any(str(parent) in files for parent in p.parents if str(parent)!='.'):raise ValueError('Manifest file/directory collision')
        if not isinstance(entry,dict) or set(entry)!={'sha256','bytes','mode'}:raise ValueError('Invalid manifest entry')
        valid_hash(entry['sha256'])
        if type(entry['bytes']) is not int or not 0<=entry['bytes']<=MAX_BYTES or entry['mode'] not in (420,493):raise ValueError('Invalid manifest size/mode')
        total+=entry['bytes']
    if total>MAX_BYTES:raise ValueError('Expanded source exceeds 64 MiB')
    return digest(document(value))


def wire(value):return base64.b64encode(zlib.compress(document(value),6)).decode()
def unwire(value):
    encoded=base64.b64decode(value,validate=True);decoder=zlib.decompressobj()
    raw=decoder.decompress(encoded,2*1024*1024+1)
    if len(raw)>2*1024*1024 or not decoder.eof or decoder.unused_data:raise ValueError('Invalid compressed manifest')
    return json.loads(raw)


def private_copy(source,target):
    """Try filesystem copy-on-write, then an ordinary copy; never hardlink."""
    source,target=Path(source),Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():raise ValueError('Snapshot target is a symlink')
    if os.uname().sysname=='Darwin':
        import ctypes
        libc=ctypes.CDLL(None,use_errno=True)
        if hasattr(libc,'clonefile') and libc.clonefile(os.fsencode(source),os.fsencode(target),0)==0:return
    else:
        try:
            with source.open('rb') as src,target.open('xb') as dst:fcntl.ioctl(dst.fileno(),0x40049409,src.fileno())
            return
        except OSError:target.unlink(missing_ok=True)
    shutil.copyfile(source,target)


class Store:
    def __init__(self,root):
        self.root=Path(root)
        if self.root.is_symlink():raise ValueError('Content store is a symlink')
        self.root.mkdir(parents=True,exist_ok=True)

    def path(self,sha):
        valid_hash(sha);path=self.root/'blobs'/sha[:2]/sha
        if any(p.is_symlink() for p in [path,path.parent,path.parent.parent]):raise ValueError('Blob path is a symlink')
        return path

    def has(self,sha,size=None):
        p=self.path(sha)
        return p.is_file() and (size is None or p.stat().st_size==size) and digest(p.read_bytes())==sha

    def put(self,data):
        sha=digest(data)
        with lock(self.root/'gc.lock',exclusive=False),lock(self.root/'locks'/sha):
            if not self.has(sha,len(data)):atomic(self.path(sha),data)
        return sha

    def get(self,sha):
        with lock(self.root/'gc.lock',exclusive=False):
            p=self.path(sha);data=p.read_bytes()
            if digest(data)!=sha:raise ValueError('Cached blob checksum mismatch')
            return data

    def put_files(self,files):
        value=manifest(files)
        for data in files.values():self.put(data)
        return self.commit(value)

    def prepare(self,value,known_packs=()):
        mid=validate_manifest(value)
        # A live staging pin protects all existing dependencies during upload.
        atomic(self.root/'staging'/(mid+'.json'),document({'manifest':value,'expires':time.time()+86400}))
        missing={e['sha256']:e['bytes'] for e in value['files'].values() if not self.has(e['sha256'],e['bytes'])}
        packs={}
        for sha in known_packs:
            valid_hash(sha);directory=self.root/'packs'/sha
            packs[sha]={p.name[5:]:digest(p.read_bytes()) for p in directory.glob('part-*') if p.name[5:].isdigit()}
        return {'manifest_id':mid,'missing':missing,'packs':packs}

    def seed_archives(self,directory,missing):
        """Import matching content from verified V1 archives; never trust paths."""
        imported=0;remaining=dict(missing)
        for path in sorted(Path(directory).glob('*.zip'),key=lambda p:p.stat().st_mtime,reverse=True)[:64]:
            if not remaining:break
            if path.is_symlink() or not re.fullmatch('[a-f0-9]{64}',path.stem):continue
            raw=path.read_bytes()
            if len(raw)>MAX_BYTES or digest(raw)!=path.stem:continue
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                    infos=archive.infolist()
                    if len(infos)>MAX_FILES or sum(i.file_size for i in infos)>MAX_BYTES:continue
                    for info in infos:
                        if info.file_size not in remaining.values():continue
                        blob=archive.read(info);sha=digest(blob)
                        if sha in remaining and remaining[sha]==len(blob):self.put(blob);remaining.pop(sha);imported+=1
            except (zipfile.BadZipFile,RuntimeError):continue
        return imported

    def commit(self,value):
        mid=validate_manifest(value)
        with lock(self.root/'gc.lock',exclusive=False):
            if any(not self.has(e['sha256'],e['bytes']) for e in value['files'].values()):raise ValueError('Source manifest has missing or corrupt blobs')
            atomic(self.root/'manifests'/(mid+'.json'),document(value))
        return mid

    def load(self,mid):
        valid_hash(mid);value=json.loads((self.root/'manifests'/(mid+'.json')).read_text())
        if validate_manifest(value)!=mid:raise ValueError('Cached manifest checksum mismatch')
        return value

    def files(self,mid):return {name:self.get(e['sha256']) for name,e in self.load(mid)['files'].items()}

    def materialize(self,mid,directory):
        directory=Path(directory)
        if directory.exists():raise ValueError('Snapshot destination already exists')
        staging=directory.with_name(directory.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with lock(self.root/'gc.lock',exclusive=False):
                value=self.load(mid);staging.mkdir(parents=True)
                for name,e in value['files'].items():
                    if not self.has(e['sha256'],e['bytes']):raise ValueError('Snapshot dependency missing or corrupt')
                    target=staging/name;private_copy(self.path(e['sha256']),target);target.chmod(e['mode'])
                staging.replace(directory)
        finally:
            if staging.exists():shutil.rmtree(staging)

    def put_pack(self,sha,index,count,size,data):
        valid_hash(sha)
        if (type(index) is not int or type(count) is not int or not 0<=index<count<=1000
                or type(size) is not int or not 0<size<=MAX_BYTES+1024*1024 or len(data)>90000):raise ValueError('Invalid content batch')
        directory=self.root/'packs'/sha
        with lock(self.root/'pack-locks'/sha):
            directory.mkdir(parents=True,exist_ok=True)
            metadata={'count':count,'bytes':size}
            meta=directory/'meta.json'
            if meta.exists() and json.loads(meta.read_text())!=metadata:raise ValueError('Content batch identity mismatch')
            atomic(meta,document(metadata));atomic(directory/('part-'+str(index)),data)
            if not all((directory/('part-'+str(i))).exists() for i in range(count)):return {'complete':False}
            raw=b''.join((directory/('part-'+str(i))).read_bytes() for i in range(count))
            if len(raw)!=size or digest(raw)!=sha:raise ValueError('Content batch checksum mismatch')
            if not zipfile.is_zipfile(io.BytesIO(raw)):raise ValueError('Invalid content batch archive')
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                items=archive.infolist()
                if len(items)>MAX_FILES or sum(x.file_size for x in items)>MAX_BYTES:raise ValueError('Content batch exceeds limit')
                if len({x.filename for x in items})!=len(items):raise ValueError('Duplicate content batch objects')
                for item in items:
                    valid_hash(item.filename);blob=archive.read(item)
                    if digest(blob)!=item.filename:raise ValueError('Content object checksum mismatch')
                    self.put(blob)
            return {'complete':True,'objects':len(items)}

    def gc(self,max_bytes,keep=(),ttl_s=86400):
        """Evict disposable snapshots by age/size; callers pin durable/live IDs."""
        if type(max_bytes) is not int or max_bytes<0:raise ValueError('Invalid cache capacity')
        keep=set(keep);removed=[]
        with lock(self.root/'gc.lock'):
            manifests=[];referenced=set();now=time.time()
            for p in (self.root/'staging').glob('*.json'):
                v=json.loads(p.read_text())
                if v['expires']>now:referenced.update(e['sha256'] for e in v['manifest']['files'].values())
                else:p.unlink()
            for p in (self.root/'manifests').glob('*.json'):
                value=json.loads(p.read_text());mid=validate_manifest(value)
                manifests.append((p.stat().st_mtime,p,mid,value))
            for _,p,mid,value in sorted(manifests,reverse=True):
                sizes={e['sha256']:e['bytes'] for e in value['files'].values()}
                growth=sum(size for sha,size in sizes.items() if sha not in referenced)
                if mid in keep or (now-p.stat().st_mtime<=ttl_s and sum(self.path(h).stat().st_size for h in referenced if self.path(h).exists())+growth<=max_bytes):
                    referenced.update(sizes)
                else:p.unlink();removed.append(mid)
            freed=0
            for p in (self.root/'blobs').glob('*/*'):
                if p.is_file() and p.name not in referenced:freed+=p.stat().st_size;p.unlink()
            # Partial wire batches are independently disposable after their TTL.
            for p in (self.root/'packs').glob('*'):
                if p.is_dir() and now-p.stat().st_mtime>ttl_s:
                    try:
                        with lock(self.root/'pack-locks'/p.name,blocking=False):shutil.rmtree(p)
                    except BlockingIOError:pass
        return {'evicted_manifests':removed,'freed_bytes':freed}


def packs(files,missing,max_raw=4*1024*1024):
    """Deduplicate by blob hash and batch objects; final wire chunks are bounded."""
    unique={digest(data):data for data in files.values() if digest(data) in missing}
    batches=[];batch={};size=0
    def flush():
        nonlocal batch,size
        if not batch:return
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for sha,data in sorted(batch.items()):archive.writestr(zipfile.ZipInfo(sha,date_time=(1980,1,1,0,0,0)),data,compress_type=zipfile.ZIP_DEFLATED)
        raw=stream.getvalue();batches.append((digest(raw),raw,list(batch)));batch={};size=0
    for sha,data in sorted(unique.items()):
        if batch and size+len(data)>max_raw:flush()
        batch[sha]=data;size+=len(data)
    flush();return batches


def load_torch_extension(root,paths,cflags,ldflags):
    """One verified host extension per complete source/toolchain recipe, all ranks."""
    import importlib.util
    import torch
    from torch.utils.cpp_extension import load,CUDA_HOME
    root=Path(root);base=Path(os.environ['AKA_BUILD_CACHE_DIR']);base.mkdir(parents=True,exist_ok=True)
    def compiler(argv):
        p=subprocess.run(argv,capture_output=True,text=True,check=True);return p.stdout.strip()
    recipe={'version':VERSION,'sources':{str(p.relative_to(root)):digest(p.read_bytes()) for p in paths if p.is_file()},
        'loader':digest((root/'source_build.py').read_bytes()),'builder':digest(inspect.getsource(load_torch_extension).encode()),
        'cflags':cflags,'ldflags':ldflags,'torch':torch.__version__,'cuda':torch.version.cuda,
        'abi':torch.compiled_with_cxx11_abi(),'python':list(__import__('sys').version_info[:2]),
        'arch':list(torch.cuda.get_device_capability()),'sm_count':torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
        'arch_list':os.environ.get('TORCH_CUDA_ARCH_LIST',''),'cxx':os.environ.get('CXX','c++'),
        'compiler':compiler([os.environ.get('CXX','c++'),'--version']),
        'nvcc':compiler([str(Path(CUDA_HOME)/'bin/nvcc'),'--version']),
        'environment':{key:os.environ.get(key,'') for key in ['CFLAGS','CXXFLAGS','LDFLAGS','CC','CUDAHOSTCXX','NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS']}}
    # Torch appends link options to caller-owned lists. Freeze the recipe
    # before invoking it, and pass separate lists to the builder.
    recipe=json.loads(document(recipe))
    key=digest(document(recipe));directory=base/'host'/key;name='aka_dg_'+key[:16]
    pin_build(base,key)
    hit=False;recovered=''
    with lock(base/'locks'/(key+'.lock')):
        proof=directory/'complete.json'
        if proof.exists():
            try:
                value=json.loads(proof.read_text())
                if Path(value['library']).name!=value['library']:raise ValueError('Invalid cached library name')
                library=directory/value['library']
                hit=value['recipe']==recipe and library.is_file() and digest(library.read_bytes())==value['sha256']
            except (OSError,ValueError,KeyError,TypeError):
                hit=False
        if hit:
            spec=importlib.util.spec_from_file_location(name,library);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        else:
            # The outer OS lock serializes every writer. A killed Torch build
            # can leave its O_EXCL FileBaton behind without complete.json; a
            # retry must reset that uncommitted directory before calling load.
            # Never infer abandonment from age or remove an active writer's lock.
            if directory.exists():
                recovered='invalid' if proof.exists() else 'incomplete'
                shutil.rmtree(directory)
            directory.mkdir(parents=True,exist_ok=True)
            module=load(name=name,sources=[str(root/'bindings.cpp')],
                extra_include_paths=[str(root),str(root/'include'),str(Path(CUDA_HOME)/'include'),str(Path(CUDA_HOME)/'include/cccl')],
                extra_cflags=list(cflags),extra_ldflags=list(ldflags),with_cuda=True,build_directory=str(directory),verbose=True)
            library=Path(module.__file__)
            atomic(proof,document({'recipe':recipe,'library':library.name,'sha256':digest(library.read_bytes())}))
        os.utime(proof,None)
    os.environ['DG_JIT_CACHE_DIR']=str(base/'jit'/key)
    os.environ['AKA_JIT_BUILD_KEY']=key
    print('AKA_BUILD_CACHE='+json.dumps({'key':key,'hit':hit,'library':str(library),'recovered':recovered}),flush=True)
    gc_builds(base,int(os.environ.get('AKA_BUILD_CACHE_LIMIT',str(16*1024**3))),keep={key})
    return module


@contextmanager
def jit_compile():
    root=os.environ.get('AKA_BUILD_CACHE_DIR');key=os.environ.get('AKA_JIT_BUILD_KEY')
    if not root or not key:yield;return
    with lock(Path(root)/'locks'/(valid_hash(key)+'.jit.lock')):yield


_pins={}
_pin_guard=threading.RLock()
def pin_build(base,key):
    import atexit
    name=str(Path(base)/'locks'/(key+'.lifetime'))
    with _pin_guard:
        if name in _pins:return
        p=Path(name);p.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(p,os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_SH);_pins[name]=fd
    def release():
        value=_pins.pop(name,None)
        if value is not None:os.close(value)
    atexit.register(release)


def gc_builds(base,max_bytes,keep=()):
    base=Path(base);entries=[];total=0;freed=0
    for p in (base/'host').glob('*'):
        if not p.is_dir() or not re.fullmatch('[a-f0-9]{64}',p.name):continue
        paths=[p,base/'jit'/p.name]
        size=sum(f.stat().st_size for path in paths for f in path.rglob('*') if f.is_file())
        stamp=(p/'complete.json').stat().st_mtime if (p/'complete.json').exists() else p.stat().st_mtime
        entries.append((stamp,p,size));total+=size
    for _,p,size in sorted(entries):
        if total<=max_bytes:break
        if p.name in keep:continue
        try:
            with lock(base/'locks'/(p.name+'.lifetime'),blocking=False),lock(base/'locks'/(p.name+'.lock'),blocking=False):
                shutil.rmtree(p);shutil.rmtree(base/'jit'/p.name,ignore_errors=True);total-=size;freed+=size
        except BlockingIOError:continue
    return {'freed_bytes':freed,'remaining_bytes':total,'over_capacity':total>max_bytes}
