"""Select one implementation; legacy discovery occurs only when no selection is recorded."""
import json,subprocess
from importlib import import_module
from pathlib import Path
from .common import POLICY,SELECTION,MAX_FILE,read_regular_bytes,FileSystem

IMPLEMENTATIONS={'single_file':('single_file','SingleFile'), 'source_tree':('source_tree','SourceTree'),
                 'cuda_manifest':('cuda_manifest','CudaManifest')}

def select(name: str) -> FileSystem:
    try:module,kind=IMPLEMENTATIONS[name]
    except KeyError:raise ValueError('Unknown candidate file system: '+str(name)) from None
    return getattr(import_module('.'+module,__package__),kind)()

def for_policy(policy):
    if policy is not None and not isinstance(policy,dict):raise ValueError('Invalid source contract')
    return select('single_file' if policy is None else 'cuda_manifest' if policy.get('kind')=='agate_cuda' else 'source_tree')

def task_selection(name, *, framework, source_tree=False):
    selected=name or ('source_tree' if source_tree else 'cuda_manifest' if framework.lower()=='cuda' else 'single_file')
    system=select(selected)
    if source_tree != (selected=='source_tree'):
        raise ValueError('source_tree file system requires source_tree configuration, and other file systems cannot use it')
    if (framework.lower()=='cuda') != (selected=='cuda_manifest') and not source_tree:
        raise ValueError('framework=cuda requires the cuda_manifest file system')
    return system

def resolve(workspace, *, revision=None):
    def get(name):
        if revision is not None:
            p=subprocess.run(['git','show',f'{revision}:{name}'],cwd=workspace,capture_output=True)
            return p.stdout if p.returncode==0 else None
        try:return read_regular_bytes(workspace/name,limit=MAX_FILE+1)
        except FileNotFoundError:return None
    return resolve_reader(get)

def resolve_reader(get):
    selection = get(SELECTION)
    if selection is not None:
        system = select(json.loads(selection)['name'])
        claimed, contract = system.inspect_policy(get)
        contract = contract if claimed else system.default_policy
        system.validate_policy(contract)
        return system, contract
    # Only legacy inputs without a selector use discovery. A claimed invalid
    # format raises immediately; it never falls through to another implementation.
    for name in ('source_tree', 'cuda_manifest', 'single_file'):
        system = select(name)
        claimed, contract = system.inspect_policy(get)
        if claimed:
            system.validate_policy(contract)
            return system, contract
    raise ValueError('No file system accepts this legacy workspace')

def from_inputs(inputs, prefix=''):
    return resolve_reader(lambda name: inputs.get(prefix+name))
