"""Single Python source file; independent file-system implementation."""
import ast
import json
from pathlib import Path
from .common import Layout,ExecutionSource,read_regular_bytes,publish,git_blob,relative,MAX_FILE

class SingleFile:
    def prepare_source(self, request, source_provider, *, options, state_root, native_eval):
        return source_provider.prepare(request)
    name = 'single_file'
    default_policy = None
    def inspect_policy(self, get):return True, None
    def validate_restore(self, previous, target):pass
    layout = Layout('kernel.py', auxiliary_metadata=('solution.json',))
    def validate_policy(self, policy):
        if policy is not None:raise ValueError('Single-file candidates do not have a source-tree contract')
        return None
    def validate(self, files, policy):
        if set(files) != {'kernel.py'}:raise ValueError('Single-file candidate requires exactly kernel.py')
        return files
    def collect(self, workspace, policy, *, validate=True):
        return {'kernel.py':read_regular_bytes(workspace/'kernel.py',limit=MAX_FILE+1)}
    def from_git(self, workspace, revision, policy):return {'kernel.py':git_blob(workspace,revision,'kernel.py')}
    def from_inputs(self, inputs, policy, prefix):return {'kernel.py':inputs[prefix+'kernel.py']}
    def editable(self, policy, name):return name=='kernel.py'
    def instructions(self, policy):
        return (
            "\n## Single-file Kernel\n"
            "Editable `kernel.py` is the complete candidate and evaluator entry point. "
            "Keep the implementation self-contained in that file; do not add auxiliary candidate "
            "sources or edit `solution.json`, which is Supervisor-owned metadata in this layout. "
            "Evaluate, Profile, Check, Disassemble, reports and broadcasts bind the committed "
            "`kernel.py`.\n"
        )
    def metadata(self, policy):return {}
    def obsolete(self, workspace, policy):return {'kernel.py'}
    def install(self, workspace, files, policy):publish(workspace,'kernel.py',files['kernel.py'])
    def execution_source(self, files, policy, archive):return ExecutionSource(files['kernel.py'].decode(),{})
    def declared_paths(self, workspace, policy):
        path=workspace/'solution.json'
        if not path.is_file():return set()
        value=json.loads(path.read_text());entries=value.get('sources',[]) if isinstance(value,dict) else []
        if not isinstance(entries,list):raise ValueError('Candidate declaration sources must be a list of paths')
        return {relative(entry if isinstance(entry,str) else entry.get('path') if isinstance(entry,dict) else None) for entry in entries}
    def evidence_paths(self, workspace, policy):return {'kernel.py','solution.json'}|self.declared_paths(workspace,policy)

    @staticmethod
    def _selected_language(workspace, framework):
        language = framework.strip().lower()
        aliases = {
            'cute': 'cutedsl',
            'cute_dsl': 'cutedsl',
            'nvidia-cutlass-dsl': 'cutedsl',
        }
        language = aliases.get(language, language)
        if language:
            return language
        entry = workspace / 'kernel.py'
        try:
            tree = ast.parse(entry.read_text())
        except (OSError, UnicodeDecodeError, SyntaxError):
            return ''
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(value.name.split('.')[0] for value in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        for module, selected in (
            ('gluon', 'gluon'),
            ('triton', 'triton'),
            ('cutlass', 'cutedsl'),
            ('cute', 'cutedsl'),
            ('flydsl', 'flydsl'),
        ):
            if module in imports:
                return selected
        return ''

    def seed_metadata(self, workspace, framework, name, *, native_eval):
        """Synchronize Supervisor-owned metadata with the current Python candidate."""
        path = workspace / 'solution.json'
        language = self._selected_language(workspace, framework)
        if not native_eval:
            return
        if language == 'cuda':
            path.unlink(missing_ok=True)
            return
        dependencies = {
            'triton': ['torch', 'triton'],
            'gluon': ['torch', 'triton', 'gluon'],
            'cutedsl': ['torch', 'nvidia-cutlass-dsl'],
            'flydsl': ['torch', 'flydsl'],
        }.get(language, ['torch'])
        languages = ['python', language] if language else ['python']
        reference = workspace / 'reference.py'
        symbol = 'Model' if reference.is_file() and 'class Model' in reference.read_text() else 'run'
        suffix = language or 'python'
        roles = {
            'torch': 'Tensor interface and candidate entry-point plumbing.',
            'triton': 'Selected DSL and compiler for the candidate kernel.',
            'gluon': 'Selected DSL and compiler for the candidate kernel.',
            'nvidia-cutlass-dsl': 'Selected DSL and compiler for the candidate kernel.',
            'flydsl': 'Selected DSL and compiler for the candidate kernel.',
        }
        value = {'name': f'{name}_v1_{suffix}',
                 'spec': {'languages': languages, 'dependencies': dependencies, 'entry_point': f'kernel.py::{symbol}'},
                 'sources': [{'path': 'kernel.py', 'role': 'Self-contained candidate implementation and framework loader.'}],
                 'dependency_roles': {dependency: roles[dependency] for dependency in dependencies}}
        contents = json.dumps(value, indent=2) + '\n'
        if not path.is_file() or path.read_text() != contents:
            path.write_text(contents)

    def typed_limitation(self, workspace, policy):
        entry = self.layout.python_entrypoint
        if not (workspace / entry).is_file():
            return 'missing ' + entry
        if self.declared_paths(workspace, policy) - {entry}:
            return 'candidate declaration includes auxiliary sources'
        try:
            tree = ast.parse((workspace / entry).read_text())
        except (OSError, UnicodeDecodeError, SyntaxError):
            return None
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(v.name.split('.')[0] for v in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
        local = sorted(n for n in imports if n not in {'input', 'reference', Path(entry).stem}
                       and ((workspace / (n + '.py')).is_file() or (workspace / n / '__init__.py').is_file()))
        return 'candidate imports local auxiliary modules: ' + ', '.join(local) if local else None

    def language_hints(self, workspace):
        selected = self._selected_language(workspace, '')
        if selected:
            return ['python', selected]
        path = workspace/'solution.json'
        value = json.loads(path.read_text()) if path.is_file() else {}
        spec = value.get('spec', {}) if isinstance(value, dict) else {}
        hints = spec.get('languages', []) if isinstance(spec, dict) else []
        return [str(v) for v in hints] if isinstance(hints,list) else []
