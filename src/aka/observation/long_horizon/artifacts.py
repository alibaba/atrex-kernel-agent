"""Capture public outputs once and bind download IDs to exact content."""
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

OUTPUTS = (('kernel', 'kernel.py', 'Kernel source', 'text/x-python'),
           ('solution', 'solution.json', 'Solution manifest', 'application/json'))
SCOPE = 'Measured incumbent revision content captured with this observation; evaluator truth is excluded.'


def capture_artifacts(workspace: Path, snapshot):
    contents, descriptions = {}, []
    report = {key: snapshot[key] for key in ('observation_version', 'name', 'framework',
              'platform', 'arch', 'updated_at', 'canonical', 'counts', 'tokens', 'progress', 'issues')}
    summary = (json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()
    files = []
    revision = (snapshot.get('canonical', {}).get('incumbent') or {}).get('revision')
    if isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40}', revision):
        for identifier, filename, label, mime in OUTPUTS:
            blob = subprocess.run(['git','show',f'{revision}:{filename}'], cwd=workspace,capture_output=True)
            if blob.returncode == 0:
                files.append((identifier, filename, label, mime, blob.stdout, 'file'))
    files.append(('summary', 'result-summary.json', 'Optimization summary', 'application/json', summary, 'file'))
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
        for _, filename, _, _, content, _ in files:
            archive.writestr(filename, content)
    files.append(('bundle', 'results.zip', 'Result bundle', 'application/zip', buffer.getvalue(), 'bundle'))
    snapshot_id = hashlib.sha256(summary + b''.join(hashlib.sha256(row[4]).digest() for row in files)).hexdigest()
    for identifier, filename, label, mime, content, kind in files:
        digest = hashlib.sha256(content).hexdigest()
        download_id = identifier + '_' + snapshot_id
        contents[download_id] = content
        descriptions.append({'id': download_id, 'filename': filename, 'label': label,
                             'content_type': mime, 'kind': kind, 'size_bytes': len(content),
                             'sha256': digest, 'snapshot_id': snapshot_id, 'visibility': 'public'})
    return descriptions, contents
