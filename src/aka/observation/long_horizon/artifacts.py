"""Bind declared Workflow output bytes to detached observation downloads."""
import hashlib
import io
import json
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from aka.contracts.observation import check_observation_version

SCOPE = 'Declared Workflow public outputs captured with this observation; evaluator truth is excluded.'


def capture_artifacts(facts, snapshot):
    check_observation_version(snapshot.get('observation_version'))
    report = {key: snapshot[key] for key in ('observation_version', 'name', 'framework',
              'platform', 'arch', 'updated_at', 'canonical', 'counts', 'tokens', 'progress', 'issues')}
    summary = (json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
    files = []
    for artifact in facts.artifacts:
        identifier = 'output_' + hashlib.sha256(artifact.id.encode()).hexdigest()
        files.append((identifier, artifact.filename, artifact.role.replace('-', ' ').capitalize(),
                      artifact.content_type, facts.artifact(artifact.id), 'file'))
    files.append(('summary', '_observation/result-summary.json', 'Optimization summary', 'application/json', summary, 'file'))
    buffer = io.BytesIO()
    with ZipFile(buffer, 'w', compression=ZIP_DEFLATED) as archive:
        for _, filename, _, _, content, _ in files:
            entry = ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            archive.writestr(entry, content)
    files.append(('bundle', 'results.zip', 'Result bundle', 'application/zip', buffer.getvalue(), 'bundle'))
    identity = json.dumps([(row[0], row[1], hashlib.sha256(row[4]).hexdigest()) for row in files],
                          ensure_ascii=False, separators=(',', ':')).encode()
    snapshot_id = hashlib.sha256(identity).hexdigest()
    contents, descriptions = {}, []
    for identifier, filename, label, mime, content, kind in files:
        download_id = identifier + '_' + snapshot_id
        contents[download_id] = content
        descriptions.append({'id': download_id, 'filename': PurePosixPath(filename).name, 'label': label,
                             'content_type': mime, 'kind': kind, 'size_bytes': len(content),
                             'sha256': hashlib.sha256(content).hexdigest(), 'snapshot_id': snapshot_id, 'visibility': 'public'})
    return descriptions, contents
