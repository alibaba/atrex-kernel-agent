"""Agent-owned durable normalized invocation facts, independent of CLI layout."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from aka.contracts.workflow_export import SessionFact


def record_invocation(workspace, runtime, session_id, usage, events, errors=(), *, resume_supported=False, stdout=""):
    root = Path(workspace) / '.aka-agent-observations'
    root.mkdir(parents=True, exist_ok=True)
    value = {'schema': 1, 'runtime': runtime, 'session_id': session_id,
             'updated_at': datetime.now(timezone.utc).isoformat(),
             'usage': asdict(usage), 'events': [asdict(event) for event in events],
             'errors': list(errors), 'resume_supported': resume_supported}
    value['events'].extend(_conversation_events(runtime, stdout, len(value['events'])))
    path = root / (str(uuid.uuid4()) + '.json')
    temporary = path.with_suffix('.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(value, allow_nan=False) + '\n')
    temporary.replace(path)


def read_sessions(workspace, *, direct=False):
    rows = []
    root = Path(workspace) if direct else Path(workspace)/'.aka-agent-observations'
    for path in root.glob('*.json'):
        try:
            value = json.loads(path.read_bytes())
            if value['schema'] != 1:
                raise ValueError('unsupported Agent observation schema')
            rows.append(SessionFact(value['session_id'], value['runtime'], None,
                'observation_error' if value.get('errors') else 'observed', value['updated_at'], value['usage'], tuple(value['events']),
                value['resume_supported']))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError('Agent observation is corrupt') from exc
    return tuple(sorted(rows, key=lambda row: (row.updated_at or "", row.session_id)))


def _conversation_events(runtime, stdout, sequence):
    """Normalize Agent-visible conversation content; never export native records."""
    result = []
    for line in stdout.splitlines():
        try:
            record=json.loads(line)
        except ValueError:
            continue
        if not isinstance(record,dict):
            continue
        message=record.get('message')
        blocks=message.get('content',[]) if isinstance(message,dict) else []
        if isinstance(blocks,str): blocks=[{'type':'text','text':blocks}]
        if runtime=='codex':
            item=record.get('item') or {}
            if isinstance(item,dict):
                if item.get('type')=='agent_message': blocks=[{'type':'text','text':item.get('text','')}]
                elif item.get('type')=='command_execution':
                    blocks=[{'type':'tool_use','name':'command','id':item.get('id'),'input':item.get('command')}]
        for block in blocks if isinstance(blocks,list) else []:
            if not isinstance(block,dict) or block.get('type') not in {'text','tool_use','tool_result'}:
                continue
            kind={'text':'message','tool_use':'tool_call','tool_result':'tool_result'}[block['type']]
            content=block.get('text',block.get('input',block.get('content','')))
            rendered=content if isinstance(content,str) else json.dumps(content,ensure_ascii=False)
            result.append({'sequence':sequence+len(result),'kind':kind,'role':record.get('type'),
                           'tool_name':block.get('name'),'tool_id':block.get('id',block.get('tool_use_id')),
                           'content':rendered[:16000]})
    return result
