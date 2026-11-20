"""The existing AKA execution entry point, bound to a task's tool host."""
import argparse
import base64
import hashlib
import os
import uuid
from pathlib import Path
from aka.light.tasks.content import document
from aka.light.tasks.runtime import invoke
from .package import encode_files


def main(argv=None):
    parser = argparse.ArgumentParser(prog="aka execution")
    parser.add_argument("--input", action="append", default=[], metavar="REMOTE=LOCAL")
    parser.add_argument("--output", action="append", default=[])
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--execution-id")
    parser.add_argument('--launch-mode',choices=['command','ranks'])
    parser.add_argument('--inputs-only',action='store_true',help='Send only explicit --input files for this research command')
    parser.add_argument('--artifact',metavar='SHA256',help='Fetch a declared artifact by ID')
    parser.add_argument('--save',metavar='PATH')
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.artifact:
        if not args.save or args.command or args.input or args.output or args.env or args.execution_id or args.launch_mode or args.inputs_only:
            parser.error('Artifact retrieval takes only --artifact and --save')
        target=Path(args.save);target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.'+uuid.uuid4().hex+'.aka-download.tmp');offset=0;digest=hashlib.sha256()
        try:
            with temporary.open('wb') as stream:
                while True:
                    value=invoke('artifact',{'artifact_id':args.artifact,'offset':offset,'size':500000})
                    data=base64.b64decode(value['base64'],validate=True)
                    if value['artifact_id']!=args.artifact or value['offset']!=offset or (not data and offset<value['bytes']):raise ValueError('Artifact transfer identity mismatch')
                    stream.write(data);digest.update(data);offset+=len(data)
                    if offset==value['bytes']:break
                stream.flush();os.fsync(stream.fileno())
            if digest.hexdigest()!=args.artifact:raise ValueError('Artifact checksum mismatch')
            temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
        print(document({'artifact_id':args.artifact,'bytes':offset,'path':str(target)}));return 0
    if args.save:parser.error('--save requires --artifact')
    if args.execution_id:
        if args.command or args.input or args.output or args.env or args.launch_mode or args.inputs_only:
            parser.error("Recovery takes only --execution-id")
        payload = {"execution_id": args.execution_id}
    else:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("Provide a command after --")
        files, environment = {}, {}
        for item in args.input:
            remote, path = item.split("=", 1)
            files.update(encode_files({remote: Path(path).read_bytes()}))
        for item in args.env:
            key, value = item.split("=", 1)
            environment[key] = value
        payload = {"argv": command, "files": files, "outputs": args.output, "environment": environment}
        if args.launch_mode:payload['launch_mode']=args.launch_mode
        if args.inputs_only:payload['source_mode']='explicit'
    result = invoke("execute", payload)
    print(document(result))
    return 0 if result["receipt"]["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
