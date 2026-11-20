"""Host transport for existing Kubernetes Pods. Commands are always passed as argv."""
import argparse
import json
import os
import subprocess
import sys

# Pin each execution inside the target container, including retries after a Pod
# is recreated between the API identity read and the exec request.
VERIFY_UID = """import os, sys
if os.environ.get("AKA_POD_UID") != sys.argv[1]:
    sys.exit("Pod UID mismatch or missing Downward API environment")
os.execvp(sys.argv[2], sys.argv[2:])
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('identity', 'exec'))
    parser.add_argument('--pod', required=True)
    for option in ('context', 'namespace', 'container', 'uid'):
        parser.add_argument('--' + option)
    args = list(sys.argv[1:] if argv is None else argv)
    boundary = args.index('--') if '--' in args else len(args)
    config = parser.parse_args(args[:boundary])
    command = args[boundary + 1:]
    if any(value and value.startswith('-') for value in
           (config.pod, config.context, config.namespace, config.container, config.uid)):
        parser.error('invalid target identifier')
    kubectl = ['kubectl']
    for key in ('context', 'namespace'):
        value = getattr(config, key)
        if value:
            kubectl.extend(['--' + key, value])
    if config.operation == 'identity':
        if command:
            parser.error('identity accepts no command')
        proc = subprocess.run([*kubectl, 'get', 'pod', config.pod, '-o', 'json'], capture_output=True)
        if proc.returncode:
            sys.stderr.buffer.write(proc.stderr)
            return proc.returncode
        pod = json.loads(proc.stdout)
        print(json.dumps({'metadata': {'uid': pod['metadata']['uid']},
                          'status': {'containerStatuses': pod['status'].get('containerStatuses', [])}}))
        return 0
    if not command:
        parser.error('exec requires -- and a command')
    if config.uid:
        command = ['python3', '-c', VERIFY_UID, config.uid, *command]
    args = [*kubectl, 'exec', config.pod]
    if config.container:
        args.extend(['--container', config.container])
    os.execvp('kubectl', [*args, '--', *command])


if __name__ == '__main__':
    raise SystemExit(main())
