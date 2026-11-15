# AKA Light installation

Read this file when installing or updating Light from an AKA source checkout.

## Prerequisites

- Python 3.10 or later
- `git` and `bash`
- A complete AKA source checkout
- A configured Agent CLI, such as `claude`, `qodercli`, `codex`, or `pi`
- For real optimization runs: access to Agate, a target GPU worker, and the required framework environment

Light consists of three Python distributions that must come from the same source revision:

- `atrex-aka-framework`: the `aka` CLI and plugin framework
- `atrex-aka-legacy`: shared Task, Agent, and compatibility components required by Light
- `atrex-aka-light`: the Light Workflow, Runtime, observability, and reliable-execution implementation

The three wheels and the resource checkout passed through `--repo-root` must come from the same revision. The packages commonly all report version `0.1.0`, so rebuild and force-reinstall them after a source update; do not rely on the displayed package version alone.

## Agate dependency and Atrex-Bench source

Install `atrex-gateway-client==0.14.31`, which provides the Agate Python SDK and `agate` CLI, into
the same Python environment that will run `aka optimize`:

```bash
python3 -m pip install --upgrade \
  --index-url http://artlab.alibaba-inc.com/1/pypi/pypi-releases \
  --trusted-host artlab.alibaba-inc.com \
  'atrex-gateway-client==0.14.31'
```

Verify the exact distribution and CLI before launching:

```bash
python3 - <<'PY'
from importlib.metadata import version
print("atrex-gateway-client", version("atrex-gateway-client"))
PY
agate --version
```

Obtain Atrex-Bench as a canonical source checkout from GitHub:

```bash
git clone https://github.com/alibaba/atrex-bench.git /path/to/atrex-bench
test -f /path/to/atrex-bench/scripts/run_eval.py
test -d /path/to/atrex-bench/src/atrex_bench
```

Record the selected Atrex-Bench commit for reproducibility and supply this checkout root during
launch intake. Native inputs require the checkout itself, including `scripts/run_eval.py`,
`src/atrex_bench/`, and the operator directory; the Skill does not install or expand an
Atrex-Bench wheel.

## Recommended installation script

Create and activate an isolated virtual environment:

```bash
python3 -m venv /path/to/aka-venv
source /path/to/aka-venv/bin/activate
```

Run the installer bundled with this skill. Its one required positional argument is `AKA_REPO_ROOT`, the complete AKA checkout:

```bash
bash ~/.codex/skills/atrex-aka-light/scripts/install_light.sh \
  /path/to/atrex-kernel-agent
```

Select a specific Python interpreter with `--python`:

```bash
bash ~/.codex/skills/atrex-aka-light/scripts/install_light.sh \
  --python /path/to/aka-venv/bin/python \
  /path/to/atrex-kernel-agent
```

Retain the generated wheels with `--wheel-dir`:

```bash
bash ~/.codex/skills/atrex-aka-light/scripts/install_light.sh \
  --wheel-dir /tmp/aka-wheels \
  /path/to/atrex-kernel-agent
```

The installer validates the checkout and interpreter, builds matching framework, legacy, and Light wheels, force-reinstalls them from the local wheel directory, and verifies both `python -m aka --help` and `python -m aka optimize --help`.

If the environment cannot install the Python `build` frontend from a package index, install it beforehand and pass `--no-install-build-dependency`.

## Manual installation

These commands match the upstream installation flow:

```bash
cd /path/to/atrex-kernel-agent
python3 -m pip install build
python3 tools/build_distributions.py --output /tmp/aka-wheels \
  atrex-aka-framework atrex-aka-legacy atrex-aka-light
python3 -m pip install --force-reinstall --no-index \
  --find-links /tmp/aka-wheels atrex-aka-light
python3 -m aka optimize --help
```

`--no-index --find-links` prevents installation from mixing locally built packages with same-name, same-version packages from a remote index.

## Verify the installation

Inspect the installed distributions and CLI:

```bash
python3 - <<'PY'
from importlib.metadata import distribution

for name in ("atrex-aka-framework", "atrex-aka-legacy", "atrex-aka-light"):
    dist = distribution(name)
    print(name, dist.version, dist.locate_file(""))
PY

python3 -m aka --help
python3 -m aka optimize --help
```

At launch, pass the same source checkout through `--repo-root`. Read [operations.md](operations.md) for launch arguments and mode selection.

## Update Light

1. Inspect the AKA checkout's branch, commit, and uncommitted changes.
2. Update it as requested without overwriting local changes.
3. Run the installation script again.
4. Start a new campaign with the updated environment, or verify resume identity before resuming an existing campaign.

Do not skip reinstallation merely because the distributions still report version `0.1.0`. Do not replace code underneath a running campaign and assume its processes loaded the update; restart through the campaign's Supervisor or workload mechanism and verify the result.

## Bubblewrap proc modes

Light supports exactly two Bubblewrap process-view modes:

- `private` creates a private PID namespace and mounts a new procfs. It is supported on ordinary Linux and Lima.
- `host` shares the outer container's PID namespace and bind-mounts its complete `/proc`. It is the only supported Bubblewrap proc mode inside Aone Sandbox because Aone rejects a nested procfs mount; use a dedicated Sandbox.

Host mode requires Bubblewrap but no C compiler, development headers, compatibility shim or `LD_PRELOAD`. It preserves the filesystem allowlist and dropped capabilities, but it deliberately does not isolate process visibility within the Aone Sandbox.
For every new launch, ask the user to select `private` or `host` after they select
`light-sandbox`. Inside Aone, ask the user to accept `host` or choose native `light`; do not apply a
proc-mode default on the user's behalf.
