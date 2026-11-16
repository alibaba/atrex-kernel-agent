# AKA Light installation

Read this file when installing or updating the 0.2.0 module distributions from a matching AKA source checkout. The skill and `observability/` are checkout resources, not installed Python modules; keep the checkout available. Copy the complete skill directory into the Agent's configured skill root when installing the skill there.

Update both `skills/atrex-aka-light/` and `skills/atrex-aka-light-observability/`, including their
references and examples. Run the observability scripts from this same checkout. The optional
`observability/light_dashboard.py` also runs directly from the checkout; no Legacy observation
adapter or additional Dashboard wheel is needed for that entry point.

## Prerequisites

- Python 3.10 or later
- `git` and `bash`
- A complete AKA source checkout
- A configured Agent CLI, such as `claude`, `qodercli`, `codex`, `pi`, or `dsh`
- For real optimization runs: access to Agate, a target GPU worker, and the required framework environment

Light consists of two Python distributions that must come from the same source revision:

- `atrex-aka-framework`: the CLI, contracts, Core and shared Task, Agent, Execution, Evaluation, Experiment and Resources modules
- `atrex-aka-light`: the managed Light Workflow and its Runtime

The two wheels and the resource checkout passed through `--repo-root` must come from the same revision. The packages commonly all report version `0.2.0`, so rebuild and force-reinstall them after a source update; do not rely on the displayed package version alone.

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
bash /path/to/atrex-kernel-agent/skills/atrex-aka-light/scripts/install_light.sh \
  /path/to/atrex-kernel-agent
```

Select a specific Python interpreter with `--python`:

```bash
bash /path/to/atrex-kernel-agent/skills/atrex-aka-light/scripts/install_light.sh \
  --python /path/to/aka-venv/bin/python \
  /path/to/atrex-kernel-agent
```

Retain the generated wheels with `--wheel-dir`:

```bash
bash /path/to/atrex-kernel-agent/skills/atrex-aka-light/scripts/install_light.sh \
  --wheel-dir /tmp/aka-wheels \
  /path/to/atrex-kernel-agent
```

The installer validates the checkout and interpreter, builds matching framework and Light wheels, force-reinstalls them from the local wheel directory, and verifies both `python -m aka --help` and `python -m aka optimize --help`.

The installer passes the two newly built AKA wheel paths explicitly to pip, so
same-name packages on an index cannot replace them. Online mode resolves external
dependencies using the configured package index. Build output is isolated from old
wheels in the retained directory.

For offline installation, supply `--offline --wheel-dir /path/to/wheelhouse`.
The wheelhouse must include the complete build and runtime dependency closure,
including `build`, its dependencies, `setuptools`, `wheel` and `packaging>=23`.
Build isolation also consumes this wheelhouse. A clean dependency resolution check
runs before runtime installation; missing wheels stop installation with the original
pip/build diagnostics. Preinstalled dependencies alone do not satisfy a forced reinstall.

If the environment cannot install the Python `build` frontend from a package index, install it beforehand and pass `--no-install-build-dependency`.

## Manual installation

These commands match the upstream installation flow:

```bash
cd /path/to/atrex-kernel-agent
python3 -m pip install build packaging
python3 tools/build_distributions.py --output /tmp/aka-wheels \
  atrex-aka-framework atrex-aka-light
python3 -m pip install --force-reinstall \
  /tmp/aka-wheels/atrex_aka_framework-0.2.0-py3-none-any.whl \
  /tmp/aka-wheels/atrex_aka_light-0.2.0-py3-none-any.whl
python3 -m aka optimize --help
```

Use the actual filenames produced by this checkout when versions change. Add
`--no-index --find-links /path/to/complete-wheelhouse` for offline installation.

## Verify the installation

Inspect the installed distributions and CLI:

```bash
python3 - <<'PY'
from importlib.metadata import distribution

for name in ("atrex-aka-framework", "atrex-aka-light"):
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

Do not skip reinstallation merely because the distributions still report version `0.2.0`. Do not replace code underneath a running campaign and assume its processes loaded the update; restart through the campaign's Supervisor or workload mechanism and verify the result.

## Bubblewrap proc modes

Light supports exactly two Bubblewrap process-view modes:

- `private` creates a private PID namespace and mounts a new procfs. It is supported on ordinary Linux and Lima.
- `host` shares the outer container's PID namespace and bind-mounts its complete `/proc`. It is the only supported Bubblewrap proc mode inside Aone Sandbox because Aone rejects a nested procfs mount; use a dedicated Sandbox.

Host mode requires Bubblewrap but no C compiler, development headers, compatibility shim or `LD_PRELOAD`. It preserves the filesystem allowlist and dropped capabilities, but it deliberately does not isolate process visibility within the Aone Sandbox.
For every new launch, ask the user to select `private` or `host` after they select
`light-sandbox`. Inside Aone, ask the user to accept `host` or choose native `light`; do not apply a
proc-mode default on the user's behalf.
