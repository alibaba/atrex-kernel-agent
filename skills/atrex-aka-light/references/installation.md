# AKA Light installation

Read this file when installing or updating the 0.2.0 module distributions from a matching AKA source checkout. The skill and `observability/` are checkout resources, not installed Python modules; keep the checkout available. Copy the complete skill directory into the Agent's configured skill root when installing the skill there.

## Prerequisites

- Python 3.10 or later
- `git` and `bash`
- A complete AKA source checkout
- A configured Agent CLI, such as `claude`, `qodercli`, `codex`, `pi`, or `dsh`
- For real optimization runs: access to Agate, a target GPU worker, and the required framework environment

Light consists of three Python distributions that must come from the same source revision:

- `atrex-aka-framework`: the `aka` CLI and plugin framework
- `atrex-aka-legacy`: shared Task, Agent, and compatibility components required by Light
- `atrex-aka-light`: the Light Workflow, Runtime, observability, and reliable-execution implementation

The three wheels and the resource checkout passed through `--repo-root` must come from the same revision. The packages commonly all report version `0.2.0`, so rebuild and force-reinstall them after a source update; do not rely on the displayed package version alone.

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

The installer validates the checkout and interpreter, builds matching framework, legacy, and Light wheels, force-reinstalls them from the local wheel directory, and verifies both `python -m aka --help` and `python -m aka optimize --help`.

The current installer resolves only local wheels with `--force-reinstall --no-index`. Supply a complete wheel directory using `--wheel-dir`, including external runtime dependencies such as `packaging`; preinstalling dependencies alone does not satisfy a forced reinstall. The script's temporary directory contains only the three AKA wheels, so its default dependency bootstrap needs a separate fix. For example, download the external dependency wheels into the retained directory before running the installer.

If the environment cannot install the Python `build` frontend from a package index, install it beforehand and pass `--no-install-build-dependency`.

## Manual installation

These commands match the upstream installation flow:

```bash
cd /path/to/atrex-kernel-agent
python3 -m pip install build packaging
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

Do not skip reinstallation merely because the distributions still report version `0.2.0`. Do not replace code underneath a running campaign and assume its processes loaded the update; restart through the campaign's Supervisor or workload mechanism and verify the result.
