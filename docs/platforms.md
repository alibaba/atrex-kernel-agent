# Platform support and macOS migration

## Breaking change: the coordinator must run Linux

The simplified workflow requires **Linux + Bubblewrap on the coordinator**, the machine running
`orchestrator/optimize.py`, the Supervisor HTTP service, and the coding Agent. It is not sufficient
to have Linux only on the GPU worker. Native macOS campaign execution from earlier workflows is
not preserved in this branch.

| Execution environment | Normal Git-backed Campaign |
| --- | --- |
| Linux with `bwrap` and permitted user/mount/PID namespaces | Supported |
| macOS running the Supervisor and Agent inside a Lima Linux VM | Supported Linux execution path |
| Native macOS Supervisor/Agent, including with remote GPU execution | Not supported |
| Linux without usable Bubblewrap, including restricted containers | Not supported |
| `--agent-sandbox=none` | Trusted non-Git tool tests only; not a Campaign mode |

`auto` selects Bubblewrap on Linux; it does not allow unisolated Campaign fallback. `bwrap` makes
the same requirement explicit. The CLI checks the platform and executable before creating a
workspace, initializing submodules, or contacting the GPU service. Actual launch also requires
namespace permissions; merely finding the executable is not a successful isolation probe.

This requirement protects Supervisor Git metadata, private evaluation inputs, credentials, and
measurement records from the coding Agent. Moving those files outside its working directory or
using HTTP tools does not prevent an unsandboxed process from reading same-user host files.
Removing the Git check would weaken the boundary, not restore equivalent macOS isolation.

The Linux coordinator does **not** need a local GPU when jobs use a remote Gateway or SSH worker.
Install the selected coding CLI for the coordinator's Linux architecture, not the GPU's
architecture. A macOS terminal may launch the command through Lima/SSH, but the Supervisor and
coding Agent must execute together on Linux.

## macOS migration with Lima

### 1. Create or enter a Linux coordinator

On macOS, install Lima and create a dedicated Ubuntu VM if needed:

```bash
brew install lima
limactl start --name=aka --cpus=4 --memory=8 --mount-none template:ubuntu
limactl shell --workdir /tmp aka -- bash -l
```

An existing Ubuntu VM is also usable: substitute its instance name in `limactl shell`. The new-VM
example disables host-directory mounts; it does not expose the macOS Home or provider transcripts.
VM setup follows the official [Lima installation](https://lima-vm.io/docs/installation/),
[usage](https://lima-vm.io/docs/usage/), and [mount controls](https://lima-vm.io/docs/config/mount/).

### 2. Prepare Linux dependencies and check isolation

Run inside the VM, as the ordinary user who will run AKA:

```bash
cd "$HOME"
sudo apt-get update
sudo apt-get install -y bubblewrap git python3 python3-venv python3-pip ca-certificates openssh-client
python3 -m venv "$HOME/.venvs/aka"
. "$HOME/.venvs/aka/bin/activate"
python3 -m pip install torch
bwrap --unshare-user --unshare-pid --unshare-ipc --unshare-uts \
  --ro-bind / / --dev /dev --proc /proc --die-with-parent -- /usr/bin/true
```

Install the selected Linux coding CLI and, for Gateway execution, your deployment's Agate CLI in
this environment. Authenticate them inside the VM through their normal setup, then check:

```bash
command -v python3 git bwrap
# Replace claude with the configured coding CLI.
claude --version
agate --version
```

Do not reuse macOS `.venv`, native binaries, or npm global installations: their interpreter paths
and executable formats may be incompatible. Provider login, Gateway configuration, proxy/CA
settings, and SSH access must work from Linux. Prefer VM-local configuration over mounting the
entire host provider Home. Keep secrets in private configuration/environment files, not command
arguments or committed files. SSH-only deployments do not need the Agate CLI check above.

If the Bubblewrap probe fails with `Operation not permitted`, have the Linux administrator inspect
user namespaces, AppArmor/seccomp, and container policy. The probe must succeed as the Campaign
user; do not run the Agent as root or switch to `none` to evade it.

### 3. Install this branch and launch on Linux

Clone the repository containing the simplified changes at the revision you intend to run. Set
`AKA_REPOSITORY_URL` and `AKA_REVISION` in the VM first; `AKA_REVISION` should be a full commit ID
available from that repository. Do not assume an unrelated upstream branch contains these changes.

```bash
: "${AKA_REPOSITORY_URL:?Set the repository containing the simplified workflow}"
: "${AKA_REVISION:?Set the full commit ID to run}"
mkdir -p "$HOME/src"
git clone "$AKA_REPOSITORY_URL" "$HOME/src/atrex-kernel-agent"
cd "$HOME/src/atrex-kernel-agent"
git checkout --detach "$AKA_REVISION"
python3 orchestrator/optimize.py --help
```

Provide an evaluator-owned operator checkout inside Linux. Native Atrex-Bench operators require
their evaluator source tree, not just a copied operator directory. See the supported layouts in
[Quick Start](quickstart.md#1-clone-the-repository). Required submodules are checked by the normal
startup flow; repository credentials and network access must therefore be available in the VM.

For Gateway execution, configure Agate credentials in Linux and set `AKA_OP_DIR`, `AKA_PLATFORM`,
`AKA_GATEWAY_HARDWARE`, and `AGATE_URL` to your operator and deployment values:

```bash
: "${AKA_OP_DIR:?Set the Linux operator directory}"
: "${AKA_PLATFORM:?Set the target GPU product}"
: "${AKA_GATEWAY_HARDWARE:?Set the Gateway environment name}"
: "${AGATE_URL:?Set the Gateway service URL}"
python3 orchestrator/optimize.py \
  --op-dir "$AKA_OP_DIR" --platform "$AKA_PLATFORM" \
  --sandbox-hardware "$AKA_GATEWAY_HARDWARE" --sandbox-url "$AGATE_URL" \
  --framework Triton --agent-cli claude --agent-sandbox bwrap \
  --workspace "$HOME/aka-runs" --max-iters 3
```

Change the framework/backend to your task. This command launches a real Campaign and consumes
model/GPU resources; the earlier help and namespace checks do not. For SSH GPU execution use the
[SSH example](quickstart.md#isolated-openssh-gpu-host) from inside the VM instead of `--sandbox-url`.
The coordinator's Bubblewrap and the SSH worker's Bubblewrap are separate requirements.

### 4. Preserve existing runs before moving

Stop the old Campaign cleanly and keep a complete backup before changing any paths. Prefer a fresh
Linux Campaign workspace for the first run; a plain directory copy is not a supported cross-OS
resume migration. Git worktrees can contain absolute links to their shared Git directory, while
Campaign metadata, evaluator locations, and recovery commands can also retain old host paths.

To preserve evidence for inspection, archive the Campaign checkout, shared Git metadata, Episode
worktrees, and sibling `.atrex-supervisor-runtime/` tree together. Treat that archive as private;
do not place it in an Agent-visible directory. Copying only `kernel.py` or `memory/` is not a complete
recovery backup. Do not edit private audits or Git linkage to force a resume.

An existing Linux run whose original paths, evaluator inputs, and private state remain intact can
use normal same-command recovery. Relocating an old macOS run is a separate migration task; this
guide does not claim automatic recovery of relocated state. macOS can still edit source and inspect
exported results, while all active Campaign execution stays on the Linux coordinator.
