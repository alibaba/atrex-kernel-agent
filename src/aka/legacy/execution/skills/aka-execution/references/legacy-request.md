# Legacy Execution requests

The `legacy` Execution provider runs the existing sandbox launcher. It supports
SSH, Agate Gateway, and direct HTTP transport through one provider. The selected
Execution module contributes this Skill to the optimization Agent's workspace
through the application's resource installer.
Inspect the actual installed `aka execution --help` before preparing a request.

## Standalone entry point

Use Python 3.10+, matching installed AKA packages, a matching AKA checkout, and
the original runtime dependencies. `--repo-root` must contain `tools/sandbox.py`;
`--workspace` identifies prepared execution inputs. Run the following on the host,
replacing paths and using the endpoint and hardware selected for this run:

```sh
aka execution --repo-root /path/to/matching-checkout \
  --workspace /path/to/prepared-workspace --hardware L20N \
  --gateway-profile prod --kind dev --timeout 60 \
  -- python3 gpu_smoke.py
```

`gpu_smoke.py` is a caller-prepared file, not an asset supplied by this Skill.

| Option | Current meaning |
| --- | --- |
| `--profile` | Execution profile; the only explicit choice is legacy. Omission inherits the selection |
| `--profiles-dir` / `--patch` | Profile root and Core patches; explicit patches append to inherited patches |
| `--gateway-profile` / `--url` / `--ssh` | Mutually exclusive endpoints; omission retains original environment / client resolution |
| `--hardware` | Required sandbox hardware token; the backend determines its validity |
| `--kind` | auto / run / profile / dev; auto retains original worker-command routing |
| `--timeout` | Original sandbox execution timeout, validated by the matching launcher |
| `--wall-timeout` | Host supervision limit; omission retains queue and cleanup allowances |
| `--sync` | Repeatable output synchronization paths; omission uses no-sync |
| `--preflight` | Original health check; currently requires SSH |
| argv after `--` | Prepared worker command; optional for preflight |

`AKA_EXECUTION_SELECTION` can carry profile, profiles_dir, and patches.
Invalid inherited selections fail rather than silently changing providers.
The CLI does not add arbitrary worker environment flags. The original launcher
and caller determine worker injection; host `os.environ` is not forwarded in full.

## Backend paths

**Gateway:** Use the selected profile / URL and existing Agate authentication.
Typed run / profile mappings and necessary dev fallbacks remain inside the
original sandbox. An explicit URL with no installed Agate client can use its
direct HTTP path.

**SSH:** Use `--ssh`, any required `--ssh-init`, and a valid `--health-command`.
The matching launcher reads existing `ATREX_SANDBOX_SSH_GPU` and
`ATREX_SANDBOX_SSH_RUNTIME_BINDS` settings; this CLI adds no corresponding flags.
The target requires OpenSSH/SCP, Bubblewrap, explicitly bound runtimes, and one
assigned NVIDIA GPU. Original distributed evaluation / profiling limits apply.

**Local HTTP:** Use the selected service URL and a hardware alias it accepts
(local is the default alias). `tools/local_gateway.py` serves this path and
currently rejects multi-GPU requests. Starting or changing a service is not an
automatic consequence of reading the Skill.

## Python interface and results

The public interface is `Executor.run(CommandSpec) -> JobResult`. CommandSpec
contains workspace, command, timeout, environment, hardware, artifacts,
operation, and optional cancellation / control. Legacy requires an application
adapter's private runtime binding; use the CLI or existing application entry
point rather than calling an unbound LegacyExecutor directly.

Legacy implements no DurableExecutor methods and rejects neutral budget / control
requests. Original wall timeout, host hooks, and cooperative cancellation remain
active in the compatibility path.

When results match, the application adapter preserves the original
CompletedProcess. The CLI writes its stdout / stderr and returns its returncode.
JobResult alone omits original args and a separately parsed remote exit code;
the application adapter supplies the original launcher args. Preserve the caller's
existing full result and artifacts when reporting execution outcomes.
