# Plugin fingerprint cost

Measured on 2026-09-22, Apple M5 / 16 GiB RAM, macOS 26.5.2 arm64, Python 3.14.6,
local checkout. Each case has one warmup followed by nine samples in the same process; Python
imports, temporary-workspace creation, service shutdown and cleanup are outside the timed region.
OS caches were not flushed. No Agent, model or GPU requests were made.

The real checkout contains one `gpu-wiki` plugin: 3 plugin files (1,698 bytes), 962 files under
`gpu-wiki/` (9,780,561 bytes), and one public Skill file (1,561 bytes). The optional
`internal_gpu_wiki/` is absent. Git metadata and Python/test caches are excluded. Total content
is 9,783,820 bytes (9.33 MiB); the largest file is 3,012,538 bytes. Catalog construction accounts
for 9,785,518 bytes including metadata parsing, and 1,170 entries/metadata reads.

## Results

All times are milliseconds. CPU is process CPU time, not percentage.

| Timed operation | Median wall | Wall min–max | Median CPU |
| --- | ---: | ---: | ---: |
| Full plugin catalog | 28.868 | 28.713–29.149 | 28.820 |
| Selected Wiki catalog reload | 28.815 | 28.544–29.154 | 28.750 |
| Supervisor construction + HTTP listener startup | 61.122 | 41.605–65.962 | 59.627 |
| Workspace linking, discover catalog | 39.071 | 38.617–40.418 | 39.011 |
| Workspace linking, reuse Supervisor catalog | 9.555 | 9.364–11.948 | 9.552 |

The selected-Wiki row measures catalog validation, not a Wiki query: the routed Wiki executor
does not reconstruct a registry per query. Generic tool subprocesses revalidate only their
selected plugin and declared dependencies; unrelated Wiki data is not rescanned. Campaign and
Episode linking use the reuse path. The Supervisor row measures a new local Campaign, not
complete application startup or remote-queue latency.

On this store the bounded content check costs tens of milliseconds. This supports retaining
content revalidation instead of trusting an mtime-only cache for this workload; it is not a
cold-cache, network-filesystem, large internal-store or production p99 claim. The catalog's
file/count/byte limits prevent unbounded growth; exceeding a limit fails with a diagnostic rather
than accepting a truncated hash. Same-size/same-mtime edits, oversized/growing files, FIFO/socket/
device inputs, symlinks, shared budgets and directory-depth limits have separate local regression
coverage. Test/benchmark scripts are not installed into Agent workspaces or committed as modules.

## Raw samples

Each array follows execution order after its warmup.

```json
{
  "catalog": {
    "wall_ms": [29.112, 29.075, 28.996, 28.713, 28.771, 28.841, 29.149, 28.852, 28.868],
    "cpu_ms": [29.055, 29.025, 28.955, 28.675, 28.724, 28.770, 29.080, 28.772, 28.820]
  },
  "selected_wiki": {
    "wall_ms": [29.154, 28.788, 28.544, 28.815, 28.878, 28.827, 28.896, 28.637, 28.704],
    "cpu_ms": [29.106, 28.701, 28.472, 28.750, 28.758, 28.751, 28.849, 28.562, 28.647]
  },
  "runtime_start": {
    "wall_ms": [62.038, 61.122, 62.613, 41.605, 65.962, 59.504, 60.639, 60.317, 61.764],
    "cpu_ms": [60.212, 59.627, 60.920, 40.286, 61.879, 57.767, 59.219, 58.263, 59.888]
  },
  "link_discover": {
    "wall_ms": [40.418, 39.071, 38.695, 38.621, 39.120, 39.556, 38.869, 39.109, 38.617],
    "cpu_ms": [40.365, 39.020, 38.512, 38.548, 39.011, 39.496, 38.763, 39.053, 38.506]
  },
  "link_reuse": {
    "wall_ms": [10.348, 9.512, 9.555, 11.948, 9.470, 9.364, 10.498, 11.602, 9.455],
    "cpu_ms": [10.337, 9.493, 9.552, 11.935, 9.428, 9.347, 10.486, 11.590, 9.447]
  }
}
```

## Reproduction

Run from the repository root using the desired Python environment. The Supervisor case requires
permission to bind a loopback listener. This uses only temporary workspaces and does not modify
an existing Campaign. Expect timings to vary with hardware, OS cache and catalog contents.

```bash
python3 - <<'PY'
import json, platform, statistics, sys, tempfile, time
from pathlib import Path
from orchestrator.plugins import PluginRegistry
from orchestrator.supervisor_runtime import RuntimeConfig, SupervisorRuntime
from orchestrator.workspace_runtime import link_runtime

print(platform.platform(), sys.version)
shared = PluginRegistry()
print("catalog read budget:", vars(shared.file_budget))
for mode in ("catalog", "selected_wiki", "runtime_start", "link_discover", "link_reuse"):
    wall, cpu = [], []
    for index in range(10):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory).resolve() / "workspace"
            workspace.mkdir()
            service = None
            started, used = time.perf_counter(), time.process_time()
            if mode == "catalog":
                PluginRegistry()
            elif mode == "selected_wiki":
                PluginRegistry(plugin_root=Path("plugins/gpu-wiki").absolute())
            elif mode == "runtime_start":
                service = SupervisorRuntime(RuntimeConfig("l20n", workspace=workspace))
            elif mode == "link_discover":
                link_runtime(workspace)
            else:
                link_runtime(workspace, plugin_registry=shared)
            wall_ms = 1000 * (time.perf_counter() - started)
            cpu_ms = 1000 * (time.process_time() - used)
            if service is not None:
                service.close()
            if index:
                wall.append(round(wall_ms, 3))
                cpu.append(round(cpu_ms, 3))
    print(mode, json.dumps(dict(wall_ms=wall, cpu_ms=cpu,
          median_wall_ms=statistics.median(wall), median_cpu_ms=statistics.median(cpu))))
PY
```
