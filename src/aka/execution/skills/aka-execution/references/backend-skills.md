# Backend Skill references

## Agate

Find and read **`atrex-gpu-gateway`** in this optimization Agent's supplied Skill
catalog or workspace-local discovery roots. It owns client configuration,
authentication, Gateway operations, parameters, file transfer, and client limits.
Load only the references required by the request, such as custom SOL problems or
multi-GPU configuration. This Skill does not duplicate that manual.

The selected legacy plugin contributes the complete Gateway Skill tree when its
`gateway_skill_path` configuration names an existing resource directory. The
application installs it into this optimization workspace alongside `aka-execution`
and publishes both entry points in the Skill catalog and prompt instructions.
The default empty path contributes only `aka-execution`. If Gateway guidance is
absent, report the missing dependency. Existing fixed
requests can continue through their deterministic AKA adapter; newly constructed
Agate-specific options require the corresponding guidance.

Follow the original Skill's requirement to inspect the installed `agate --help`
and relevant subcommand help. Help runs on the host without submitting GPU work.
Installation, upgrades, or endpoint changes are separate actions, not automatic
prerequisites. Use only options supported by the matching AKA adapter.

Execution's Skill and the selected Gateway Skill retain separate resource
ownership. Install complete reference trees into the optimization Agent's
environment and expose their real entry points. The module does not rely on
personal global Skill directories on the developer's machine. Resource sources
must contain `SKILL.md` and its referenced files, with no symlinks.

To supply Gateway guidance, prepare an execution patch in the host environment:

```json
{
  "api_version": 1,
  "patches": [
    {
      "id": "executor",
      "config": {"gateway_skill_path": "/path/to/atrex-gpu-gateway"}
    }
  ]
}
```

Pass that file with `aka optimize --execution-patch /path/to/patch.json` or the
matching application's execution configuration. Use a resource path supplied
by the deployment. The selected Skill's files and source path are locked for
workspace recovery; restore those resources or use a new workspace if they
change.

## SSH and Local

The current Execution implementation supplies no separate SSH or direct Local
Skill. Use [Legacy requests](legacy-request.md) and the matching guidance
provided by this run. When a backend supplies its own Skill, resource selection
must publish its actual catalog name and entry point.

Local HTTP Gateway uses `atrex-gpu-gateway` guidance, subject to the local
service's supported requests and resource limits.
