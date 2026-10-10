# Plugin framework interface

Core assembles plugins for Bootstrap, which invokes the profile-selected Startup. The
legacy profile serves the existing optimizer; `aka run` can select independent
implementations. This guide describes the framework shipped in this source tree. See
[application invocation](application-plugin-migration.md) for setup and the supported
checkout dependency.

## Responsibilities

- `aka.core`: declarations, JSON composition, dependencies, service Realms,
  contribution Scopes, effects, events and optional identity locks.
- `aka.contracts.application`: `ApplicationRequest` and synchronous
  `Application.run(request) -> int`; no dependency on Core or a backend.
- `aka.contracts.startup`: neutral `Invocation` and synchronous
  `Startup.run(invocation) -> int`, shared by legacy and future Workflow targets.
- `aka.bootstrap`: launch-profile resolution, startup target selection, invocation,
  continuation identity and disposal of the assembled composition. It has no fixed
  Application or Workflow dependency and contributes `aka run`.
- `aka.legacy.application`: optimizer variables, default profiles, construction
  plugin, checkout binding and legacy process adapters; contributes `aka optimize`.

Core imports neither the optimization application nor Execution. The existing
`plugin_runtime/` tool/skill registry is unrelated and remains untouched.

## Plugin declaration and lifetime

A plugin module declares a stable `name` matching its module name (underscores become
hyphens), a plain synchronous `apply(ctx, config)`, and optional `provide`, `inject`,
`optional_inject`, `Config`, `Defaults` and `interpolate` exports. `apply` constructs
and registers objects; importing or booting the default application plugin does not
start optimization.

`ctx.provide(name, value)` registers a module implementation. Consumers declare
injections and obtain the selected object via `ctx.get(name)`. Service tokens validate
declared metadata, not the complete implementation protocol. Bootstrap additionally
checks that `run` is callable, synchronous, and returns an integer exit code; behavior
still requires tests.

Required injections gate activation. Dependencies track **registration serials**, not
just provider identity; optional provider appearance and withdrawal also change the
dependency epoch. `Root.settle()` drains queued transitions. This supports Core-level
replacement but does not promise safe live replacement during optimization. Divergent
settling is bounded and reported.

Setup, config validators, event callbacks and cleanup are synchronous. Known async hooks
are rejected before invocation; synchronous wrappers returning awaitables are rejected
at the call site. Returned coroutine objects are closed rather than silently discarded.
Core does not create an event loop or await these hooks.

`ctx.effect(dispose)` and a cleanup callable returned by `apply` register owned effects.
Teardown unwinds effects in reverse order, is idempotent, and drains the stack even if
one cleanup fails. Ordinary cleanup errors are aggregated; `BaseException` is re-raised
after draining. Borrowing an object does not transfer ownership. Unregistered side
effects cannot be undone automatically. Local teardown does not imply cancellation of
detached or remote jobs.

## Composition and required checks

`aka.core.boot.compose()` loads a profile from a caller-selected directory. Bundles are
applied first, followed by profile patches, patch files and programmatic patches.
Patches replace the **whole config**, never deep-merge it. Dependency injection, not row
order, determines activation order.

`boot(composition, tokens=..., required_rows=..., required_services=...)` validates
configuration before applying plugins and rejects duplicate single providers. A required
row must be active even if it was disabled. An explicitly requested missing row fails; a
required service must exist in the root Realm. Removing a row removes its own
declaration, so Bootstrap independently requires the launch profile's target service
(named `startup` in the default legacy profile). There is no fallback to the default
when selection fails. Still-mounted children declared with `ctx.plugin(...,
required=True)` must also be active when boot finishes; a pending or failed required
child fails boot and disposes the composition. Explicitly disposed children are no
longer requirements.

Core receives a caller-supplied variable map; it has no application-variable allowlist
or hidden defaults. `${aka:name}` references that map; `${env:NAME}` reads the process
environment (an unset environment name yields an empty string). References are permitted
only in plugin-declared dotted field paths. Unknown variables, unsupported reference
syntax and references outside those paths fail. No expression evaluation or recursive
interpolation occurs. The legacy host owns optimization variable names and explicit
blanks; generic Bootstrap does not parse optimization argv or impose business variables.

Identity locks are opt-in through `boot(..., workspace=..., lock_mode=...)`. Bootstrap
does not enable Core identity locks by default. Its launch-selection record separately
captures Core/Bootstrap/Contracts and plugin/resource identity for child and recovery
reconstruction; business locks retain their own scope.

## Launch profiles and reconstructible compositions

A Bootstrap launch manifest references a sibling Core composition under `compositions/`.
The manifest's `target` selects a service using the neutral Startup protocol. Optional
registered business tokens, required services, resource identities, string bindings and
environment keys belong to launch selection; Core still owns bundle/patch merging and
declaration checks. See the [runnable profile example](application-plugin-migration.md#select-a-startup-implementation).

`freeze()` resolves allowed interpolation/defaults and isolates nested configs. After
assembly, `BootReport.verify_composition()` checks the recorded rows and configs before
invocation. Core's `Context.plugin()` can mount dynamic children, but this frozen
Bootstrap host rejects unrecorded children and setup config changes because their
identity cannot be reconstructed from the captured rows. Standalone Core hosts may
choose different policies.

Bootstrap supplies continuation environment overrides in `Invocation`. Targets forward
these to owned children and persist the selection before delayed restart. The legacy
adapter currently uses a durable `launch-selection.json` referenced by schema 4 restart
metadata. The [launch selection variable reference](application-plugin-migration.md#launch-selection-environment-variables)
defines the snapshot path, digest, paired transport, temporary/durable lifetime and
explicit `aka run --resume` entrypoint. See the [current recovery constraints](application-plugin-migration.md#current-child-and-recovery-behavior): selection-free main schema 3
records may use the built-in default legacy profile and be upgraded by the
validated recovery owner; new launch records remain strict.

## Isolation and events

Service **Realm** isolation controls visibility of selected implementations. Named
contribution **Scope** rules control contribution lookup. Neither provides per-plugin
event routing: events are shared inside a Root. Separate independent application
invocations use separate roots.

Supported event modes are `emit` and `waterfall`. Listeners use order and registration
order; all dispatch is synchronous:

| Mode | Behavior |
| --- | --- |
| `emit` | Notify all live observers; report ordinary observer exceptions |
| `waterfall` | Nested delegation through `next`; a listener may short-circuit, which is recorded |

Waterfall errors propagate. Waterfall delegates may be called once and only during their
dispatch. Monotonic waterfalls enforce declared shrink checks on both forwarding and
returning. Observer errors reach the Root's error reporting path; they are not silently
treated as successful results.

## DSH comparison and limits

AKA Core is inspired by Cordis plus loader mechanics, not by DSH's product
`packages/core` plugin collection. It is not a Cordis-compatible runtime. DSH's
asynchronous lifecycle, contextual event routing and event stop rules differ. No DSH
dependency is installed. Async lifecycle, contextual event isolation, automatic
discovery and application hot reload are not PR1 promises.

See [the application adapter guide](application-plugin-migration.md) for invocation,
package ownership and the legacy adapter's composition and recovery limits.

The optional [optimization dashboard plugin](optimization-dashboard.md) uses an independent
Core composition and Startup to observe existing campaign progress and token records.
