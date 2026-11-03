# Capacity ownership and external floors

One profile owns capacity control; instances within that profile compete for a shared lease.
Production and preproduction use the same Hologres operations event/lease tables and matching cluster configuration names.
Pointing at the same Ray address alone does not coordinate them.

## Cluster registration and initialization

Each gateway uses its own registered cluster list. The shared control key remains the configured `name`.
Operators must use the same name in production/preproduction for the same dashboard URL and different names for independent clusters.

An enabled local registration initializes profile ownership automatically; normal floor configuration needs no manual claim.
Initialization reads fresh backend constraints under the cluster lease and preserves complete external bundles, counts and selectors.
Existing gateway targets do not become permanent external floors. If the backend is unreadable, its session is unknown,
or constraints cannot be preserved completely, ownership is not acquired and constraints are not overwritten.
Another environment's ownership blocks enabling the registration; transfer explicitly or first remove the old registration.

Ownership constrains Rust routing, upload preselection, ordinary job admission, pool switching, staged GPU admission,
the SDK submission bridge and every capacity-write entry point. Instances in one environment share durable ownership,
GPU reservations and cluster-operation leases. Ownership snapshots refresh every 5 seconds and expire after 15 seconds.
Read failures do not extend authorization. Clusters without valid authorization stop new scheduling; existing jobs continue polling, cancellation and cleanup.
Single-environment deployments without the shared ledger use local configuration and provide no cross-environment exclusion.

External floors are separate from expiring manual floors and use Ray logical-GPU units.
For matching bundles/selectors, retain the larger count across sources; conservatively preserve incompatible shapes and accompanying resources such as CPU.
Native affinity or fallback constraints that cannot be replayed losslessly disable capacity writes. Clearing a manual floor does not clear the external floor.

## Production/preproduction transfer and deletion

A given `name` belongs to only one environment at a time; demand is not aggregated across environments.
Transfer first records a draining state. The old environment stops new routes and ordinary admission while existing executions/stages finish.
After old ownership snapshots and bounded in-flight admission expire, the old environment provides a fresh complete report from shared runtime state.
Only when active execution, cleanup and quarantine are all zero does ownership switch under the cluster lease.
The SDK rechecks current ownership before submission; its Rust-persisted registration participates in drain reports without competing for the capacity-write lease.
The drain safety window defaults to at least 90 seconds to cover old admission snapshots and bounded SDK calls.
Backend constraints and external floors are preserved.

Standalone commands can request transfer but cannot force takeover when the old environment is not upgraded, reports are unreadable or executions remain.
The target must have its own registration; transfer does not create configuration there. Return temporarily borrowed capacity through the same process.
While draining, cancel a transfer in the UI or with `cancel-transfer`. Publishing `enabled=true` during deletion drain cancels that deletion drain.

`manage_clusters delete` first publishes a disabled registration, then waits for execution/cleanup to drain.
Ownership is released after the shared coordinator confirms completion. A timeout leaves the registration disabled; investigate and retry.
`delete --force` skips waiting and immediately returns the disabled result; historical bindings remain for running jobs to finish.
Neither deletion mode destroys Ray, actively cancels jobs or directly clears backend resource constraints.

Claim/transfer/pause/resume retain BUC global-administrator authorization and revision checks.
Standalone commands require shared-ledger credentials. Registration performs normal initialization automatically;
`claim` remains for legacy administration and manual recovery. Capacity pause is independent of cluster ownership: pausing scaling does not change the owner.

## External emergency operations

Follow this sequence: **successfully pause control, modify Ray directly, then explicitly resume**.
Pause waits for operations already holding the cluster lease, preserves current backend constraints and does not require backend reachability.
Restarting the gateway while paused does not re-enable capacity writes.

If constraints are changed directly without pausing, the gateway reads complete constraints at its next actual write or steady-state verification.
Unknown drift within the same session pauses control instead of repairing it automatically.
After a normal backend session restart with empty constraints, registered floors may be replayed.
An external write of exactly the current target is indistinguishable from the gateway's write.

Resume preserves the original external floor unless the operator explicitly chooses to register current external constraints as the new floor.
It re-reads backend constraints and validates the control revision.

Ray's request API has no external fencing/CAS. Direct writes can race with in-flight gateway requests;
automatic drift detection does not replace pausing first. Old/new gateways sharing the ledger and cluster name coordinate rolling upgrades
through the compatibility lease below. External scripts that bypass this lease must still follow the pause procedure.

The server currently requires Ray SDK `>=2.56.1,<2.57`; the locked dependency and verified cluster version are `2.56.1`.
Native constraints are read inside the target cluster using that Ray version's generated GCS protocol.
When upgrading target Ray, also run the real-SDK selector round-trip check.
Missing symbols, protocol-read failures or constraints that cannot be fully replayed disable writes; failures are never treated as empty constraints.

On a test cluster with control paused and initially empty constraints, run:

```bash
python scripts/verify_ray_capacity_selectors.py --config-file operator.json --cluster gpu-pool
```

This diagnostic uses no GPU. It requests 0.125 CPU on already-online nodes to verify native protobuf operators/values for
single/multiple-value IN and NOT_IN selectors, then clears temporary constraints before exit.
It rejects initially nonempty constraints. Session changes or restoration failure require manual verification.
It creates no real GPU workload and does not replace scale-up/admission acceptance tests.

## Rolling upgrades and rollback

Compatibility covers old gateways using the same operations event table, lease table and `name` for capacity leases.
The new version waits for old operations to release their lease, then acquires the cluster-name lease used by the old version;
new instances serialize writes through a separate operation lease. Old business jobs continue, but later old capacity operations cannot acquire the old entry lease.
Production/preproduction still control their own clusters through durable ownership; compatibility leases do not replace unique registration and drain-before-transfer.

During initial handoff, after a process crash or when a write outcome is unknown, keep the old entry protected.
Wait at least 90 seconds and read the complete Ray job list; allow new writes only once all `agate-scale-` control jobs finish.
If the backend is unreadable or jobs remain, continue waiting; a timeout is not success.
If the old ledger still marks a write pending, an empty job list alone does not establish safety.
Automatic handoff is allowed only when complete receipt history proves later failures were rejected by ownership checks before backend calls,
there is a successful receipt for the current Ray session, and native constraints exactly match the original target.
Record a verification receipt before handoff without changing the original floor or expiry.
Timeouts, empty errors and other ambiguous failures retain protection until explicit completion evidence exists.
New control jobs also have execution deadlines and reject writes after excessive queueing/initialization.
Standalone pause/transfer commands only update the ledger; uncertain remote writes stay protected while background reconciliation continues.
Even after pause succeeds, confirm handoff completion in the operations UI before modifying Ray directly.

When idle, the compatibility lease defaults to 90 seconds and is renewed every 5 seconds by new instances owning the cluster.
Any surviving new instance can maintain protection. After the last exits, the lease safely expires and the old version can resume control without manual lock deletion.
Rollback neither clears native constraints nor changes historical manual floors or their expiries.

Protection does not expire automatically for unknown remote-write outcomes, paused/draining control, independent external floors,
or constraints containing multi-GPU bundles, accompanying resources such as CPU, or selectors.
Older versions cannot preserve these semantics. The UI reports pending verification or incompatible rollback.
Restore the new version to reconcile, finish cleanup or explicitly adjust control state, then reassess automatic expiry.
There is no force-unlock shortcut.

The operations snapshot exposes `capacity_control.compatibility.phase`, `reason` and `lease_until`.
`ready` means protection is active and safe automatic fallback is permitted;
`preparing`/`uncertain` means handoff/reconciliation is incomplete; `rollback_blocked` means the current state is incompatible with the old version.
This field describes capacity control only, not business-admission readiness.

## Standalone administration

Run these commands in an operations environment with the server package installed.
The configuration file must specify the current profile plus shared Hologres control-table configuration/credentials; never commit it.
These commands do not call gateway HTTP endpoints.

```bash
python -m app.manage_capacity status --config-file operator.json --cluster gpu-pool
python -m app.manage_capacity pause --config-file operator.json --cluster gpu-pool --revision CONTROL_REVISION --actor EMP_ID
python -m app.manage_capacity transfer --config-file operator.json --cluster gpu-pool --revision CONTROL_REVISION --target-profile pre --actor EMP_ID
python -m app.manage_capacity resume --config-file operator.json --cluster gpu-pool --revision CONTROL_REVISION --actor EMP_ID
```

Read `revision` from `status`. Normal registration initializes ownership automatically; control changes require the current revision.
After `transfer` returns a draining state, poll `status` for completion. Transfer does not automatically resume paused capacity control.
`status`, `pause` and `transfer` do not connect to runtime PostgreSQL, resolve the resource catalog or contact the backend.
Only `claim` and `resume` resolve the target cluster and read the backend. Commands probe existing control tables without creating them.

To adopt current constraints as the new floor after an emergency modification, explicitly run:

```bash
python -m app.manage_capacity resume --config-file operator.json --cluster gpu-pool --revision CONTROL_REVISION --adopt-external-floor --actor EMP_ID
```

The administrator UI uses the added `PUT /dashboard/api/ops/ray/{cluster_name}/capacity-control` endpoint.
Requests include `action` and `revision`; transfers add `target_profile`, and resume may include `adopt_external_floor`.
It retains BUC global-administrator authentication and the same-origin intent header; it is not an AK machine API.

## 0.14.17 preflight and historical-target migration

Freeze affected resource-catalog edits during migration. The new version automatically isolates old capacity writers; manually stopping old gateway writes is unnecessary.
Production/preproduction cannot simultaneously enable the same cluster name.
New instances receive traffic after runtime readiness without waiting for every cluster registration. `healthz` indicates process liveness only.
A single-pool failure blocks only that pool; the operations UI shows registration status.
Temporary ledger-read failures retain unexpired authorization without extending it. Unauthorized/expired clusters stop new scheduling;
capacity writes still require live ownership verification and a shared lease.

Check complete configurations on both sides. Repeat `--peer-config-file` for every other environment sharing the ledger.
The command resolves each environment's current PostgreSQL catalog. Read-only preflight changes no registration or backend constraint and exits 1 for blockers.

```bash
python -m app.manage_capacity preflight --config-file operator.json --peer-config-file peer.json
```

Resolve duplicate registrations first: stop new admission in the old environment, drain it, explicitly transfer, and disable its registration.
Migration cannot take over a cluster already owned under the new protocol.

For a historical target with no new registration, use the preflight `revision` and `catalog_digest`:

```bash
python -m app.manage_capacity migrate-legacy --config-file operator.json --peer-config-file peer.json \
  --cluster gpu-pool --revision LEGACY_REVISION --catalog-digest CATALOG_DIGEST \
  --actor EMP_ID
```

The command isolates old writers through the compatibility lease and waits for remote control jobs to finish.
Incomplete handoff returns `capacity_handoff_pending`; retry later. `--confirm-legacy-writers-stopped` is accepted but deprecated and no longer requires manual confirmation.
Under the shared operation lease, it rechecks both registrations/revisions and reads complete native constraints.
Ownership is written only when constraints match the known historical target. It invokes no capacity-write API,
changes neither the manual floor nor expiry, and does not convert that floor into a permanent external floor.
Constraint drift, revision changes, duplicate registration or unreadable backend reject migration.
Rerun preflight after migration, then validate real submission, execution and cleanup.

Registration refresh, critical control operations and observations have independent connection budgets.
Top-level `control_pools` in the operations snapshot reports usage, waits and connection-acquisition timeout counts;
cluster-level fields of the same name remain for compatibility. Authorization still refreshes every 5 seconds and expires after 15 seconds;
local contention does not extend it.

### Wave-by-wave rollout and cluster failure isolation

On first deployment or owner recovery, VIP readiness depends only on gateway runtime readiness, not successful registration of every cluster.
Unregistered, conflicting, draining or expired clusters stop new routing/admission while authorized clusters serve normally.
Even with no usable clusters, the gateway can serve queries and operations and returns its existing unavailable response for submissions without a backend.
Registration-summary `ready` means every enabled cluster may accept new jobs; it is an operations summary, not a VIP readiness condition.
Startup dependency initialization or Rust runtime recovery failures still prevent instance readiness.

1. Before each wave, rerun `preflight` against current production/preproduction catalogs, check duplicate registration and each cluster's compatibility handoff.
   It checks registration/historical targets, not backend health, and does not replace initialization acceptance.
   A pending migration, ownership conflict or backend failure may be isolated while rollout continues; do not enable that cluster's scheduling/writes because the rest is ready.
2. Unready clusters automatically stop new routes/admission and may be repaired separately after rollout.
   To explicitly drain or stop initialization attempts, record the current catalog revision and cluster configuration and change only its `enabled` to `false`.
   For database catalogs, use the following commands with the revision from this `show`; for inline catalogs, disable the cluster in its original configuration.
   Existing execution and cleanup continue. Draining may release ownership but does not destroy the backend or cancel jobs.

```bash
   AGATE_CONFIG_FILE=/secure/gateway.json python -m app.manage_clusters show
   AGATE_CONFIG_FILE=/secure/gateway.json python -m app.manage_clusters update \
     --name example-pool --set 'enabled=false' --expected-version CURRENT_VERSION \
     --author OPERATOR --reason "temporarily disable unavailable cluster before rollout"
   ```

3. Roll out one wave, confirm Rust runtime/VIP readiness and real job completion on retained serving clusters, then proceed.
   `/healthz` alone is insufficient. Registration-summary `ready=false` does not block rollout, but record unavailable clusters and business impact.
   Old instances yield capacity writes automatically through the compatibility lease while continuing business jobs; do not create manual rollout markers.
   If every instance is offline, management commands on a database-connected operations host can still repair the catalog.
4. After backend recovery, recheck both current catalogs and ownership, confirming no other environment enables or owns the cluster.
   Read the latest revision, restore only that cluster to `enabled=true`, wait for registration and submit a real verification job.
   Use the same `update` command with the latest CAS revision. Do not roll back the entire old catalog and overwrite intervening edits to other clusters.

The UI distinguishes registration incomplete from registration initialization failed. For failure, inspect registration logs;
for database connection contention, wait for retry and inspect connection-pool metrics.
Pool contention retains the `runtime_database_pool_busy` identifier/log prefix; existing matching rules need no rename for this upgrade.
