# Resource publication and persistent Agent workspaces

Light uses persistent Agent-owned Git repositories. The public initializer is
`GitCandidateWorkspace.initialize_agent`; resuming it preserves HEAD, the index,
branches and uncommitted source. Context manifests, Episode worktrees and Git
checkpoint capture/restore are not part of the current Light workflow.

## Immutable resource publication

`aka.core.publication.publish_immutable(path, payload)` is the shared publication
primitive used by Resources content snapshots. It stages in the destination
directory, writes the complete bytes, sets mode `0444`, fsyncs and closes the file
before publishing with a non-replacing hard link. It then removes the staging
alias and fsyncs the directory. The completed object must have one link and the
expected bytes; a different existing object is never overwritten.

Closing before link/unlink avoids retaining an NFS `.nfs*` alias. The persistent
`.aka-publication.lock` serializes participating writers and is never unlinked.
Process exit releases the lock. Storage must support the required advisory locks
and file/directory fsync operations; this code does not bypass storage errors.

The reserved `.<destination>.aka-publish` staging name identifies one transaction:

| Interrupted state | Retry with the same payload |
| --- | --- |
| Partial staging write | Validate the ordinary single-link prefix, remove it and rewrite. |
| Complete staging file before publication | Verify and fsync the file, close it, then publish. |
| Destination and staging alias share two links | Verify bytes, modes and inode identity, fsync, close and remove only the staging alias. |
| Destination already published with one link | Verify and fsync the existing object, then reuse it. |

Unexpected aliases, extra links, symlinks, modes or payloads are errors. A rejected
transaction remains available for diagnosis. Read-only mode alone does not prove
that a previous file fsync completed. Identity and metadata are checked across
reads and fsync. Staging stays on the destination filesystem even when `TMPDIR`
is on a different filesystem.

`replace_bytes` provides the corresponding primitive for mutable management files.
It reconciles only the reserved staging file containing the requested bytes or a
prefix of them, then atomically replaces the destination. Its optional lock
directory can keep management locks outside an Agent source tree. Light does not
reintroduce the retired `.atrex_context.json` manifest to use this helper.

## Installed package assets

`GitCandidateWorkspace.install_files` uses `independent_resources` before installing
selected contributions. A wheel asset hardlinked to an installer cache is detached
only after checking its declared package ownership, RECORD SHA-256/size, ordinary
permissions and stable file identity. The copy preserves bytes and mode; installation
and cache files are not modified. Copies exist only for the duration of the call.

The default owner is `atrex-aka-framework`. Applications may explicitly extend the
helper's owner-to-resource-prefix mapping for separately packaged resources.
Undeclared files and text contributions retain their existing Workspace copy and
validation semantics; being below `site-packages` alone does not establish ownership.
Leaf symlinks are not trusted package assets. Installation parent aliases may be
resolved without weakening leaf validation.

`initialize_agent` likewise validates the workspace leaf and selected descendants
before resolving parent aliases such as macOS `/tmp` or `/var`. Existing repositories
are resumed without restoring seed source or changing Agent Git state.

## Scope and verification

The upstream publication and package-resource fixes are retained, but their callers
are adapted to Light's current `install_files` and `initialize_agent` interfaces.
The deleted `initialize_kernel`, `prepare_sol`, Context and checkpoint interfaces
remain deleted. No Framework Baseline, Journal, promotion or Supervisor Git workflow
is restored.

Local regression tests cover interrupted publication, same-content concurrency,
collision rejection, package RECORD verification, hardlinked resource installation,
parent aliases and preservation of existing Agent Git state. Filesystem tests do
not establish behavior on a real NFS mount or constitute a GPU optimization run;
those require separate integration validation.
