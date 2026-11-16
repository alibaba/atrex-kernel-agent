# Workspace resource publication and recovery

Workspace initialization retains immutable context independently of the input
installation. The retained object must remain a regular file with one link,
mode `0444`, and the expected bytes. Those requirements also apply when a
previous initialization is retried.

## Publication transaction

`aka.core.publication.publish_immutable(path, payload)` supplies the same
publication implementation to Candidate Workspace context and Resources
content objects. It stages in the destination directory, writes and flushes
the complete payload, sets mode `0444`, fsyncs, and **closes the descriptor**
before publishing with a non-replacing hard link. It removes the staging
entry and fsyncs the directory before verifying the final single-link object.
The destination is never replaced to resolve a content collision.

Closing first matters on NFS: unlinking an open file can retain a hidden
`.nfs*` alias and make the final file have two links. No retry or relaxation of
the retained-object link count is used to hide that condition. Staging uses
the destination filesystem regardless of `TMPDIR`; publication never requires
a cross-filesystem link.

The reserved staging name `.<destination-name>.aka-publish` expresses an
intent for that destination. Cooperating writers serialize through a
persistent `.aka-publication.lock`, opened without following links. The lock
file is never unlinked: replacing it could give two processes independent
locks. Kernel process termination releases the lock. NFS mounts must support
the regular-file advisory locking and directory fsync operations used by the
contract; failures are reported rather than bypassed. A mount configured with
local-only locks (for example NFS `nolock`) only serializes writers on the
same client. Cross-Task migration still requires the source process to stop
and release its writer lease before another client resumes.

The next call with the same requested bytes can reconcile these states:

| Interrupted operation | Retry behavior |
| --- | --- |
| Partial staging write | A single-link, ordinary staging file containing a prefix of the requested bytes can be discarded and rewritten. |
| Staging complete, before link | Reuse the complete readonly staging object. |
| Link complete, before staging unlink | Require exactly two links, matching bytes and modes, and the same inode for the destination and its reserved staging name; remove only that staging alias. |
| Staging unlink complete | Verify the existing canonical single-link object and reuse it. |

A different payload, unexpected mode, symlink, third link, or alias outside the
reserved transaction name is an error. Recovery does not scan for arbitrary
aliases or delete unknown files. Data from a rejected transaction is retained
for diagnosis.

`aka.core.publication.replace_bytes` supplies durable replacement for mutable
management content. The Workspace context manifest keeps its existing
`.atrex_context.json.tmp` staging name. A retry reconciles only a single-link,
ordinary staging file whose bytes are a prefix of the current requested
manifest, then writes the complete manifest and atomically replaces it. A
different transaction is preserved and reported. Manifest locking is kept in
Git management storage rather than the candidate source tree.

These are process-crash recovery rules. They do not make arbitrary concurrent
mutation by non-cooperating processes safe, or promise recovery after storage
corruption or unsupported filesystem durability semantics. A process killed
between creation and permission normalization under an owner-restricting
umask can leave an unrecognized mode; that state fails closed. Direct
projection-copy interruption is also outside these retained-publication
boundaries; this does not claim arbitrary workspace writes are transactional.

## Installed resources and workspace aliases

An installed wheel file may be hardlinked to an installer cache. A verified
wheel-owned **input** can be copied to independent storage before passing it
to the strict Workspace writer. This preserves the original bytes and modes
without changing the installation or cache. It is a different boundary from
retained and candidate files: an unknown caller input or an alias already in
the Workspace still fails the original validation.

The public materialization helper verifies declared distribution ownership,
wheel RECORD hash and size, ordinary modes, and a stable bounded read. It
does not treat every file under `site-packages` as trusted. Claude and Codex
use the same Workspace resource boundary; no installer flag or agent-specific
link-count exception is required.

## Existing workspaces

Successful existing manifests and immutable objects keep their existing
format, hashes and modes. A matching legacy manifest `.tmp` can be reconciled
on the public initialization retry. A readonly object carrying an old random
`.retain-*` alias is not automatically repaired: that name cannot prove the
new transaction relationship. It continues to report invalid retained
context, preserving the original files. This change does not authorize
editing frozen Campaign state or provide a Campaign-version migration API.

## Verification

From a source checkout:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_publication.py -v
```

With the built wheel installed, run the same test file in isolated mode:

```bash
python -I /path/to/checkout/tests/test_publication.py -v
```

The suite kills disposable child processes with actual `SIGKILL` at write,
fsync, link, unlink and manifest-replace boundaries. It retries the public
`GitCandidateWorkspace.initialize_kernel` entry point, checks retained bytes,
permissions and link counts, and covers collisions, unknown aliases,
symlinks, partial transactions, concurrent writers and a descriptor-lifetime
guard for NFS publication.

To run these same contracts on a real mounted NFS filesystem, set
`ATREX_TEST_NFS_ROOT` to an existing writable directory on that mount. The
suite creates and removes only its own temporary test directory:

```bash
ATREX_TEST_NFS_ROOT=/mounted/nfs/test-root \
  python -I /path/to/checkout/tests/test_publication.py -v
```

Passing the local suite or its descriptor guard is not evidence of real NFS
acceptance. It also does not substitute for optimization, measured candidate
results, full source export, cumulative/reset budgets or Campaign resume
acceptance. Those require their own integration environment and evidence.
