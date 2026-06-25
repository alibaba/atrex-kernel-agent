## Synchronous Barriers

### bar/barrier — Intra-CTA Barrier

```
// Basic barrier (all threads)
bar.sync    0;                      // Barrier 0, wait for all threads in CTA
bar.sync    0, 128;                 // Barrier 0, wait for 128 threads

// Barrier arrival (no wait)
bar.arrive  1, 256;                 // Barrier 1, arrive, 256 threads participating

// Barrier with reduction
bar.red.and.pred  %p, 2, %q;       // Barrier 2 + AND reduction
bar.red.or.pred   %p, 3, %q;       // Barrier 3 + OR reduction
bar.red.popc.u32  %r, 4, %q;       // Barrier 4 + vote count
```

**Limit**: bar.sync barrier ID range 0–15 (up to 16 named barriers on SM70+).

### barrier (New Syntax for SM80+)

```
barrier.sync        0;                  // Wait for all
barrier.sync        0, 128;             // Wait for 128 threads
barrier.sync.aligned 0;                 // Aligned version

barrier.arrive      1, 64;             // Arrive only
barrier.arrive.aligned 1, 64;

// Producer-consumer pattern
// Thread group A:
barrier.arrive  5, 64;       // Group A arrives (64 threads)
// Thread group B:
barrier.sync    5, 128;      // Group B waits (A+B total 128 threads)
```

### barrier.cluster — Cluster Barrier (SM90+)

```
barrier.cluster.arrive;               // Current CTA arrives at cluster barrier
barrier.cluster.arrive.aligned;       // Aligned version
barrier.cluster.wait;                 // Wait for all CTAs in cluster to arrive

// Typical pattern
barrier.cluster.arrive;
// ... execute other work ...
barrier.cluster.wait;
```

---

## mbarrier — Asynchronous Barrier (SM80+)

mbarrier (Memory Barrier) supports **asynchronous** completion tracking and is the foundation for TMA and software pipelining.

### Lifecycle

```
Phase 0: mbarrier waits for N arrives or tx_count bytes
         → flips phase
Phase 1: mbarrier waits for N arrives or tx_count bytes
         → flips phase
... alternates
```

### Initialization

```
// Initialize mbarrier in shared memory
mbarrier.init.shared::cta.b64   [%mbar_ptr], %count;
// count = expected arrival count

// Initialize fence (ensure init visible to all threads)
fence.mbarrier_init.release.cluster;   // Cluster scope
fence.mbarrier_init.release.cta;       // CTA scope
```

### Arrive

```
// Basic arrival
mbarrier.arrive.shared::cta.b64           %state, [%mbar_ptr];

// Arrive and set transfer byte expectation
mbarrier.arrive.expect_tx.shared::cta.b64 %state, [%mbar_ptr], %tx_count;
// TMA automatically reduces tx_count when transfer completes

// Set transfer expectation (no arrive)
mbarrier.expect_tx.shared::cta.b64   [%mbar_ptr], %tx_count;

// Cross-CTA arrival (remote arrival within cluster)
mbarrier.arrive.shared::cluster.b64       %state, [%mbar_ptr];
// Pointer points to peer CTA's SMEM address
```

### Wait

```
// Blocking wait
mbarrier.try_wait.shared::cta.b64         %pred, [%mbar_ptr], %phase;
// pred = true means barrier has flipped to current phase

// Wait with timeout
mbarrier.try_wait.parity.shared::cta.b64  %pred, [%mbar_ptr], %phaseParity, %timeout;

// Non-blocking check
mbarrier.test_wait.shared::cta.b64        %pred, [%mbar_ptr], %phase;
```

### Typical mbarrier Usage Patterns

```
// === Producer threads ===
// 1. Set transfer expectation
mbarrier.expect_tx.shared::cta.b64   [%mbar_ptr], 4096;  // Expect 4KB

// 2. Initiate TMA copy (TMA hardware automatically arrives)
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [%smem_ptr], [%desc_ptr, {%x, %y}], [%mbar_ptr];

// === Consumer threads ===
// 3. Wait for data ready
mbarrier.try_wait.shared::cta.b64  %pred, [%mbar_ptr], %phase;
@!%pred bra WAIT_LOOP;

// 4. Use data
ld.shared.f32  %f, [%smem_ptr];
```

---

## Fence Instructions

### fence — Memory Ordering (SM70+)

```
// Acquire-Release Fence
fence.acq_rel.cta; // CTA range
fence.acq_rel.cluster; // Cluster range(SM90+)
fence.acq_rel.gpu; // GPU range
fence.acq_rel.sys; // range

// Fence
fence.sc.gpu; // GPU range
fence.sc.sys; // range
```### Proxy Fence

Proxy fence ensures visibility between memory operations through different **proxies** (paths):

```
// proxy(memorydifferent)
fence.proxy.alias;

// asynchronous proxy(cp.async complete)
fence.proxy.async; //
fence.proxy.async.global; // global memory
fence.proxy.async.shared::cta; // shared memory(CTA)
fence.proxy.async.shared::cluster;   // shared memory（cluster）

// Tensormap proxy
fence.proxy.tensormap::generic.acquire.cta [%desc_ptr], 128;
fence.proxy.tensormap::generic.release.cta [%desc_ptr], 128;
```

### membar (Old Interface)

```
membar.cta;        // ≈ fence.acq_rel.cta
membar.gl;         // ≈ fence.acq_rel.gpu
membar.sys;        // ≈ fence.acq_rel.sys
```

---

## cp.async — Asynchronous Copy

### cp.async (SM80+) — Global → Shared

```
// asynchronouscopy 4/8/16 bytes
cp.async.ca.shared.global    [%smem_ptr], [%gmem_ptr], 16;
cp.async.cg.shared.global    [%smem_ptr], [%gmem_ptr], 16;
// .ca = cache at all levels, .cg = cache at L2 only

// sizecopy( tile )
cp.async.ca.shared.global    [%smem_ptr], [%gmem_ptr], 16, %src_size;
// if src_size < 16,

// wait
cp.async.commit_group; // current
cp.async.wait_group 0; // waitcomplete
cp.async.wait_group 1; // wait ≤1 pending
cp.async.wait_all; // waitcomplete
```

### cp.async.bulk (SM90+) — Bulk Asynchronous Copy

```
// Global -> Shared
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
    [%smem_dst], [%gmem_src], %size, [%mbar_ptr];
// copy ≤ CTA sharedmemorysize

// Shared → Global
cp.async.bulk.global.shared::cta
    [%gmem_dst], [%smem_src], %size;

// completewait
cp.async.bulk.commit_group;
cp.async.bulk.wait_group  0;
cp.async.bulk.wait_group.read 1; // .read meansdataread
```

### cp.async.bulk.tensor — TMA (SM90+)

TMA (Tensor Memory Accelerator) describes multi-dimensional tensors via **descriptors**, completing the entire tile transfer in a single instruction:

```
// 2D TMA Load: Global → Shared
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [%smem_dst], [%tensor_desc, {%coord_x, %coord_y}], [%mbar_ptr];

// 3D TMA Load
cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes
    [%smem_dst], [%tensor_desc, {%x, %y, %z}], [%mbar_ptr];

// TMA Store: Shared → Global
cp.async.bulk.tensor.2d.global.shared::cta
    [%tensor_desc, {%x, %y}], [%smem_src];

// TMA Multicast Load (broadcast to multiple CTAs within cluster)
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
    [%smem_dst], [%tensor_desc, {%x, %y}], [%mbar_ptr], %ctamask;

// TMA Reduce Store
cp.async.bulk.tensor.2d.global.shared::cta.add
    [%tensor_desc, {%x, %y}], [%smem_src];
// .add/.min/.max/.inc/.dec/.and/.or/.xor
```

**Key TMA Features:**
- No per-element address calculation required
- Hardware automatically handles out-of-bounds checks (OOB zero-fill)
- Supports swizzle modes (hardware SMEM layout conversion)
- Completion notification via mbarrier
- Supports 1D–5D tensors

### TMA Descriptor

```
// Descriptor created on host side, contains:
// - base pointer
// - tensor shape (size per dimension)
// - tensor stride (stride per dimension)
// - element type
// - swizzle mode
// - OOB fill value

// Runtime descriptor update (SM90+)
tensormap.replace.tile.global_address.shared::cta.b1024   [%desc_ptr], %new_ptr;
tensormap.replace.tile.rank.shared::cta.b1024             [%desc_ptr], %new_rank;
tensormap.replace.tile.box_dim.shared::cta.b1024          [%desc_ptr], %dim_idx, %new_size;
tensormap.replace.tile.global_dim.shared::cta.b1024       [%desc_ptr], %dim_idx, %new_size;
tensormap.replace.tile.global_stride.shared::cta.b1024    [%desc_ptr], %dim_idx, %new_stride;
tensormap.replace.tile.element_stride.shared::cta.b1024   [%desc_ptr], %dim_idx, %new_stride;

// Descriptor prefetch
prefetch.tensormap   [%desc_ptr];
```## DSMEM — Distributed Shared Memory (SM90+)

CTAs within a cluster can directly read and write shared memory of other CTAs:

```
// Asynchronous store to remote CTA's SMEM
st.async.shared::cluster.mbarrier::complete_tx::bytes.f32
    [%remote_smem_ptr], %val, [%mbar_ptr];

// Bulk asynchronous copy SMEM → remote SMEM
cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
    [%dst_remote_smem], [%src_local_smem], %size, [%mbar_ptr];

// Address mapping: use mapa instruction to get remote SMEM address
mapa.shared::cluster.u32  %remote_ptr, %local_ptr, %target_cta_rank;
```

---

## ldmatrix / stmatrix (SM75+)

Used Expediently for loading/storing Tensor Core operands, with each thread holding a portion of the matrix:

```
// ldmatrix: load from SMEM to registers (warp collaboration)
ldmatrix.sync.aligned.m8n8.x1.shared.b16   {%r0}, [%sptr];         // 1 8x8 matrix
ldmatrix.sync.aligned.m8n8.x2.shared.b16   {%r0, %r1}, [%sptr];   // 2 8x8 matrices
ldmatrix.sync.aligned.m8n8.x4.shared.b16   {%r0,%r1,%r2,%r3}, [%sptr]; // 4 8x8 matrices

// With transpose
ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16  {%r0,%r1,%r2,%r3}, [%sptr];

// stmatrix (SM90+): store from registers to SMEM
stmatrix.sync.aligned.m8n8.x4.shared.b16  [%sptr], {%r0,%r1,%r2,%r3};
stmatrix.sync.aligned.m8n8.x4.trans.shared.b16  [%sptr], {%r0,%r1,%r2,%r3};
```

**Thread-to-Data Mapping:** 32 threads each hold different rows (ldmatrix.x4 loads a 4 × 8x8 = 32x8 matrix), each thread provides one SMEM address, and hardware automatically coalesces them into a contiguous read.

---

## Summary of Asynchronous Operation Completion Mechanisms

| Operation | Completion Method | Wait Method |
|------|---------|---------|
| `cp.async` | commit_group | `cp.async.wait_group N` |
| `cp.async.bulk` | commit_group | `cp.async.bulk.wait_group N` |
| `cp.async.bulk.tensor` (TMA) | mbarrier tx_count | `mbarrier.try_wait` |
| `st.async` | mbarrier tx_count | `mbarrier.try_wait` |
| `wgmma.mma_async` | commit_group | `wgmma.wait_group N` |
| `tcgen05.mma` | commit | `tcgen05.wait` |

---

## Synchronization Operation Selection Guide

| Scenario | Recommendation |
|------|------|
| Global synchronization within CTA | `barrier.sync 0` |
| Partial thread synchronization | `barrier.sync N, count` |
| Cluster synchronization | `barrier.cluster.arrive/wait` |
| TMA data ready | `mbarrier.try_wait` |
| Software pipeline | `mbarrier` + phase flip |
| Cross-CTA communication | `mbarrier` + `.shared::cluster` |
| Memory visibility | `fence.acq_rel.{scope}` |
| cp.async completion | `cp.async.wait_group` |
| Intra-warp synchronization | `bar.warp.sync mask` |
