# Coalesced Memory Access and Load/Store Instruction Width Optimization

On the premise of ensuring coalesced access, maximize the instruction width for global memory and LDS. Applicable to all DSLs for CDNA3/CDNA4 (Gluon/FlyDSL/Triton/CK).

---

## Core Principles

**Coalesced access is the prerequisite, instruction width is the means**. If the addresses of the 64 threads within a wavefront are not contiguous, even if each thread uses `dwordx4`, the hardware cannot coalesce them into efficient memory transactions.

### Instruction Width Targets

| Operation | Optimal Instruction | Width | Suboptimal | Worst |
|------|---------|------|------|------|
| **Global load** | `buffer_load_dwordx4` | 128-bit | `dwordx2` (64-bit) | `dword` (32-bit) |
| **Global store** | `buffer_store_dwordx4` | 128-bit | `dwordx2` (64-bit) | `dword` (32-bit) |
| **LDS read** | `ds_read_b128` | 128-bit | `b64` (64-bit) | `b32` (32-bit) |
| **LDS write** | `ds_write_b128` | 128-bit | `b64` (64-bit) | `b32` (32-bit) |

> **Experience**: `dwordx4` can reduce the instruction count by 4× compared to `dword`, lowering instruction scheduling overhead and improving bandwidth utilization.

### Coalesced Access Conditions

The memory access addresses of the 64 threads in a wavefront must satisfy:
1. **Contiguity**: Addresses are arranged contiguously within the same cache line
2. **Alignment**: The starting address is aligned to an integer multiple of the access width
3. **Uniformity**: All threads access the same memory segment

---

## Diagnostic Methods

### Assembly-Level Inspection

Search for the following patterns in the assembly:

| Instruction Pattern | Issue | Optimization Direction |
|----------|------|---------|
| `buffer_load_dword` (non-`dwordx4`) | Insufficient global load width | Adjust layout parameters |
| `buffer_store_dword` (non-`dwordx4`) | Insufficient global store width | Adjust layout parameters |
| `ds_read_b32` (non-`b128`) | Insufficient LDS read width | Adjust shared layout |
| `ds_write_b32` (non-`b128`) | Insufficient LDS write width | Adjust shared layout |

### Performance Counters

| Counter | Description |
|--------|------|
| `TCP_TOTAL_READ` / `TCP_TOTAL_WRITE` | Total read/write transaction count |
| `TCP_UTCL1_REQUEST` | L1 cache request count |

If the transaction count is much higher than the theoretical optimum (= total bytes / 128), it indicates non-coalesced access.

---

## Common Causes and Fixes

### 1. Mismatch Between Thread-to-Address Mapping and Memory Layout

**Problem**: Threads are arranged along a non-contiguous dimension in memory, causing non-contiguous addresses within a wavefront.

**Fix**: Ensure the innermost dimension of thread mapping is consistent with the dimension of the tensor that has stride=1 in HBM.

```
example: row-major tensor (stride=1 columndimension)
 Correct: dimensionmappingcolumndimension -> wavefront accesscontiguous
 Wrong: dimensionmappingrowdimension -> contiguous, nonecoalesced
```

> The specific layout parameter adjustment methods vary across DSLs (e.g., `BlockedLayout.order` for Triton/Gluon, thread cluster configuration for CK). Refer to the respective DSL-specific documentation.

### 2. Per-Thread Contiguous Access Volume Too Small

**Problem**: Each thread loads only 1-2 elements in the contiguous dimension, insufficient to fill dwordx4.

**Fix**: Increase the number of elements each thread loads in the contiguous dimension so that a single load reaches 128-bit:

| Data Type | Element Size | Consecutive Elements Needed |
|---------|---------|--------------------------------------|
| FP32 | 4 bytes | ≥ 4 |
| BF16/FP16 | 2 bytes | ≥ 8 |
| FP8/INT8 | 1 byte | ≥ 16 |

> The specific parameter names vary across DSLs (e.g., `size_per_thread` for Triton/Gluon, `ScalarPerVector` for CK). Refer to the respective DSL-specific documentation.

### 3. Matrix Dimensions Not Aligned

**Problem**: M/N/K dimensions are not integer multiples of the load width, so wide instructions cannot be emitted at the boundaries.

**Fix**:
- Pad inputs to alignment boundaries
- Or use mask + narrow load only in boundary regions

---

## Relationship with Other Optimizations

- **LDS bank conflict**: Coalesced access addresses global memory efficiency. LDS bank conflicts require separate swizzle-based elimination. See [LDS Bank Conflict Optimization](lds-bank-conflict-optimization.md) for details.
- **Scratch overflow**: Increasing `size_per_thread` increases VGPR usage and may cause register spilling. See [Eliminating Scratch Operations](scratch-elimination-vgpr-spill.md) for details.
- **Occupancy**: Increased VGPR usage also reduces occupancy. See [Occupancy Optimization](occupancy-optimization.md) for details.

---

## Related Documentation

- **Hardware Specifications**:
- **Profiling**: [Profiling Tools Overview](profiling-tools-overview.md) — Using rocprof to obtain performance counters
