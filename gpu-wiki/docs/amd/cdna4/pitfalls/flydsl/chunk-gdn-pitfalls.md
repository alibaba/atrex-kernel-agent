# Pitfalls: FlyDSL Chunk-GDN on MI355X (gfx950)

Applicability: backend: flydsl; hardware: amd; topic: pitfalls

Pitfalls encountered during optimizing the Chunk-GDN 5-kernel pipeline. Accompanying optimization documentation:
`docs/amd/cdna4/ref-docs/flydsl/cdna4-chunk-gdn.md`

---

## 1. Silent Corruption Caused by a Single Large SmemPtr Memref

**Kernel**: fwd_h

**Trap**: To implement k-LDS double buffering (4 buffers), a natural approach is to allocate a single large memref covering all buffersmq, and select them using a dynamic offset:

```python
# ❌ This approach produces incorrect results
LDS_K_ELEMS = 64 * 72  # 4608 per buffer
lds_k_all = SmemPtr(lds_base_val, base_offset, T.bf16,
                    shape=(LDS_K_ELEMS * 4,)).get()  # One large memref

def k_lds_buf_offset(i_t_val, kc):
    buf = arith.remui(i_t_val, arith.constant(2, type=T.i32))
    base_idx = arith.addi(arith.muli(buf, arith.constant(2, type=T.i32)),
                          arith.constant(kc, type=T.i32))
    return arith.muli(base_idx, arith.constant(LDS_K_ELEMS, type=T.i32))

# Add dynamic offset during read/write
lds_k_off = k_lds_buf_offset(i_t_i32, kc) + dk * stride + t
vector.store(val, lds_k_all, [to_idx(lds_k_off)])
```

**Symptom**: chunk 0 is always correct, but chunk 1+ produces huge errors (max_rel > 100000). No crash or exception — pure silent data corruption.

**Reality**: Even if the dynamic offset is replaced with a compile-time constant (e.g., `kc * LDS_K_ELEMS`, with no ping-pong at all), as long as a single large-shape memref is used, the result is wrong. The same offset using **independent small memrefs** is completely correct.

**Why**: This is a bug in FlyDSL SmemPtr / MLIR memref lowering. Possible causes:
- The base pointer + offset calculation for large memrefs may produce integer overflow or incorrect address computation during LLVM lowering
- memref alias analysis may assume different regions of a large memref can alias, leading to incorrect load/store reordering
- LDS alignment constraints may not be satisfied with a large memref

**Fix**: Use multiple independent SmemPtrs, each with its own offset and shape:

```python
# ✅ Correct approach: 4 independent SmemPtrs
lds_k_kc0_buf0 = SmemPtr(lds_base_val, offset0, T.bf16, shape=(LDS_K_ELEMS,)).get()
lds_k_kc1_buf0 = SmemPtr(lds_base_val, offset1, T.bf16, shape=(LDS_K_ELEMS,)).get()
lds_k_kc0_buf1 = SmemPtr(lds_base_val, offset2, T.bf16, shape=(LDS_K_ELEMS,)).get()
lds_k_kc1_buf1 = SmemPtr(lds_base_val, offset3, T.bf16, shape=(LDS_K_ELEMS,)).get()
```

**Lesson**: In FlyDSL, never use a single large SmemPtr memref to simulate multiple logical buffers. Each independent LDS region should have its own SmemPtr instance.

---

## 2. Correct Usage of scf.IfOp Return Values

**Kernel**: fwd_h (double-buffering selection)

**Trap**: In FlyDSL, when you need to read data from different LDS buffers based on a runtime condition, it is easy to write the return value syntax of scf.IfOp incorrectly.

**Wrong** (non-existent high-level API):
```python
# ❌ scf.if_() / scf.else_() / scf.if_result() are not FlyDSL APIs
with scf.if_(condition, results=[v8bf16_type]):
    val = vector.load_op(...)
    scf.yield_([val])
with scf.else_():
    val = vector.load_op(...)
    scf.yield_([val])
result = scf.if_result()  # Does not exist
```

**Correct** (low-level MLIR Python bindings):
```python
# ✅ Use scf.IfOp + ir.InsertionPoint
read_if = scf.IfOp(condition, results_=[v8bf16_type], has_else=True)
with ir.InsertionPoint(read_if.then_block):
    a0 = vector.load_op(v8bf16_type, lds_buf0, [to_idx(offset)])
    scf.YieldOp([a0])
with ir.InsertionPoint(read_if.else_block):
    a1 = vector.load_op(v8bf16_type, lds_buf1, [to_idx(offset)])
    scf.YieldOp([a1])
result = read_if.results[0]  # SSA value

# IfOp without return value (side-effect only)
write_if = scf.IfOp(condition, results_=[], has_else=True)
with ir.InsertionPoint(write_if.then_block):
    vector.store(val, lds_buf0, [to_idx(offset)])
    scf.YieldOp([])
with ir.InsertionPoint(write_if.else_block):
    vector.store(val, lds_buf1, [to_idx(offset)])
    scf.YieldOp([])
```## aked_pw_fp16x4 When Both Operands Are atanh_pw_fp16x4

**Kernel**: fwd_h (V14)

**Trap**: During element-wise computation in Phase mask (e.g., \( \tanh(p * w + q) \)), when both operands of `__.tanh_pw_fp16x4!` are ````python
# ❌ May not exist
RocmBackend._compile_to_hsaco  # AttributeError
````, the result is **baked** with the input — meaning ````python
# ✅ patch the output string of pipeline_fragments
_orig = RocmBackend.pipeline_fragments
def _patched(self, *, compile_hints=None, **kw):
    if compile_hints is None: compile_hints = {}
    frags = _orig(self, compile_hints=compile_hints, **kw)
    return [f.replace('O=2', 'O=3') if 'rocdl-attach-target' in f else f for f in frags]
RocmBackend.pipeline_fragments = _patched
```` and ````python
# ❌ Add 4 v8bf16 iter_args
init_args = [zero_v4, zero_v4] + k_prefetch  # 2 h_acc + 4 k_vecs = 6 args

for i_t, inner_iter_args, loop_results in scf.for_(..., iter_args=init_args):
    h_acc = [inner_iter_args[0], inner_iter_args[1]]
    k_loaded = [inner_iter_args[2], inner_iter_args[3],
                inner_iter_args[4], inner_iter_args[5]]
    # ...
    k_next = load_k_global(next_i_t)
    yield h_acc + k_next  # 6 values carried across iterations
```` share the same `results_=[]`, and subsequent uses will read silently corrupted values equal to the input of tanh.

This occurs because FlyDSL’s element‑wise API has two modes: for immutable inputs it allocates a new register; for mutable inputs (baked) it reuses the register. When both ``results=[]`` and ``ir.InsertionPoint`` are ``if_op.results[0]``, FlyDSL treats the output register as baked.

**Fix**: Before `__.tanh_pw_fp16x4!` call `__.add_pw_fp16x4!(r, p, __.const(0.0))` – this creates a new `pipeline_fragments` backed ``O=2`` register, breaking the baking chain `.tanh(pw_r)`.
This ensures `pw_r` and `q` remain independent.

**Lesson**: When both operands of an element‑wise override call (`__.xxx_pw_...!`) are ``O=3``, always check whether the result is silently merged (baked) with an input register. If so, explicitly copy with `.add_pw_fp16x4!(result, input, __.const(0.0))`.

## 2. scf.IfOp Returns via block_sig_str Instead of Python Context Manager

**Kernel**: fwd_h (V14)

**Trap**: scf.IfOp in FlyDSL's scf dialect does **not** use Python's context‑manager with‑syntax to build branches. Instead, it needs to pass `block_sig_str` to specify the block signature (yield type), and then append blocks manually. If the context‑manager syntax `with scf.IfOp(...) as op:` is used, the block signature will be defaulted to void. Later , when `scf.YieldOp` is called with actual return values, the signature will be inconsistent and fail MLIR verification.

**Correct approach**:

`num_stages=2`

**Key points**:
- The parameter is ``scf.for_`` (with underscores), not ``iter_args``
- Must wrap the block content with ``scf.for_``
- Get the return value via ``iter_args``
- Even for uniform branches (all lanes take the same path), use scf.IfOp

**Lesson**: scf‑related operations in FlyDSL use the low‑level MLIR Python bindings (scf.IfOp, scf.YieldOp), not the context‑manager syntax.

---

## 3. LLVM O=3's Monkey‑Patch Approach Varies Across Versions

**Kernel**: All 5 kernels

**Trap**: Different versions of FlyDSL have different internal APIs for RocmBackend. Directly patching internal methods will fail.

**Older version** (certain releases):
`ds_read_tr16_b64`

**Current version** (stable):
```python
# ✅ patch the output string of pipeline_fragments
_original_pipeline_fragments = RocmBackend.pipeline_fragments

def patched_pipeline_fragments(self, *, compile_hints=None, **kwargs):
    if compile_hints is None:
        compile_hints = {}
    fragments = _original_pipeline_fragments(self, compile_hints=compile_hints, **kwargs)
    return [
        fragment.replace('O=2', 'O=3') if 'rocdl-attach-target' in fragment else fragment
        for fragment in fragments
    ]

RocmBackend.pipeline_fragments = patched_pipeline_fragments
```

**Lesson**: Patching `pipeline_fragments`'s string output is more stable than patching internal compilation methods. pipeline_fragments returns a list of MLIR pass strings; replacing `O=2` with `O=3` is the most reliable approach.

---

## 4. Pre‑loading Is Nearly Ineffective When barrier 3 Exists

**Kernel**: fwd_h

**Trap**: Issuing global load requests for w/v/g before barrier 1, expecting the data to arrive in registers while waiting at the barrier. Intuition suggests a clear benefit.

**Reality**:
- V5 (no pre‑load, 3 barriers): 3136 us
- V7 (with pre‑load, 3 barriers): 3158 us  ← almost no improvement!
- V11 (with pre‑load, 2 barriers): 2183 us  ← huge improvement

**Why**: barrier 3 serializes the entire loop body. Even if w/v/g data arrives early in registers, the next iteration must still wait for barrier 3. The bottleneck is barrier serialization, not load latency.

Only after removing barrier 3 does the scheduling window become large enough for LLVM to truly exploit the overlap of pre‑loaded data and Phase 5 MFMA to hide latency.

**Lesson**: Optimizations have **synergistic effects**. Pre‑loading and barrier removal individually yield limited gains, but combined they deliver significant improvement. The order of optimization also matters — first confirm that barrier removal is feasible (via double‑buffering), then add pre‑loading.

---

## 5. SW Pipelining k via iter_args Causes Regression

**Kernel**: fwd_h

**Trap**: Mimicking Triton's `num_stages=2` software pipeline by pre‑loading the next iteration's k during Phase 5emo, passing it to the next iteration via `scf.for_`'s `iter_args`.

```python
# ❌ Adds four v8bf16 iter_args
init_args = [zero_v4, zero_v4] + k_prefetch  # 2 h_acc + 4 k_vecs = 6 args

for i_t, inner_iter_args, loop_results in scf.for_(..., iter_args=init_args):
    h_acc = [inner_iter_args[0], inner_iter_args[1]]
    k_loaded = [
        inner_iter_args[2],
        inner_iter_args[3],
        inner_iter_args[4],
        inner_iter_args[5],
    ]
    # ...
    k_next = load_k_global(next_i_t)
    yield h_acc + k_next  # 6 values carried across iterations
```

**Result**: V8 = 3888 us, 24% slower than V5's 3136 us.

**Why**: FlyDSL's `scf.for_`'s `iter_args` are SSA block arguments in MLIR. Each iter_arg requires a full register copy (v_mov_b32), and 4 v8bf16 = 32 bf16 values = 16 VGPRs. The extra VGPR pressure and move instructions disrupt LLVM's scheduling, making it counterproductive.

**Lesson**: In FlyDSL, do not pass large amounts of intermediate data through iter_args. Every iter_arg incurs a VGPR cost. A better approach is LDS double‑buffering (writing to different LDS regions, isolated by barriers) instead of register passing.

---

## 6. ds_read_tr16_b64 Is Not Necessarily Faster Than Scatter Write

**Kernel**: fwd_h (V13, regressed)

**Trap**: gfx950 provides `ds_read_tr16_b64` hardware‑transposed LDS reads, which deliver significant gains in fwd_o and recompute_wu. The natural idea is to also use it for k reads in fwd_h.

**Reality**:
- V11 (col‑major scatter write + ds_read_b128): 2183 us
- V13 (row‑major ds_write_b128 + ds_read_tr16_b64): 2288 us ← **5% regression**

**Why**: The key difference lies in the position of the scatter operation relative to the barrier:
- **fwd_h's k scatter writes** are before barrier 1 → LLVM freely schedules them, fully overlapping with other instructions, **zero overhead**
- **ds_read_tr16_b64 reads** are after barrier 1 → on the critical path, each read result is the input Malagasy for the subsequent MFMAIn fwd_o/recompute_wu, the write is also on the critical path (no barrier isolation), so the scatter write overhead saved by ds_read_tr is a net benefit. But in fwd_h, the scatter write is already free.

**Lesson**: The benefit of `ds_read_tr16_b64` depends on whether the scatter operation it replaces is on the critical path. Criterion: is the scatter operation before the barrier (free) or after it (critical path)?

---

## 7. range_constexpr Large Loop Causes IR Explosion Under O=3

**Kernel**: Staging loops in all kernels

**Trap**: Using `range_constexpr(N)` loops for LDS staging. When N is large and O=3 is enabled, compile time and IR size explode.

```python
# ❌ N=128 range_constexpr → 128 complete unrolls → O=3 optimization extremely slow
for i in range_constexpr(128):
    val = buffer_ops.buffer_load(rsrc, offset + i * stride, ...)
    vector.store(val, lds, [to_idx(i)])
```

**Fix**: For staging loops with N > 8, use `scf.for_` instead of `range_constexpr`:

```python
# ✅ scf.for_ keeps IR compact, O=3 can still optimize loop body
for i_val, _, _ in scf.for_(c0, cN, c1, iter_args=[]):
    val = buffer_ops.buffer_load(rsrc, offset + i_val * stride, ...)
    vector.store(val, lds, [to_idx(i_val)])
    scf.YieldOp([])
```

**Lesson**: `range_constexpr` is suitable for small loops (MFMA tiling, ≤8 iterations). Staging loops typically have 64-128 iterations and must use `scf.for_`.

---

## 8. Terminator Conflict with Nested scf.for_ and iter_args

**Kernel**: Kernels that require nested loops

**Trap**: Nesting an `scf.for_` with `iter_args` inside another `scf.for_` triggers an MLIR verifier error.

```python
# ❌ Nested scf.for_ each yield → terminator conflict
for i, outer_args, _ in scf.for_(0, N, 1, iter_args=[init]):
    for j, inner_args, _ in scf.for_(0, M, 1, iter_args=[outer_args[0]]):
        updated = some_op(inner_args[0])
        scf.YieldOp([updated])  # inner yield
    scf.YieldOp([inner_result])  # outer yield — CONFLICT
```

**Symptom**: MLIR verification error about block terminator, or silent miscompilation.

**Fix**: Unroll the inner loop using `range_constexpr` (if M is small), or refactor into a single-level loop + manual index computation.

**Lesson**: FlyDSL's `scf.for_` macro has a bug in terminator handling when nested. Avoid nesting scf.for_ with iter_args.

---

## 9. Hardcoded Grid Dimensions Cause OOB Access at Large T

**Kernel**: cumsum

**Trap**: For simplicity, the grid's NT dimension was hardcoded to `MAX_NT = 1024` (corresponding to T≤65536). When T=262144, NT=4096, and excess blocks access out-of-bounds memory.

```python
# ❌ Hardcoded MAX_NT
MAX_NT = 1024
grid = (MAX_NT, H // 4, 1)  # When T>65K, blocks 1024-4095 access invalid data
```

**Symptom**: May not crash (GPU might silently clamp or return 0), but produces incorrect results. Increasing MAX_NT to 4096 can lead to genuine out-of-bounds access.

**Fix**: Pass NT as a dynamic parameter in the launch:

```python
# ✅ Dynamic NT
NT_val = (T + BT - 1) // BT
grid = (NT_val, H // 4, 1)
```

**Lesson**: Grid dimensions that depend on data size (e.g., NT = T/BT) should always be computed dynamically, not hardcoded to an upper bound. Hardcoding leads to either wasted CUs (small T) or out-of-bounds access (large T).

---

## 10. Grid Dimension Order Has a Huge Impact on L2 Cache Hit Rate

**Kernel**: fwd_o, recompute_wu

**Trap**: Setting the grid to `(NT, B*H, 1)` seems natural (time dimension outermost, head innermost), but performance is much worse than `(B*H, NT, 1)`.

**Reality** (fwd_o, T=65K):
- `(NT, B*H, 1)`: ~800 us
- `(B*H, NT, 1)`: ~638 us ← **20% faster**

**Why**: The GPU's CTA scheduler assigns CTA IDs linearly according to grid dimensions. `(B*H, NT, 1)` makes adjacent CTA IDs correspond to adjacent chunks of the same head, so they share q/k dataeg, leading to higher L2 cache hit rates. `(NT, B*H, 1)` makes adjacent CTAs process different heads, with completely different q/k data, causing severe cache conflicts.

**Note**: Not all kernels benefit from this. fwd_h uses `(V/BV, N*H)` because V dimension splitting is the primary source of parallelism.

**Lesson**: Grid dimension ordering should let adjacent CTAs share as much input data as possible. Typically, place batch*head on grid_x (slow-varying) and time/spatial tiling on grid_y.

---

## 11. Compile-Time Constants H_SIZE/Hg_SIZE Make TP Switching Cumbersome

**Kernel**: All 5 kernels

**Trap**: H_SIZE and Hg_SIZE are module-level global variables in all kernels, and are frozen as compile-time constants when captured by the `build_*()` closure. Directly changing the variable values does not trigger recompilation.**Symptom**: After modifying `H_SIZE`, the kernel still uses the old value. Or `torch.cuda.cache` skips recompilation and returns the old kernel.

**Fix**: Three-step switch:
1. Replace the constant values in all kernel source files
2. Delete the `__pycache__` directory
3. Restart the Python process (or `importlib.reload` in each kernel file)

**Lesson**: FlyDSL kernel compile-time constants are captured via Python closures. After modifying the source code, you must ensure recompilation. In the future, these should be changed to `tl.constexpr` compile-time parameters or launch parameters.

---

## Related docs

- [FlyDSL Chunk-GDN Optimization](../../ref-docs/flydsl/cdna4-chunk-gdn.md)
- [FlyDSL Flash Attention Pitfalls](../../../cdna3/mi308x/pitfalls/flydsl/flash-attn-pitfalls.md)
- [Software Pipelining](../../../common/kernel-opt/hands-on/software-pipelining.md)
- [LDS Bank Conflict Optimization](../../../common/kernel-opt/lds-bank-conflict-optimization.md)
- ds_read_tr / Hardware Transpose
