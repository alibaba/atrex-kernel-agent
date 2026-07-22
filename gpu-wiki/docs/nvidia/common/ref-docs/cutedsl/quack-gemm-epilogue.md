# QuACK GEMM and Composable Epilogue System

QuACK (by Tri Dao) is a high-performance GEMM library built on CuTeDSL, with the following core features:
- **Multi-architecture unified interface**: The same `gemm()` API automatically dispatches to SM90 (Hopper) / SM100 (Blackwell) / SM120 (Blackwell GeForce)
- **Composable Epilogue**: Componentized epilogue operations (bias, activation, normalization, reduction) via `EpiOp`, freely composable into fused kernels
- **High-level fusion operations**: GEMM+Activation, GEMM+Norm+Activation, complete MLP blocks, all with PyTorch autograd integration
- **JIT compilation cache**: Compiled via TVM-FFI, `@jit_cache` ensures each configuration is compiled only once

## Part 1: Multi-architecture GEMM

### 1. GEMM Dispatch Mechanism

QuACK obtains the GPU's SM version number via `get_device_capacity()`, then uses a simple dictionary to dispatch to the corresponding GEMM implementation class:

```python
# gemm.py
sm_to_cls = {
    9: GemmDefaultSm90,   # SM90 Hopper (H100/H20)
    10: GemmDefaultSm100,  # SM100 Blackwell (B200)
    11: GemmDefaultSm100,  # SM110 also uses SM100 path
    12: GemmDefaultSm120,  # SM120 Blackwell GeForce (B580)
}
GemmCls = sm_to_cls[device_capacity[0]]
```

The `gemm()` function interface is unified, with key parameters including:
- `tile_M, tile_N`: CTA tile size
- `cluster_M, cluster_N`: Cluster shape (for TMA multicast)
- `pingpong`: Whether to enable pingpong scheduling (2 warp groups alternating tile processing)
- `persistent`: Whether to enable persistent tile scheduling
- `is_dynamic_persistent`: Dynamic persistence (SM90 requires GMEM semaphore)
- `rowvec_bias, colvec_bias`: Optional bias vector
- `alpha, beta`: Scalar or device pointer
- `cu_seqlens_m, cu_seqlens_k`: Variable-length sequence support
- `rounding_mode`: RN (default) or RS (stochastic rounding, SM100 only)

Compiled kernels are cached via the `@jit_cache` decorator, with the key being all combinations of parameters that affect code generation (dtype, layout, tile shape, cluster shape, etc.).

### 2. SM90 Hopper GEMM (`GemmSm90`)

SM90 GEMM is the base class of the entire architecture, from which SM100 and SM120 both inherit.

**WGMMA (Warp-Group MMA)**

SM90 uses a 128-thread warp group to collaboratively execute MMA:

```python
# sm90_utils.py
def gemm(tiled_mma, acc, tCrA, tCrB, zero_init=False, wg_wait=0):
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    if wg_wait >= 0:
        warpgroup.wait_group(wg_wait)
```

- Both A and B are read directly from SMEM (no explicit SMEM->RMEM copy required)
- Accumulators are stored in registers
- Supports FP16, BF16, FP8 (e4m3fn, e5m2)

**TMA Asynchronous Loading and Multicast**

- Uses TMA (Tensor Memory Access) to asynchronously load A and B from GMEM to SMEM
- Intra-cluster multicast: A tensor is broadcast along the N cluster dimension, B tensor is broadcast along the M cluster dimension
- `cpasync.CopyBulkTensorTileG2SMulticastOp()` is used for multicast mode

**Warp Specialization Architecture**

SM90 GEMM uses warp specialization to divide threads into two types of roles:

| Role | Warp Range | Register Budget | Responsibility |
|------|----------|-----------|------|
| MMA warps | `0..ab_load_warp_id-1` | `num_regs_mma` (224-240) | `setmaxregister_increase`, execute WGMMA |
| Load warp(s) | `ab_load_warp_id..` | `num_regs_load` (24-40) | `setmaxregister_decrease`, TMA load A/B |

```python
# kernel entry warp
if warp_idx >= self.ab_load_warp_id:
    cute.arch.setmaxregister_decrease(self.num_regs_load)
 # TMA load mainloop ...
if warp_idx < self.ab_load_warp_id:
    cute.arch.setmaxregister_increase(self.num_regs_mma)
 # MMA + Epilogue mainloop ...
```

**Pingpong Scheduling**

When `pingpong=True` is enabled, 2 warp groups alternate processing different output tiles:- WG0 and WG1 each take 128 threads, synchronized via `NamedBarrier`
- When WG0 performs MMA, WG1 performs epilogue (or vice versa)
- Pipeline state must use `advance_iters()` to skip the other party's tile
- Constraint: `tile_M` is limited to 64/128/192, `atom_layout = (1,1,1)`

```python
# Pingpong synchronous
self.pingpong_barrier_sync(warp_group_idx, stage="mma") # wait
self.pingpong_barrier_arrive(1 - warp_group_idx, stage="mma") # notify
```

**Pipeline Configuration**

```python
# automaticcompute pipeline
ab_stage = remaining_smem_bytes // ab_bytes_per_stage # 4-7
epi_stage = 4 if epi_tile_N <= 16 else 2
sched_stage = 2 if pingpong else 1
```

**FP8 Slow Accumulation Mode**

For FP8 inputs, the `fp8_slow_accum` mode can be selected: perform one FP32 reduction after each K-tile to avoid precision loss from FP16 accumulator.

### 3. SM100 Blackwell GEMM (`GemmSm100`)

SM100 GEMM is a comprehensive upgrade over SM90, introducing TMEM (Tensor Memory) and CLC (Cooperative Launch Control).

**tcgen05 UMMA**

SM100 uses the `tcgen05.mma` instruction to replace WGMMA:
- A and B are read from SMEM
- Accumulator is written to **TMEM** (Tensor Memory, a dedicated on-chip memory), rather than registers
- Supports 2-CTA mode (`use_2cta_instrs`), where two CTAs collaborate to execute larger tiles

```python
# 2-CTA
self.use_2cta_instrs = cluster_shape_mnk[0] == 2 and mma_tiler_mn[0] in (256,)
self.cta_group = tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
```

**CLC Dynamic Persistent Scheduling**

SM100 uses CLC (Cooperative Launch Control) by default for persistent scheduling, replacing SM90's static/dynamic modes:

```python
# tile_scheduler.py
class PersistenceMode(IntEnum):
    NONE = 0
    STATIC = 1    # SM90: grid = max_active_clusters
    DYNAMIC = 2   # SM90: atomicInc semaphore
    CLC = 3       # SM100: hardware-assisted scheduling
```

CLC uses the hardware mbarrier + `clc_response` mechanism to obtain the next work tile, avoiding the overhead of global atomic operations.

**5-Warp Role Architecture**

SM100 GEMM uses finer-grained warp specialization:

| Role | Warp ID | Responsibility |
|------|---------|------|
| Epilogue warps | 0-3 (4 warps) | TMEM→RMEM load + data type conversion + TMA store |
| MMA warp | 4 | Issue tcgen05.mma instructions |
| AB Load warp | 5 | TMA load A/B (+SFA/SFB for blockscaled) |
| Epi Load warp | 6 | TMA load C matrix |
| Scheduler warp | 7 | CLC tile scheduling |
| A Prefetch warp | 8 (gather_A only) | Prefetch indices for A |

```python
self.epilog_warp_id = tuple(range(num_epi_warps))  # (0, 1, 2, 3)
self.mma_warp_id = len(self.epilog_warp_id)         # 4
self.ab_load_warp_id = self.mma_warp_id + 1          # 5
self.epi_load_warp_id = self.ab_load_warp_id + 1     # 6 (+ num_ab_load_warps)
self.scheduler_warp_id = self.epi_load_warp_id + 1   # 7
```

**TMEM Accumulator**

SM100 MMA results are written directly to TMEM. During the epilogue stage, TMEM data is loaded into registers via `tcgen05.ld`:

```python
# TMEM
tmem = cutlass.utils.TmemAllocator(
    storage.tmem_holding_buf,
    storage.tmem_dealloc_mbar_ptr,
    self.num_tmem_alloc_cols,
    barrier_for_retrieve=tmem_alloc_barrier,
    allocator_warp_id=self.epilog_warp_id[0],
    is_two_cta=use_2cta_instrs,
)

# Epilogue use tiled_copy_t2r (TMEM -> RMEM)
tiled_copy_t2r = tcgen05.make_tmem_copy(...)
```

**Register Spilling Management**

When `gather_A=True`, SM100 uses 3 warp groups (12 warps), and the register budget is tight (504 / 3 = 168 per thread). QuACK reallocates registers via `setmaxregister_increase/decrease`:

```python
# gemm_sm100.py
# Epilogue warps (WG0): requiresregister compute-heavy
cute.arch.setmaxregister_increase(self.num_regs_epi)    # 256

# warps (WG1, WG2): TMEM/TMA, requiresregister
cute.arch.setmaxregister_decrease(self.num_regs_other)  # 120

# check: 256 + 120 + 120 = 496 <= 504
```**Key Constraints**

- **All** warps in each WG must call `setmaxregister_decrease`; otherwise, the corresponding `increase` will deadlock.
- The value must be a multiple of 8.
- The total budget must not exceed `max_regs_per_thread * num_warp_groups`.

See [Warp Specialization](../warp-specialization-design-principles.md) for more on warp specialization design.

**Block-Scaled GEMM**

SM100 supports block-scaled GEMM in MX (Microscaling) format (FP4/FP8 + per-block scale factors):

```python
# blockscaled_gemm_utils.py / mx_utils.py
GemmCls = GemmDefaultSm100 # sf_vec_size parameter
# load SFA/SFB (scale factor) tensors
tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(...)
tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(...)
```

- `sf_vec_size`: the number of elements covered by each scale factor.
- Scale factors use the `Float8E8M0FNU` type.
- Additional SMEM and TMA descriptors are required for SFA/SFB.

### 4. SM120 Blackwell GeForce GEMM (`GemmSm120`)

SM120 is an implementation designed for consumer GPUs (e.g., B580), using warp-level MMA instead of warp-group MMA.

**Key Differences**

| Feature | SM90 (Hopper) | SM100 (Blackwell) | SM120 (GeForce) |
|------|-------------|------------------|----------------|
| MMA Instruction | WGMMA (128 threads) | tcgen05/UMMA (128 threads) | `MmaF16BF16Op` (32 threads) |
| A/B Read | Directly from SMEM | Directly from SMEM | Requires `ldmatrix` SMEM->RMEM |
| Accumulator | Register | TMEM | Register |
| FP8 | Supported | Supported | **Not supported** |
| Atom Layout | (1-2, 1-2, 1) | N/A | (4, 2, 1) or (2, 2, 1) pingpong |
| Thread Configuration | warp group = 128 threads | 5-role warps | num_mma_warps + 1 DMA warp |

**Warp-Level MMA and ldmatrix**

SM120 uses `MmaF16BF16Op` (16x8x16 warp-level MMA), which requires an explicit SMEM->RMEM copy before MMA:

```python
# gemm_sm120.py
def _setup_tiled_mma(self):
    op = warp.MmaF16BF16Op(self.a_dtype, self.acc_dtype, self.mma_inst_mnk)
    tC = cute.make_layout(self.atom_layout_mnk)
    self.tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

# MMA mainloop: ldmatrix + gemm
def mma(self, ...):
    # 1. SMEM -> RMEM via ldmatrix
    load_sA(tCsA_p[None, None, k], tCrA_copy_view[None, None, k])
    load_sB(tCsB_p[None, None, k], tCrB_copy_view[None, None, k])
    # 2. Warp-level MMA
    cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
```

- `ldmatrix` is a warp-level SMEM load operation (`LdMatrix8x8x16bOp`).
- Each K-tile contains multiple k-blocks, with load and compute alternating.
- The epilogue flow reuses the SM90 implementation (inherits `GemmSm90.epilogue`).

See [CUTLASS 3.x Architecture](cutlass-3x-architecture.md) for CuTeDSL layout algebra fundamentals.

## Part 2: Composable Epilogue System

### 5. Epilogue Operations (`epi_ops.py`)

QuACK abstracts epilogue operations into the `EpiOp` base class  Each op encapsulates the full lifecycle of a tensor operation:

```python
class EpiOp:
 def smem_bytes(self, arg_tensor, cta_tile_shape_mnk, epi_tile): ... # SMEM
 def smem_struct_field(self, gemm, params): ... # SMEM struct
 def begin(self, gemm, param, smem_tensor, ctx): ... # tile
 def begin_loop(self, gemm, state, epi_coord): ... # subtile
 def end(self, gemm, param, state, ...): ... # tile reduction/write
```

**Built-in EpiOp Types**

| Op Type | Purpose | SMEM | Description |
|---------|------|------|------|
| `Scalar` | alpha/beta scalar or device pointer | None | `load_scalar_or_pointer` |
| `RowVecLoad` | Row vector bias (N,) | tile_N bytes | Loaded via cp_async, broadcast along M with stride (0,1) |
| `ColVecLoad` | Column vector bias (M,) | tile_M bytes | Loaded via cp_async, broadcast along N with stride (1,0) |
| `ColVecReduce` | Column vector reduction | None (register) | Accumulate across N subtiles, warp shuffle reduction |
| `TileStore` | TMA write-out of extra tensor (e.g., postact) | epi_tile bytes | Used for activation output |**RowVecLoad Example Flow**

```
begin():    cp_async load bias[tile_N] from GMEM -> SMEM
 partition_for_epilogue -> tDsV ( subtile )
begin_loop(): SMEM -> RMEM copy (tDsV_cur -> tDrV)
              cast to acc_dtype -> tDrV_cvt
```

**ColVecLoad Optimization**

The column vector remains unchanged for the same M row during N-major subtile traversal, so the SMEM→RMEM copy is only performed at `epi_n == 0`:

```python
def begin_loop(self, gemm, state, epi_coord):
    epi_n = epi_coord[1]
 if epi_n == 0: # N subtile copy
        cute.autovec_copy(tDsV_cur, tDrV)
        tDrV_cvt.store(tDrV.load().to(gemm.acc_dtype))
    return tDrV_cvt
```

**ColVecReduce**

Used for column vector reduction in the backward pass (e.g., dnorm_weight of dgated kernel):

```python
# register subtile
tDrReduce.fill(0.0)
# epi_visit_subtile : colvec_reduce_accumulate(tDrReduce, input)

# epi_end : warp shuffle reduction + direct gmem write
for i in range(size):
    tDrReduce[i] = cute.arch.warp_reduction(tDrReduce[i], operator.add, threads_in_group=lanes_in_N)
# directwrite GMEM(warps_in_N == 1, does not require inter-warp reduction)
```

### 6. Epilogue Composition (`epi_composable.py`)

`ComposableEpiMixin` is the core of the composition mechanism. The op list is declared via the class variable `_epi_ops`:

```python
class GemmDefaultEpiMixin(ComposableEpiMixin):
    _epi_ops = (
        Scalar("alpha"),
        Scalar("beta"),
        RowVecLoad("mRowVecBroadcast"),
        ColVecLoad("mColVecBroadcast"),
        Scalar("sr_seed", dtype=Int32),
    )
```

Mixin automatically generates the following methods:

| Method | Automatic Behavior |
|------|---------|
| `epi_smem_bytes_per_stage` | Aggregates `smem_bytes()` of all ops |
| `epi_get_smem_struct` | Generates `@cute.struct` containing fields of all ops |
| `epi_get_smem_tensors` | Extracts SMEM tensors of all ops from storage |
| `epi_begin` | Calls `begin()` of each op in sequence, returns `{name: state}` dictionary |
| `epi_begin_loop` | Calls `begin_loop()` of each op in sequence, returns `{name: value}` dictionary |
| `epi_end` | Calls `end()` of each op in sequence |
| `EpilogueParams` | Automatically generates dataclass from `param_fields()` |

**Extending Epilogue**

Adding a new operation only requires extending the `_epi_ops` tuple:

```python
class GemmActMixin(GemmDefaultEpiMixin):
    _epi_ops = (*GemmDefaultEpiMixin._epi_ops, TileStore("mPostAct"))
    _extra_param_fields = (("act_fn", cutlass.Constexpr, None),)
```

The `epi_visit_subtile` method must be implemented manually, as it defines the specific mathematical logic of the operation:

```python
def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC):
 alpha = epi_loop_tensors["alpha"] # dict by
    tDrRowVec = epi_loop_tensors["mRowVecBroadcast"]
    tDrColVec = epi_loop_tensors["mColVecBroadcast"]
 # definition compute logical...
```

## Part 3: Fused High-Level Operations

### 7. GEMM + Activation (`gemm_act.py`, `gemm_dact.py`)

**Forward: GemmActMixin**

Apply activation directly in the GEMM epilogue, avoiding extra kernel launches and GMEM round-trips:

```python
class GemmActMixin(GemmDefaultEpiMixin):
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC):
 # 1. standard epilogue (alpha, beta, bias)
        GemmDefaultEpiMixin.epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC)
        # 2. Apply activation in-place
        for i in range(cute.size(tRS_rPostAct)):
            tRS_rPostAct[i] = params.act_fn(tRS_rD[i])
 return tRS_rPostAct # write PostAct tensor via TMA
```SM100+ uses packed f32x2 operations sec="/api/index?aid=55a3fecc-9e80-4788-a Enteriting/ indexing higher throughput:

```python
# SM100+ path
for i in range(cute.size(tRS_rPostAct) // 2):
    tRS_rPostAct[2*i], tRS_rPostAct[2*i+1] = params.act_fn(
        (tRS_rD[2*i], tRS_rD[2*i+1])
    )
```

Each architecture has its corresponding concrete class: `GemmActSm90`, `GemmActSm100`, `GemmActSm120`.

**Gated Activations (SwiGLU, etc.)**

`GemmGatedMixin` handles gated activation (output N dimension is half of input):

```python
class GemmGatedMixin(GemmActMixin):
    _epi_ops = (*GemmDefaultEpiMixin._epi_ops,
                TileStore("mPostAct", epi_tile_fn=_gated_epi_tile_fn))
 # PostAct shape: (M, N//2), epi_tile N dimension
```

Supported activation types:

| Activation | Category | Forward | Backward |
|-----------|------|---------|----------|
| `gelu_tanh_approx` | non-gated | `act_fn_map` | `dact_fn_map` |
| `relu` | non-gated | `act_fn_map` | `dact_fn_map` |
| `relu_sq` | non-gated | `act_fn_map` | `dact_fn_map` |
| `swiglu` | gated | `gate_fn_map` | `dgate_fn_map` |
| `swiglu_oai` | gated | `gate_fn_map` | `dgate_fn_map` |
| `reglu` | gated | `gate_fn_map` | `dgate_fn_map` |
| `geglu` | gated | `gate_fn_map` | `dgate_fn_map` |
| `glu` | gated | `gate_fn_map` | `dgate_fn_map` |

**Backward: GemmDActMixin / GemmDGatedMixin**

The backward pass fuses GEMM + activation derivative via `gemm_dact`:

```python
class GemmDActMixin(GemmActMixin):
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC):
        # D = dout @ W2 (from mainloop)
        # C = preact (loaded via TMA)
        # act_fn(preact, dout) -> (dpreact, postact)
        tRS_rD[i], tRS_rPostAct[i] = params.act_fn(tRS_rC_acc[i], tRS_rD[i])
```

`GemmDGatedMixin` is more complex Kauz additional supports `ColVecReduce` for column-wise reduction:

```python
class GemmDGatedMixin(GemmActMixin):
    _epi_ops = (*GemmActMixin._epi_ops, ColVecReduce("mColVecReduce"))
```

### 8. GEMM + Norm + Activation (`gemm_norm_act.py`)

Triple fusion: `PostAct = act((A @ B + C) * colvec * rowvec)`

```python
class GemmNormActMixin(GemmActMixin):
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC):
        # 1. D = alpha * (A @ B) + beta * C
        # 2. D *= colvec (rstd) * rowvec (norm_weight)
        vec_multiply(self, tRS_rD, tDrColVec, tDrRowVec)
        # 3. PostAct = act(D)
        for i in range(size):
            tRS_rPostAct[i] = params.act_fn(tRS_rD[i])
```

- `colvec` is typically inverse standard deviation (rstd)
- `rowvec` is typically normalization weight
- The D tensor stores the normalized pre-activation values (for backward)

### 9. Linear (`linear.py`)

QuACK provides a high-performance alternative to `nn.Linear`:

```python
class Linear(nn.Linear):
    def forward(self, input):
        if input.is_cuda and self.in_features % 8 == 0 and self.out_features % 8 == 0:
            return linear_func(input, self.weight, self.bias,
                             fuse_grad_accum=self.fuse_grad_accum)
        else:
            return F.linear(input, self.weight, self.bias)
```

**Autograd Integration**

| Function | Forward | Backward dx | Backward dW |
|----------|---------|-------------|-------------|
| `LinearFunc` | `gemm(x, W.T, bias)` | `gemm(dout, W)` | `gemm(dout.T, x)` |
| `LinearActFunc` | `gemm_act(x, W.T, bias, act)` | `gemm(dout, W)` | `gemm(dout.T, x)` |
| `DActLinearFunc` | `gemm(x, W.T)` | `gemm_dact(dout, W, preact)` | `gemm(dout.T, postact)` |**Fuse Grad Accumulation**

When `fuse_grad_accum=True`, dW in backward is directly atomic-added to `weight.grad`, avoiding extra allocation and copying:

```python
if not ctx.fuse_grad_accum or weight_og.grad is None:
    dweight = matmul_fn(dout.T, x, out_dtype=ctx.weight_dtype)
else:
    gemm_add_inplace(dout.T, x, weight_og.grad)
    dweight = weight_og.grad
 weight_og.grad = None # PyTorch
```

### 10. Linear Cross Entropy (`linear_cross_entropy.py`)

Fused linear projection + cross entropy loss, avoiding materializing the full logits matrix through chunking.

**Chunked Forward**

```python
def chunked_linear_cross_entropy_fwd(x, weight, target, chunk_size=4096):
 # chunk (chunk_size, d):
    for chunk in chunks:
 logits = x_chunk @ weight.T # (chunk_size, V) -
        cross_entropy_fwd_out(logits, target)  # in-place gradient
        dx_chunk = dlogits @ weight            # (chunk_size, d)
 dw += dlogits.T @ x_chunk # dW
```

- Memory savings: only requires `O(chunk_size * V)` of logits storage instead of `O(B*L * V)`
- The dlogits and x_chunk of the last chunk are deferred to backward processing (avoiding unnecessary GEMM)
- dW of all intermediate chunks uses `gemm_add_inplace` in-place accumulation

### 11. MLP (`mlp.py`)

Complete MLP block implementation: fc1 + activation + fc2, with intelligent activation recomputation.

**Two Execution Modes**

```python
def mlp_func(x, weight1, weight2, activation, recompute=False):
    if recompute:
        return MLPRecomputeFunc.apply(x, weight1, weight2, activation, ...)
    else:
        # Normal mode: 2 separate autograd functions
        preact, postact = fc1_fn(x, weight1, activation)  # gemm_act / gemm_gated
        out = fc2_fn(preact, weight2, postact, activation) # act_linear / gated_linear
```

**Activation Recomputation (`recompute=True`)**

| Mode | Forward GEMMs | Backward GEMMs | Total | Saved Activation |
|------|:---:|:---:|:---:|------|
| normal | 2 | 4 | **6** | x + preact |
| `torch.utils.checkpoint` | 2 | 2 (replay) + 4 | **8** | x only |
| `recompute=True` | 2 | 1 (replay fc1) + 4 | **7** | x only |

`recompute=True` saves **1 GEMM** compared to `torch.utils.checkpoint`:

```python
class MLPRecomputeFunc(torch.autograd.Function):
    @staticmethod
    def backward(ctx, dout):
 # fc1(1 GEMM), gemm_dact preact postact
        preact = ops.matmul_fwd(x_flat, weight1.T)        # replay fc1 only
        dpreact, postact = ops.matmul_bwd_dact(dout, weight2, preact, activation)
 # dW2 = dout.T @ postact (postact gemm_dact)
        # dx = dpreact @ W1
        # dW1 = dpreact.T @ x
```

Principle: while computing dpreact, `gemm_dact` simultaneously recomputes postact (from preact), so there is no need to replay fc2 like checkpoint does.

**MLP Module**

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, activation="gelu_tanh_approx",
                 fuse_grad_accum=False, recompute=False):
        fc1_out = 2 * hidden_features if gated else hidden_features
        self.fc1 = nn.Linear(in_features, fc1_out)
        self.fc2 = nn.Linear(hidden_features, out_features)
```

For gated activation, the fc1 output is `2 * hidden_features` (interleaved), with automatic Muon reshape handling.

## Part 4: Supporting Infrastructure

### 12. Fast Math

`FastDivmod` wraps CUTLASS's `FastDivmodDivisor` to convert integer division into multiplication + shift, avoiding expensive integer division on the GPU:

```python
class FastDivmod(cute.FastDivmodDivisor):
    def __init__(self, divisor, is_power_of_2=None):
        super().__init__(divisor, is_power_of_2=is_power_of_2)
 self.divisor = divisor # keep divisor compute
```

Widely used in tile schedulers (`group_size_fdd`, `num_clusters_per_problem_fdd`, etc.).

### 13. Stochastic Rounding

SM100+ supports stochastic rounding for FP32->BF16 down-conversion in quantization-aware training:

```python
class RoundingMode(IntEnum):
 RN = 0 # Round to nearest even(default)
    RS = 1  # Stochastic rounding（SM100+ only）
```

The implementation uses a Philox 4x32b PRNG to generate pseudo-random numbers, then calls Blackwell hardware instructions:

```python
# PTX: cvt.rs.satfinite.bf16x2.f32 dst, src_hi, src_lo, rand
def cvt_f32x2_bf16x2_rs(a, b, rand_bits):
    return llvm.inline_asm("cvt.rs.satfinite.bf16x2.f32 $0, $2, $1, $3;", ...)
```

- Each pair of FP32 values consumes one 32-bit random number
- Each Philox call generates four 32-bit random numbers (4 pairs / call)
- The seed is salted with tile coordinates and thread index to ensure independent randomness for each element

### 14. Tile Scheduler

QuACK implements multiple tile scheduling strategies:

**TileScheduler (Standard)**

- **NONE**: Non-persistent, each CTA processes one tile
- **STATIC**: Persistent, CTAs iterate over all tiles with a fixed stride (`num_persistent_clusters`)
- **DYNAMIC**: Persistent, acquires the next work index via `atomicInc`
- **CLC**: SM100+ hardware-assisted scheduling, acquires the next tile via `clc_query` + mbarrier

**CTA Swizzle (L2 Cache Friendly)**

Tile iteration uses a swizzle pattern to improve L2 data reuse:

```python
# : group
if group_id % 2 == 1:  # serpentine order
    cid_slow = ncluster_slow - 1 - cid_slow
```

- `group_size`: Controls the swizzle granularity (default `max_swizzle_size=8`)
- `raster_order`: AlongM / AlongN / Heuristic (auto-selected)

**Special Schedulers**

- `VarlenMTileScheduler`: Supports variable-length M dimensions (e.g., batched attention), using `cu_seqlens_m` + warp-level prefix sum Tracy for efficient tile allocation
- `TriangularTileScheduler`: Used for triangular matrix operations (e.g., dKdV of causal attention), processing only lower-triangular tiles

See [SM100 CuTeDSL](../../../blackwell/ref-docs/cutedsl/blackwell-cutedsl-sm100.md) for Blackwell-specific CuTeDSL programming details.
