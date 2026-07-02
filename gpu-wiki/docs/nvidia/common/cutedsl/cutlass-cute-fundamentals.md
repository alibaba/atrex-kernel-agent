# CUTLASS/CuTe Core Concepts and Layout Algebra


**Last updated**: 2026-06-30

## Relationship Between CUTLASS, CuTe, and CuTeDSL

| Component | Role | Language |
|------|------|------|
| **CUTLASS** | High-performance linear algebra template library providing GEMM/Conv implementations | C++ |
| **CuTe** | Core abstraction layer of CUTLASS 3.x, defining Layout/Tensor/Atom concepts | C++ |
| **CuTeDSL** | Python frontend for CuTe, generating GPU kernels equivalent to C++ through JIT compilation | Python |

CuTeDSL shares all core concepts with CuTe C++ (Layout, Tensor, Atom, TiledMMA/TiledCopy), with consistent API naming. CuTeDSL is not a replacement for CUTLASS C++, but rather a high-productivity kernel authoring framework.

---

## Layout Basics

### What Is a Layout

A Layout is a **function mapping from coordinate space to index space**, consisting of a pair of IntTuples: `(Shape, Stride)`.

```
Layout = (Shape, Stride)
index = inner_product(coordinate, stride)
```

Layout abstracts how array elements are organized in memory, allowing MxN layouts in row-major and column-major to be handled with the same code.

### IntTuple Hierarchy

IntTuple is defined recursively: an IntTuple is an **integer** or a **tuple of IntTuples**.

```
6 - scalar
(2) - 1 tuple
(4, 3) - 2 tuple
(3, (6, 2), 8) - tuple
```

**IntTuple Operations:**

| Operation | Meaning |
|------|------|
| `rank(x)` | Number of elements (scalar rank=1) |
| `get<I>(x)` | The I-th element |
| `depth(x)` | Nesting depth (scalar=0, int tuple=1) |
| `size(x)` | Product of all elements |

### Static vs. Dynamic Integers

- **Dynamic integer**: `int`, `size_t` — determined at runtime
- **Static integer**: `cute::C<Value>` / `Int<N>` / `_1, _2, _3` — compile-time constant

Static integers participate in operations producing static results, enabling compile-time optimization.

### Shape and Stride

Shape defines the coordinate space, and Stride defines the mapping to indices. **Shape and Stride must be congruent** (same tuple structure).

```
Layout: (4, 2):(1, 4)    — 4x2 column-major
  0  4
  1  5
  2  6
  3  7

Layout: (4, 2):(2, 1)    — 4x2 row-major
  0  1
  2  3
  4  5
  6  7
```

When stride is omitted:
- **LayoutLeft** (default): left-to-right exclusive prefix product → generalized column-major
- **LayoutRight**: right-to-left exclusive prefix product → generalized row-major

---

## Coordinate System

Each Layout accepts three compatible coordinate types:

| Coordinate Type | Description | Example (Shape=(3,(2,3))) |
|----------|------|----------------------|
| **1-D** | Single integer, colexicographic ordering | `16` |
| **R-D** | Rank-dimensional coordinate | `(1, 5)` |
| **Natural (h-D)** | Hierarchical coordinate congruent with Shape | `(1, (1, 2))` |

The three coordinate types are equivalent: `16 ↔ (1,5) ↔ (1,(1,2))` all map to the same index.

**Core formula:** Index = inner product of natural coordinate and Stride

```
Layout (3,(2,3)):(3,(12,1))
coordinate (i,(j,k)) → index = i*3 + j*12 + k*1
```

**Coordinate conversion functions:**
- `idx2crd(idx, shape)` — index → natural coordinate
- `crd2idx(coord, shape, stride)` — coordinate → index

---

## Five Layout Algebra Operations

### 1. Coalesce (Merge and Simplify)

Merge adjacent modes without changing the function mapping, reducing Layout complexity.

**Four coalescing rules (adjacent modes `s0:d0` and `s1:d1`):**

1. `s0:d0 ++ _1:d1 => s0:d0` — ignore size-1 modes
2. `_1:d0 ++ s1:d1 => s1:d1` — ignore size-1 modes
3. `s0:d0 ++ s1:s0*d0 => s0*s1:d0` — merge when strides are contiguous
4. `s0:d0 ++ s1:d1 => (s0,s1):(d0,d1)` — otherwise keep separated

**Example:** `(2,(1,6)):(1,(6,2))` → coalesce → `12:1`

**Coalesce by mode:** Accepts a profile parameter, preserves rank structure, and independently coalesces each sub-layout.

### 2. Composition (Function Composition)

`R(c) := (A ∘ B)(c) := A(B(c))`, where the result R satisfies `compatible(B, R)`.

**Computation steps (multi-mode A composed with 1-D B=s:d):**

1. **Shape Division (stride calculation):** Divide out d from A's shape mode by mode
   - `(6,2) / 2 => (3,2)`
   - `(3,6,2,8) / 72 => (1,1,1,4)`

2. **Shape Mod (shape restriction):** Retain the first s elements
   - `(6,2) % 3 => (3,1)`
   - `(1,2,2,8) % 16 => (1,2,2,4)`

**Example:**
```
A = (6,2):(8,2), B = (4,3):(3,1)

A ∘ 4:3 = (2,2):(24,2)    — divide by 3, mod by 4
A ∘ 3:1 = (3,1):(8,2)     — divide by 1, mod by 3
result: ((2,2),3):((24,2),8)
```**Tiler**: Layout, Tiler tuple, or Shape (interpreted as stride-1 layout), supports per-mode compositing.

### 3. Complement

Find the Layout of elements **not covered** by Layout A — the "remainder".

**Post-conditions:**
- cosize of `(A, complement)` ≥ `size(cotarget)`
- Result is **ordered** (positive increasing strides) → unique
- The value domains of A and complement are disjoint

**Example:**
```
complement(4:1, 24) = 6:4 - 4 layout 6
complement(6:4, 24) = 4:1 -
complement(4:2, 24) = (2,3):(1,8) -
```

### 4. Division (Tiling)

Divide Layout A by Tiler B into **elements within a tile** and **indices between tiles**:

```
A ⊘ B := A ∘ (B, B*)
where B* = complement(B, size(A))
```

Result mode-0 = elements within a tile (selected by B), mode-1 = layout of tiles (iterating over repetitions).

**Variants:**
```
logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide : ((TileM,TileN), (RestM,RestN,L,...)) ←
tiled_divide   : ((TileM,TileN), RestM, RestN, L, ...)
flat_divide    : (TileM, TileN, RestM, RestN, L, ...)
```

`zipped_divide` is the most useful: mode-0 is the tile layout, mode-1 indexes tiles. `zd(0,k)` accesses the k-th tile.

### 5. Product (Replication)

Replicate Layout A according to Layout B:

```
A ⊗ B := (A, A* ∘ B)
where A* = complement(A, size(A)*cosize(B))
```

Mode-0 = a copy of A, mode-1 = unique replication arranged by B.

**blocked_product vs raked_product:**
- **blocked_product**: tiles arranged contiguously (block-cyclic distribution)
- **raked_product**: tiles interleaved (cyclic distribution)

---

## Tensor

### Tensor = Engine + Layout

```
Tensor T = E ∘ L
```

Engine (iterator) provides data access, Layout provides coordinate → index mapping.

**Iterator types:**

| Type | Use Case |
|------|----------|
| Pointer (`gmem_ptr`, `smem_ptr`) | Ordinary memory tensor |
| Counting iterator (`counting_iterator`) | Implicit tensor (value = index) |
| ArithTupleIterator | TMA coordinate tensor |

**Construction:**
```cpp
// C++
Tensor mA = make_tensor(make_gmem_ptr(A), make_layout(shape, stride));

# Python (CuTeDSL)
mA = cute.make_tensor(cute.make_ptr(Float16, ptr, AddressSpace.gmem), layout)
```

### Register Tensor

```cpp
// C++
Tensor tCrC = make_tensor_like(tCgC);

# Python
tCrC = cute.make_rmem_tensor_like(src, dtype)
```

---

## MMA Atom Abstraction

CuTe encapsulates GPU matrix multiply-accumulate hardware instructions into four layers of abstraction:

```
Operation → Traits → Atom → TiledMMA
```

### Operation

Directly wraps PTX instructions. Naming convention encodes: architecture, dimensions, types, input arrangement.

```
SM70_8x8x4_F32F16F16F32_NT
│    │       │            └── A=col-major, B=row-major
│    │       └── D=F32, A=F16, B=F16, C=F32
│    └── MxNxK = 8x8x4
└── Volta
```

### Traits

Defines metadata for Operation:
- `ValTypeD/A/B/C` — logical computation type
- `Shape_MNK` — logical MxNxK shape
- `ThrID` — thread mapping layout
- `ALayout/BLayout/CLayout` — `(thread, value) → (m, n)` mapping

### TiledMMA

Combines multiple Atoms via `make_tiled_mma(operation, atom_layout, tiler)`:

```cpp
// 4 quadpair atom 2x2 -> 16x16x4 Volta HMMA
TiledMMA mma = make_tiled_mma(SM70_8x8x4_F32F16F16F32_NT{},
                              Layout<Shape<_2,_2>, Stride<_2,_1>>{});
```

**Hardware granularity:**

| Granularity | Thread Count | Architecture |
|-------------|--------------|--------------|
| Single thread (FMA) | 1 | General |
| Quadpair | 8 | Volta |
| Warp | 32 | Ampere |
| Warpgroup | 128 | Hopper |

### Partitioning with TiledMMA

```cpp
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);       // (MMA, MMA_M, MMA_K)
Tensor tCsB = thr_mma.partition_B(sB);       // (MMA, MMA_N, MMA_K)
Tensor tCgC = thr_mma.partition_C(gC);       // (MMA, MMA_M, MMA_N)
Tensor tCrC = thr_mma.make_fragment_C(tCgC); // register

cute::gemm(mma, tCsA, tCsB, tCrC);
```## TiledCopy Abstraction

### Copy_Atom → TiledCopy

```cpp
TiledCopy copyA = make_tiled_copy(
 Copy_Atom<UniversalCopy<uint128_t>, TA>{}, // 128-bit copy
 Layout<Shape<_32,_8>>{}, // 32x8 layout
 Layout<Shape<_4,_1>>{} // 4x1
);
```

Each thread reads 4 elements using 128-bit instructions. If vectorization to 128-bit is not possible, CuTe raises a static error.

### Partitioning with TiledCopy

```cpp
ThrCopy thr_copy = copy_a.get_slice(threadIdx.x);
Tensor tAgA = thr_copy.partition_S(gA);  // (CPY, CPY_M, CPY_K, k) — source
Tensor tAsA = thr_copy.partition_D(sA);  // (CPY, CPY_M, CPY_K)    — destination

cute::copy(copy_a, tAgA, tAsA);
```

`CPY` The first mode contains all elements consumed by a single instruction.

---

## CuTe Algorithms

### copy

```cpp
// default dispatch( tensor typeautomatic)
copy(src, dst);

// Copy_Atom
copy(copy_atom, src, dst);
```

**Three levels of optimization:**
1. **Instruction dispatch**: Use optimized instructions when the memory space is known (e.g., `cp.async`)
2. **Vectorization**: Coalesce accesses when the static layout proves vectorization is possible (4 × `ld.b32` → 1 × `ld.b128`)
3. **Validation**: Confirm that the instruction applies to the src/dst tensor

### gemm

Automatically dispatches based on the number of tensor modes:

| Modes | Operation |
|-------|-----------|
| `(V) x (V) => (V)` | Element-wise FMA/MMA |
| `(M) x (N) => (M,N)` | Outer product |
| `(M,K) x (N,K) => (M,N)` | Matrix multiply |
| `(V,M,K) x (V,N,K) => (V,M,N)` | Batched matrix multiply |

The V mode (vector) is at the innermost level, and the K mode (reduction) is at the outermost level.

### copy_if

Conditional copy with a predicate. Only copies elements where the predicate tensor is non-zero. Used for boundary handling.

### Others

- `axpby(alpha, x, beta, y)` — y = αx + βy
- `fill(tensor, value)` — Fill with a scalar value
- `clear(tensor)` — Zero fill


## Related

- [CuTeDSL API Reference Guide](cutedsl-api-reference-guide.md)
- [CuTeDSL Inline PTX Writing Overview](cutedsl-inline-ptx-patterns.md)
- [CuTeDSL Software Pipeline and Synchronization Patterns](cutedsl-pipeline-patterns.md)
- [CuTeDSL Programming Model](cutedsl-programming-model.md)
- [CUTLASS 3.x Architecture](cutlass-3x-architecture.md)
- [CUTLASS GEMM Optimization Strategy](cutlass-gemm-optimization.md)
- [Community GEMM Optimization Practical Summary](../../../generic/gemm-optimization-guide.md)
