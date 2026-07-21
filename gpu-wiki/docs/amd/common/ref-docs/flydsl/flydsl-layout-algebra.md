# FlyDSL Layout Algebra Reference

Applicability: backend: flydsl; hardware: amd; topic: reference

FlyDSL implements CuTe layout algebra (derived from NVIDIA CUTLASS, BSD-3-Clause), providing complete layout algebra operations on AMD GPUs via the Fly MLIR dialect.

> **All `fx.*` layout operations generate MLIR IR and must be called within a `@flyc.kernel` or `@flyc.jit` function body.**

---

## 1. Core Types

### 1.1 MLIR Types

| Type | MLIR Syntax | Description |
|------|-----------|------|
| `!fly.int_tuple` | `!fly.int_tuple<(8, 16)>` | Integer tuple, nestable |
| `!fly.layout` | `!fly.layout<(8, 16):(1, 8)>` | Layout = (Shape, Stride) pair |
| `!fly.pointer` | `!fly.pointer<f16>` | Typed pointer |
| `!fly.memref` | `!fly.memref<...>` | Memory reference with layout |
| `!fly.swizzle` | `!fly.swizzle<...>` | Swizzle descriptor |
| `!fly.copy_atom` | `!fly.copy_atom_universal_copy<...>` | Copy atom type |
| `!fly.mma_atom` | `!fly.mma_atom_universal_fma<...>` | MMA atom type |

### 1.2 Layout Basics

A **Layout** is defined by a pair `(Shape, Stride)`, describing the mapping from logical coordinates to physical memory indices:

```
Index = dot(Coord, Stride) = sum(c_i * s_i)
```

| Concept | Mathematical Definition | FlyDSL API |
|------|----------|------------|
| **Shape** | Positive integer tuple describing the size of each dimension | `fx.make_shape(M, N)` |
| **Stride** | Integer tuple describing the stride of each dimension | `fx.make_stride(s0, s1)` |
| **Layout** | (Shape, Stride) pair | `fx.make_layout(shape, stride)` |
| **Coord** | Integer tuple, a position in logical space | `fx.make_coord(i, j)` |

---

## 2. Construction Operations

```python
import flydsl.expr as fx

# Shape Stride
shape = fx.make_shape(8, 16)              # !fly.int_tuple<(8, 16)>
stride = fx.make_stride(1, 8)             # !fly.int_tuple<(1, 8)>
layout = fx.make_layout(shape, stride)    # !fly.layout<(8, 16):(1, 8)>

# : direct Python
layout = fx.make_layout((8, 16), (1, 8))

# English note
coord = fx.make_coord(i, j)

# English note
it = fx.make_int_tuple((4, 8, 2))

# Shape
shape_nested = fx.make_shape(9, (4, 8))   # (9, (4, 8))

# Layout - stride sort
col_major = fx.make_ordered_layout((M, N), order=(0, 1)) # M-first (columnmain)
row_major = fx.make_ordered_layout((M, N), order=(1, 0)) # N-first (rowmain)

# Identity layout / tensor
identity = fx.make_identity_layout((M, N))
id_tensor = fx.make_identity_tensor((M, N))
```

---

## 3. Coordinate Mapping

Fundamental layout operations: logical coordinates ↔ physical memory indices.

### `crd2idx` — Coordinate → Index

```python
idx = fx.crd2idx(coord, layout)
```

### `idx2crd` — Index → Coordinate

```python
coord = fx.idx2crd(idx, layout)
```

### Example

For layout `((8, 16), (1, 8))` (8×16 column-major):
- `crd2idx((3, 5), layout)` = `3*1 + 5*8` = `43`
- `idx2crd(43, layout)` = `(43 % 8, 43 / 8)` = `(3, 5)`

---

## 4. Query Operations

| Operation | Description | Example |
|------|------|------|
| `fx.size(layout)` | Total number of elements = product(shape) | `size((8, 16)) = 128` |
| `fx.cosize(layout)` | Codomain size = max(index) + 1 | `cosize(((8,16),(1,8))) = 128` |
| `fx.rank(layout)` | Number of top-level modes | `rank((8, 16)) = 2` |
| `fx.get_shape(layout)` | Extract Shape | Returns `!fly.int_tuple` |
| `fx.get_stride(layout)` | Extract Stride | Returns `!fly.int_tuple` |
| `fx.get(int_tuple, i)` | Extract the i-th element | `get((8, 16), 0) = 8` |

---

## 5. Algebraic Operations

### 5.1 Composition — `fx.composition(A, B)`

Compose two layouts: `result(x) = A(B(x))`

**Purpose**: Apply a permutation or tile coordinate mapping to a memory layout.

### 5.2 Complement — `fx.complement(tiler, target_size)`

Computes the "remainder" modes not covered by the tiler.

**Purpose**: An internal building block of `logical_divide`; computes the complementary iteration space during tiling.

### 5.3 Coalesce — `fx.coalesce(layout)`

Simplify layout: flatten nested patterns and merge adjacent compatible patterns.

**Invariant**: `size(result) == size(layout)` with equivalent mapping.

### 5.4 Right Inverse — `fx.right_inverse(layout)`

Compute the right inverse of a layout mapping.

### 5.5 Recast Layout — `fx.recast_layout(layout, old_bits, new_bits)`

Adjust the layout when the type width changes:

```python
# 16-bit -> 8-bit
recasted = fx.recast_layout(layout, old_type_bits=16, new_type_bits=8)
```

---

## 6. Product Operations

Product combines two layouts to create a larger layout Helm. Different variants organize patterns in different ways:

| Variant | Description | API |
|------|------|-----|
| **Logical Product** | Concatenate patterns element by element, the most basic | `fx.logical_product(A, B)` |
| **Zipped Product** | Interleave inner patterns | `fx.zipped_product(A, B)` |
| **Tiled Product** | Group by tile | `fx.tiled_product(A, B)` |
| **Flat Product** | Flatten all patterns | `fx.flat_product(A, B)` |
| **Raked Product** | Interleaved access patterns | `fx.raked_product(A, B)` |
| **Blocked Product** | Block access patterns | `fx.block_product(A, B)` |

```python
result = fx.logical_product(layout, tiler)
result = fx.raked_product(thr_layout, val_layout) # tiled copy
```

---

## 7. Divide Operations

Divide partitions a layout by a tiler, creating a hierarchical layout with "tile" and "remainder" dimensions:

| Variant | Description | API |
|------|------|-----|
| **Logical Divide** | Basic partitioning, internally uses complement | `fx.logical_divide(A, tiler)` |
| **Zipped Divide** | Partition with Zip semantics | `fx.zipped_divide(A, tiler)` |
| **Tiled Divide** | Hierarchical tile partitioning | `fx.tiled_divide(A, tiler)` |
| **Flat Divide** | Flattened partitioning | `fx.flat_divide(A, tiler)` |

```python
# : by block tensor
tileA = fx.make_tile(block_m, block_k)
bA = fx.zipped_divide(A, tileA)
bA = fx.slice(bA, (None, bid)) # current block tile
```

---

## 8. Structural Operations

```python
# mode
selected = fx.select(int_tuple, indices=[0, 2])

# mode
grouped = fx.group(int_tuple, begin=1, end=3)

# English note
extended = fx.append(base_tuple, new_elem)
extended = fx.prepend(base_tuple, new_elem)

# Zip IntTuple
zipped = fx.zip(shapes_a, shapes_b)

# Slice
sliced = fx.slice(layout, coord)
```

---

## 9. MemRef / View Operations

```python
# memory
alloca = fx.memref_alloca(memref_type, layout)

# Load / Store
val = fx.memref_load(memref, indices)
fx.memref_store(value, memref, indices)

# vector Load / Store
vec = fx.memref_load_vec(memref)
fx.memref_store_vec(vector, memref)

# layout / iterator
ly = fx.get_layout(memref)
it = fx.get_iter(memref)

# create view / offset
view = fx.make_view(iterator, layout)
ptr = fx.add_offset(ptr, offset)
```

---

## 10. Copy Atom and Tiled Copy

### 10.1 Construction

```python
# Copy atom
copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

# MMA atom
mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))

# Tiled copy(passed raked product layout)
thr_layout = fx.make_layout((4, 1), (1, 1))
val_layout = fx.make_layout((1, 8), (1, 1))
layout_thr_val = fx.raked_product(thr_layout, val_layout)
tile_mn = fx.make_tile(4, 8)
tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)

# TiledMma tiled copy
tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)

# Tiled MMA
tiled_mma = fx.make_tiled_mma(mma_atom, atom_layout)
tiled_mma = fx.make_tiled_mma(mma_atom, atom_layout, permutation)
```### 10.2 Thread Slicing and Partitioning

```python
# Tiled Copy
thr_copy = tiled_copy.get_slice(tid)    # → ThrCopy
src_part = thr_copy.partition_S(src) # tensor
dst_part = thr_copy.partition_D(dst) # tensor
retiled = thr_copy.retile(tensor) # tile copy atom

# Tiled MMA
thr_mma = tiled_mma.thr_slice(tid)      # → ThrMma
frag_a = thr_mma.make_fragment_A(bA) # register fragment
frag_b = thr_mma.make_fragment_B(bB)
frag_c = thr_mma.make_fragment_C(bC)
part_a = thr_mma.partition_A(tensor_a) #
```

### 10.3 Execution

```python
# execute copy
fx.copy(copy_atom, src_part, dst_part)
fx.copy(copy_atom, src_part, dst_part, pred=pred_tensor) # predicate

# execute GEMM: D = A * B + C
fx.gemm(mma_atom, d, a, b, c)
```

### 10.4 Introspection Properties

| Property | Class | Description |
|------|-----|------|
| `.thr_layout` | CopyAtom/MmaAtom | Thread layout |
| `.tv_layout_src/dst` | CopyAtom | Source/destination thread-value layout |
| `.shape_mnk` | MmaAtom | M×N×K tile dimensions |
| `.tv_layout_A/B/C` | MmaAtom | Thread-value layout for each operand |
| `.tile_size_mnk` | TiledMma | Tiled MMA dimensions |
| `.thr_layout_vmnk` | TiledMma | Thread layout across V, M, N, K |

---

## 11. IntTuple Arithmetic

```python
sum_it = fx.int_tuple_add(a, b)
diff_it = fx.int_tuple_sub(a, b)
prod_it = fx.int_tuple_mul(a, b)
quot_it = fx.int_tuple_div(a, b)

total = fx.int_tuple_product(int_tuple) # reduction
products = fx.int_tuple_product_each(int_tuple) # mode
```

---

## 12. Complete Example: Tiled MFMA GEMM

```python
import flydsl.compiler as flyc
import flydsl.expr as fx

block_m, block_n, block_k = 64, 64, 8

@flyc.kernel
def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

 # definition tile
    tileA = fx.make_tile(block_m, block_k)
    tileB = fx.make_tile(block_n, block_k)
    tileC = fx.make_tile(block_m, block_n)

    # Buffer tensor
    A = fx.rocdl.make_buffer_tensor(A)
    B = fx.rocdl.make_buffer_tensor(B)
    C = fx.rocdl.make_buffer_tensor(C)

 # by tile
    bA = fx.zipped_divide(A, tileA)
    bB = fx.zipped_divide(B, tileB)
    bC = fx.zipped_divide(C, tileC)
    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))
    bC = fx.slice(bC, (None, bid))

 # MMA
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

 # Copy atom
    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

 # fragment
    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

 # execute copy + GEMM
    fx.copy(copy_atom, copy_src_A, copy_frag_A)
    fx.copy(copy_atom, copy_src_B, copy_frag_B)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)
    fx.copy(copy_atom, copy_frag_C, copy_dst_C)

@flyc.jit
def tiledMma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
             stream: fx.Stream = fx.Stream(None)):
    gemm_kernel(A, B, C).launch(grid=(1,), block=(256,), stream=stream)
```## 13. Decision Tree

```
requires layout ？
│
├── create layout？
│ ├── shape + stride -> make_layout(shape, stride)
│   ├── Identity layout → make_identity_layout(shape)
│ └── layout -> make_ordered_layout(shape, order)
│
├── layout？
│ ├── -> size(layout)
│ ├── -> get_shape/get_stride(layout)
│ └── singlemode -> get(shape, i)
│
├── mapping？
│ ├── -> -> crd2idx(coord, layout)
│ └── -> -> idx2crd(idx, layout)
│
├── layout？
│ ├── mapping -> composition(A, B)
│ ├── -> logical_product / raked_product
│ └── -> coalesce(layout)
│
├── / tiling？
│ ├── layout -> logical_divide / zipped_divide
│ └── tile -> tiled_divide
│
├── type？
│   └── recast_layout(layout, old_bits, new_bits)
│
English description
 ├── mode -> select(it, indices)
 ├── mode -> group(it, begin, end)
 └── -> append/prepend(it, elem)
```

---

## Related Documents

- [FlyDSL Programming Guide](flydsl-programming-guide.md) — Compilation pipeline, APIs, debugging
- [FlyDSL Pre-built Kernel Library](flydsl-prebuilt-kernels.md) — Production-grade kernel references
- [CuTeDSL Programming Model](../../../../nvidia/common/ref-docs/cutedsl/cutedsl-programming-model.md) — Corresponding CuTe DSL for NVIDIA
- [CUTLASS/CuTe Core Concepts and Layout Algebra](../../../../nvidia/common/ref-docs/cutedsl/cutlass-cute-fundamentals.md) — Theoretical foundations of CuTe algebra
