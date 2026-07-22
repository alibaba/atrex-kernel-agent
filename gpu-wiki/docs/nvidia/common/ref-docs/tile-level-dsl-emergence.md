# Why Tile-Level DSLs Emerged: Triton, TileLang, cuTile and Mojo

An analysis of why tile-level DSLs emerged, what they automate, and where their abstractions leak — connecting hardware constraints to software design decisions.

---

## 1. Glossary

| Term | Abbreviation | One-Line Description |
|---|---|---|
| Domain-specific language | DSL | A language designed for a specific domain, at a higher abstraction than general-purpose languages |
| Tile-level programming | — | Programming with data blocks (tiles) rather than individual threads as the unit |
| Program instance | — | In Triton, a logical unit processing one tile, corresponding to one block |
| Block pointer | — | A pointer to an entire data block, automatically handling boundaries and strides |
| Software pipelining | — | Compiler automatically arranges load/compute overlap |
| Autotuning | — | Searching across multiple configurations (tile sizes, etc.) for the fastest one |
| Leaky abstraction | — | When a high-level abstraction fails to fully hide low-level details |
| Intermediate representation | IR | Compiler-internal representation between source code and machine code |
| Tensor descriptor | — | Metadata describing tensor shape/stride for hardware data movement units |
| Occupancy | — | Ratio of active warps to maximum warps on an SM |

---

## 2. Manual Orchestration is Too Expensive, So It Gets Automated

Writing a high-performance kernel requires manually handling a long chain of hardware details: choosing block sizes based on occupancy, tiling data into shared memory, avoiding bank conflicts, using double-buffering for load-compute overlap, and arranging fragment layouts for Tensor Cores. These orchestrations are interconnected — adjusting one affects everything else. **Writing a library-quality GEMM often requires thousands of lines of CUDA.**

Tile-level DSLs like Triton, TileLang, cuTile, and Mojo exist to automate this orchestration — **you only describe "what to compute on one tile"; tiling, pipelining, and bank handling are delegated to the compiler.**

The key insight is not "DSLs let you ignore hardware" — quite the opposite. **Each DSL essentially automates certain hardware constraints, and the boundary of what it can automate — plus what it still requires you to understand — is precisely where abstraction leaks.**

### 2.1 Programming Unit Shifts from Thread to Tile

CUDA's programming unit is the thread: you write "what thread i does." Triton raises this to tiles: you write "what tile N does." **One program instance directly operates on an entire data block; how threads are distributed and how shared memory is used is decided by the compiler.**

First leaky abstraction: you still need to understand tiling and shared memory to choose appropriate tile sizes and diagnose why the compiler didn't meet expectations.

### 2.2 A Triton Kernel: Tiling and Boundary Handling Automated

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(axis=0)              # Which tile (corresponds to one block)
    offs = pid * BLOCK + tl.arange(0, BLOCK)  # Element indices this tile covers
    mask = offs < n                           # Boundary mask, auto-handles tail block
    x = tl.load(x_ptr + offs, mask=mask)      # Block load; coalescing guaranteed by compiler
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)            # num tiles = ceil(n / BLOCK)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out
```

Reading this code:

- **No `threadIdx`** — you don't write per-thread logic; you write what happens on the index range `offs` for this tile.
- **`mask = offs < n` handles tail-block boundary** in a single line (CUDA requires `if (i < n)`; Triton turns this into a vector mask).
- **`tl.load` loads an entire block at once** — coalescing and shared memory usage are compiler-determined based on hardware.
- **`BLOCK` is still your choice** — the right size depends on occupancy and data volume; the DSL doesn't remove this decision.

Real benefits materialize in GEMM, attention, and other kernels requiring complex tiling and pipelining — **a few dozen lines can approach thousands of lines of hand-written CUDA.**

### 2.3 Automation Boundaries

| Layer | Content |
|---|---|
| Hardware constraints | Tiling/SMEM/bank/pipelining/Tensor Core layout require manual orchestration at high cost |
| **DSL automates** | Intra-tile thread allocation, SMEM staging, coalesced access, software pipelining, partial layouts |
| **Still exposed to you** | Tile shape and size, occupancy impact, matrix unit layout constraints, autotune configuration |
| Leaky abstraction consequence | Wrong tile → low occupancy or bandwidth-bound; suboptimal compiler pipelining is hard to intervene on |
| Correct approach | Understand hardware to choose tile/autotune, read IR for diagnosis, fall back to hand-written CUDA for critical kernels when necessary |

> **DSLs reduce coding effort, not the requirement to understand hardware.**

### 2.4 Orientation Differences Among Four DSLs

| DSL | Orientation |
|---|---|
| **Triton** | Tile-unit, Python syntax, emphasizes autotune; most widely used for DL operators. Hides warp/SMEM/pipeline; tile shape and autotune config remain exposed |
| **TileLang** | More explicit description of tile memory hierarchies and pipeline stages; control lies between Triton and CUDA — leaks closer to hardware but **offers more tuning knobs** |
| **cuTile** | NVIDIA-direction tile programming model aligned with tensor descriptors and TMA (hardware data movement unit's software entry); **encapsulates Hopper/Blackwell transfer capabilities into tile abstractions** |
| **Mojo** | Python superset with systems-level control; not GPU-kernel-only; emphasizes **zero-cost abstraction** with gradual descent from high-level syntax to low-level control |

> Which to choose depends on how much control you need: for convenience use Triton; for pushing limits or handling unusual access patterns, use higher-control options or fall back to CUDA.

---

## 3. Cross-Vendor Perspective: DSLs Want to Flatten Hardware, But Leak Points Vary

| Dimension | NVIDIA | AMD | Does Abstraction Leak? |
|---|---|---|---|
| Backend | Via IR lowering to PTX/SASS | Via IR lowering to ROCm/GCN ISA | No (compiler handles) |
| Warp/wavefront width | 32 | 64 (CDNA) / 32 (RDNA) | **Yes**: affects optimal tile size |
| Shared memory/LDS capacity | Architecture-dependent | Architecture-dependent | **Yes**: affects maximum tile size |
| Matrix unit layout | Tensor Core tiles | Matrix Core (MFMA) tiles | **Yes**: autotune space differs |

Triton and others support multiple backends — **functionally, the same kernel runs on both vendors, but performance is not portable**: optimal tile size varies with warp width (32 vs 64) and shared memory capacity; autotuning must be re-run on target hardware.

> **DSLs downgrade differences from "rewrite code" to "re-run autotune," but don't eliminate them.**

---

## 4. Hardware Constraint to Software Design Rule Mapping

| Hardware Fact | Cost/Limitation | DSL Automation | Still Leaks to You |
|---|---|---|---|
| Tiling/SMEM requires manual orchestration | Thousands of lines for hand-written kernels | Auto tiling + SMEM staging | Tile shape/size must be chosen based on occupancy and reuse |
| Load/compute must overlap | Manual double-buffering is error-prone | Auto software pipelining | Hard to intervene when pipeline is suboptimal |
| Bank conflict / coalesced access | Layout tuning is tedious | Auto coalescing + partial layout | Unusual access patterns still require bank understanding |
| Matrix units require fixed tile layout | Fragment layout is difficult to write | Partial matrix unit call encapsulation | Tile shape and precision still require understanding |
| Warp width/capacity varies by vendor | Optimal config changes with hardware | Multi-backend IR lowering | Autotune must re-run on target hardware |

---

## 5. Summary

The extreme cost of manually orchestrating tiling, SMEM, bank conflicts, pipelining, and Tensor Core layouts drove tile-level DSLs into existence: **you write only what a tile computes; the compiler takes over thread allocation, SMEM, pipelining, and coalesced access.**

Triton, TileLang, cuTile, and Mojo each have different orientations but follow the same logic: **automate mechanizable hardware orchestration, leaving tile shape, occupancy tradeoffs, matrix unit layout, and autotune — the judgment-based decisions — to you.** These remaining decisions are precisely the abstraction leak points.

The natural follow-up: **HIP / SYCL-oneAPI / OpenCL / Vulkan-Metal compute** — why switching vendors still requires re-tuning, and what separates functional portability from performance portability.

---

## References

- Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*
- OpenAI Triton official documentation and tutorials
- TileLang project documentation; NVIDIA cuTile / CUDA Tile programming materials
- Modular, Mojo language documentation
