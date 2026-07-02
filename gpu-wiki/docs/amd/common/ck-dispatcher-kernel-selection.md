# CK Tile Dispatcher and ML-Driven Kernel Selection

The Composable Kernel (CK) Tile Dispatcher subsystem provides a unified kernel selection and execution engine, supporting both C++ and Python frontends. Its core highlight is the LightGBM-based ML heuristic: through 72-dimensional feature engineering, it predicts the TFLOPS of each kernel configuration, completing the selection of the optimal kernel from over 4600 candidate kernels in microseconds, achieving 98.28% of oracle-best efficiency in benchmarks.


**Last updated**: 2026-07-01

---

## Architecture Overview

```
KernelConfig → Registry → Dispatcher → GPU Execution
                              |
                         SelectionStrategy
                        /                \
                  FirstFit            Heuristic
                                    /         \
                              Rule-based    ML (LightGBM)
```

### Core Components

| Component | File | Description |
|------|------|------|
| Registry | `registry.hpp` | Thread-safe kernel storage, supporting priority and lookup by KernelKey |
| Dispatcher | `dispatcher.hpp` | Top-level scheduling engine, supporting both FirstFit and Heuristic strategies |
| KernelKey | `kernel_key.hpp` | Unique identifier for kernel configuration (dtype, layout, tile size, pipeline, etc.) |
| KernelInstance | `kernel_instance.hpp` | Executable kernel instance, encapsulating launch parameters |
| ML Heuristic | `ml_heuristic.hpp` | C++ side LightGBM inference, 72-dimensional feature extraction |

### Selection Strategies

```cpp
enum class SelectionStrategy {
 FirstFit, // current problem kernel
 Heuristic // use heuristic function sort
};

// definition heuristic
dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
 // returnsbyperformancesort kernel identifier column
    return ranked_kernel_ids;
});
```

---

## ML-Driven Kernel Selection

### Pipeline Overview

```
1. Profiling: GPU benchmark( shape x kernel)
2. Feature Extraction: 72
3. Model Training: LightGBM (log-space)
4. Prediction: problem kernel TFLOPS
5. Selection: TFLOPS high kernel
```

### 72-Dimensional Feature Engineering

The `GemmUniversalFeatureEngine` in `feature_engine.py` extracts 72 features (the code actually defines 72, while the documentation marking 55 is from an earlier version), divided into five categories:

#### Problem Features (13)

| Feature | Description |
|------|------|
| `M, N, K` | GEMM dimensions |
| `split_k` | Split-K factor |
| `log2_M, log2_N, log2_K` | Log-scale dimensions |
| `log2_MNK` | Log of total computational volume |
| `arithmetic_intensity` | Compute density = 2MNK / (M*K + K*N + M*N) / bytes_per_element |
| `aspect_ratio_mn, _mk, _nk` | Dimension ratios |
| `layout` | Encoded as integers: rcr=0, rrr=1, crr=2, ccr=3 |

#### Kernel Features (21)

| Feature | Description |
|------|------|
| `tile_m, tile_n, tile_k` | Block tile sizes |
| `warp_m, warp_n, warp_k` | Warp distribution |
| `warp_tile_m, _n, _k` | Warp tile sizes |
| `pipeline` | compv3=0, compv4=1, compv5=2, mem=3, preshufflev2=4 |
| `scheduler` | intrawave=0, interwave=1 |
| `epilogue` | default=0, cshuffle=1 |
| `pad_m, pad_n, pad_k` | Whether padding is enabled |
| `persistent` | Whether it is a persistent kernel |
| `num_warps` | warp_m * warp_n * warp_k |
| `tile_volume` | tile_m * tile_n * tile_k |
| `tile_mn` | tile_m * tile_n |
| `lds_usage_estimate` | Estimated LDS usage = (tile_m*tile_k + tile_n*tile_k) * bpe |
| `lds_usage_ratio` | LDS utilization (compv4 calculated at 32KB, others at 64KB) |

#### Interaction Features (9)

| Feature | Description |
|------|------|
| `num_tiles_m, _n, _k` | Number of tiles per dimension |
| `total_output_tiles` | Total number of output tiles |
| `tile_eff_m, _n, _k` | Tile efficiency per dimension (effective ratio of trailing tiles) |
| `overall_tile_efficiency` | Product of three-dimensional tile efficiencies |
| `cu_utilization` | CU utilization = total_output_tiles / num_cus |#### Problem-to-Tile Ratio Features (17)

This is the key feature group for performance prediction:

| Feature | Description |
|------|------|
| `ratio_M/N/K_to_tile_m/n/k` | Ratio of problem dimension to tile size |
| `problem_smaller_than_tile_m/n/k` | Whether problem dimension is smaller than tile (binary) |
| `any_dim_too_small` | Whether any dimension is smaller than tile |
| `needs_padding_m/n/k` | Whether padding is needed |
| `has_padding_when_needed_m/n/k` | Kernel has padding and problem requires it |
| `missing_required_padding_m/n/k` | Kernel has no padding but problem requires it (key failure feature) |
| `missing_any_required_padding` | Any dimension missing required padding |

#### Hardware Profile Features (12)

| Feature | Default (MI300) | Description |
|------|---------------|------|
| `hw_num_cus` | 256 | Number of CUs |
| `hw_simds_per_cu` | 4 | SIMDs per CU |
| `hw_total_simds` | 1024 | Total SIMDs |
| `hw_shader_engines` | 32 | Number of Shader Engines |
| `hw_max_clock_mhz` | 2400 | Maximum frequency |
| `hw_max_waves_per_cu` | 32 | Max waves per CU |
| `hw_wavefront_size` | 64 | Wavefront size |
| `hw_lds_capacity` | 65536 | LDS capacity (bytes) |
| `hw_l1_cache_kb` | 32 | L1 cache |
| `hw_l2_cache_kb` | 4096 | L2 cache |
| `hw_l3_cache_kb` | 262144 | L3/Infinity Cache |
| `hw_num_xcd` | 8 | Number of XCDs |

---

## Training Pipeline

### Data Generation

```bash
# 1. benchmark data( shape kernel configuration)
python3 generate_benchmark_data.py \
    --build_dir /path/to/build \
    --output_dir data/fp16_original \
    --dtype fp16 --layout rcr \
    --num_build_jobs 4 --warmup 10 --repeat 50

# 2. conversion parquet
python3 convert_json_to_parquet.py \
    --input data/fp16_original/benchmark_results_fp16_rcr.json \
    --output data/fp16_original/fp16_training_data.parquet \
    --arch gfx950
```

`generate_wide_coverage.py` can generate benchmark data for 706 diverse shapes, and `generate_edge_dims.py` covers edge cases such as N=1 Blocharacter1.

### Model Training

```bash
python3 train.py \
    --data_dir data/ \
    --out_dir models/gemm_universal_fp8_gfx950 \
    --op gemm_universal --dtype fp8 --arch gfx950
```

Key training design decisions:

| Design Decision | Choice | Reason |
|----------|------|------|
| Regression target | `log1p(TFLOPS)` | TFLOPS spans 5 orders of magnitude (0.02~2230); log transform gives equal weight to all shapes |
| Cross-validation | GroupKFold(n_splits=5) | Group key = (M,N,K), ensuring test shapes do not appear in training |
| Early stopping | early_stopping(50) | Prevents overfitting |
| Model | LGBMRegressor | 255 leaves, max_depth=15, 2000 trees |
| Complexity | <= 5000 estimators | Inference latency constraint for C++ deployment |

### Three Target Models

Three models are trained for each (op, dtype, arch) combination:

| Model | Target Column | Purpose |
|------|--------|------|
| `model_tflops.lgbm` | measured_tflops | Main model, used for kernel ranking |
| `model_latency.lgbm` | latency_ms | Latency-sensitive scenarios |
| `model_bandwidth.lgbm` | bandwidth_gb_s | Memory bottleneck analysis |

### Incremental Training (Warm Start)

```bash
python3 train.py \
    --data_dir data/ \
    --out_dir models/v2 \
    --warm_start models/gemm_universal_fp8_gfx950 \
    --warm_start_n_estimators 200
```

Appends 200 new trees on top of an existing model; feature schema must match exactly (auto-validated).

---

## Model Performance

### FP8 RCR, gfx950

| Metric | 108 shapes | 168 shapes (wide coverage) |
|------|-----------|---------------------------|
| Mean TFLOPS Efficiency | 98.28% | 97.51% |
| P10 Efficiency | 94.64% | 93.89% |
| tiny_m (M=1) Efficiency | 95.57% | 96.04% |
| R² (TFLOPS) | 0.997 | 0.993 |

### FP16 RCR, gfx950 (25 shapes, 1024 kernels)

| Metric | Value |
|------|-----|
| Mean Efficiency | 99.36% |
| P10 Efficiency | 98.05% |
| Min Efficiency | 95.45% |### By Pipeline

| Pipeline | Mean Eff | P10 Eff |
|----------|----------|---------|
| compv3 | 99.75% | 99.09% |
| compv4 | 99.40% | 98.54% |
| mem | 99.08% | 96.59% |

---

## C++ Side Integration

### ML Heuristic (C++)

`ml_heuristic.hpp` implements C++ side inference via the LightGBM C API:

```cpp
// 72 ( Python )
static constexpr int NUM_FEATURES = 72;
std::array<double, NUM_FEATURES> features =
    extract_features(problem, kernel_key, hw_profile);

// LightGBM
LGBM_BoosterCreateFromModelfile(model_path, &num_iterations, &handle);
LGBM_BoosterPredictForMat(handle, features.data(), C_API_DTYPE_FLOAT64,
    1, NUM_FEATURES, /*is_row_major=*/1, ...);
```

The HardwareProfile struct encapsulates GPU hardware parameters, corresponding one-to-one with the `GemmUniversalFeatureEngine.__init__` parameters on the Python side.

### Registry + Dispatcher

```cpp
// kernel
Registry registry;
registry.register_kernel(kernel_instance, Registry::Priority::High);

// create dispatcher
Dispatcher dispatcher(&registry);
dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
dispatcher.set_heuristic(my_ml_heuristic);

// execute
Problem problem(M, N, K);
float time_ms = dispatcher.run(a_dev, b_dev, c_dev, problem);
```

---

## Code Generation

`codegen/unified_gemm_codegen.py` generates kernel instance code based on `arch_specs.json`:

### Supported GEMM Variants

| Variant | Pipeline | Scheduler | Epilogue |
|------|----------|-----------|----------|
| gemm_universal | mem, compv3, compv4 | intrawave, interwave | cshuffle, default |
| gemm_preshuffle | preshufflev2 | Auto | cshuffle |
| gemm_multi_d | mem, compv3, compv4 | intrawave, interwave | cshuffle, default |

### Warp Tile Combinations per GPU

| GPU | FP16 | FP8 | FP4 |
|-----|------|-----|-----|
| gfx942 (MI300) | 32x32x8, 16x16x16, 32x32x16, 16x16x32, 4x64x16, 64x4x16 | 32x32x16, 32x32x32, 16x16x32, 16x16x64 | -- |
| gfx950 (MI355) | same as above | same as above + 16x16x128, 32x32x64 | 16x16x128 |

Code generation command:

```bash
# kernel
make generate_all_kernels

# English note
make generate_kernels_gfx942

# English note
make regenerate_all_kernels
```

---

## Pre-selected Kernel Instances

For common scenarios, CK Tile provides pre-selected kernel configurations, requiring no runtime ML inference:

- `gemm_multi_d` element-wise ops: `MultiDAdd`, `MultiDMultiply`, `Relu`, `Gelu`, `FastGelu`
- Default warp tile combinations for each GPU architecture are built into `arch_specs.json`

---

## Heuristic Fallback

When the ML model is unavailable, the Dispatcher falls back to rule-based strategies:

1. **FirstFit**: Iterates through all kernels in the Registry, selecting the first one for which `IsSupportedArgument` returns true
2. **Priority-based**: The Registry supports `Priority::High/Normal/Low`, with kernels sharing the same key sorted by priority
3. **Custom heuristic function**: Users can register custom sorting functions

---

## Supported GPU Architectures

| Architecture | GPU | ROCm Version |
|------|-----|-----------|
| gfx90a | MI200 (MI250/MI250X) | 6.0+ |
| gfx942 | MI300 (MI300X/MI300A/MI308/MI325) | 6.0+ |
| gfx950 | MI350 | 6.3+ |
| gfx1101 | RDNA3 | 6.0+ |
| gfx1201 | RDNA4 | 6.3+ |

---

## Key Takeaways

The following are experimental conclusions from `LEARNINGS.md`:

1. **Log-transform is essential**: Raw TFLOPS regression only achieves 84% efficiency on tiny-M (M=1) scenarios; `log1p(TFLOPS)` improves this to 96%
2. **Tiny-M shapes are the hardest**: When M=1, most kernel configurations perform extremely poorly (tile_m=128 wastes 127/128 of the space), and model noise exceeds the performance differences between kernels
3. **IHEM (Hard Example Mining) is not applicable**: For the scale mismatch problem, resampling actually degrades overall efficiency (94.31% -> 92.90%)
4. **Padding feature is critical**: `missing_required_padding` is one of the most important features, as it directly predicts whether a kernel will fail

## File Structure

```
dispatcher/
├── include/ck_tile/dispatcher/
│ ├── dispatcher.hpp # main Dispatcher
│   ├── registry.hpp          # Kernel Registry
│ ├── kernel_key.hpp # Kernel configuration
│ ├── kernel_instance.hpp # execute kernel
│ ├── ml_heuristic.hpp # C++ ML (LightGBM)
│   └── backends/             # Generated kernel backend
├── codegen/
│ ├── unified_gemm_codegen.py # GEMM kernel
│ └── arch_specs.json # GPU
├── heuristics/
│ ├── feature_engine.py # 72
│ ├── train.py # LightGBM
│ ├── predict.py # sort
│ ├── evaluate.py # evaluation
│   ├── search.py             # Surrogate search
│ ├── models/ # (.lgbm.gz)
│ ├── LEARNINGS.md # experiment
│ └── DATA_GENERATION.md # data
└── examples/
    └── gemm/
 ├── cpp/ # C++ example (01-06)
 └── python/ # Python example (01-11)
```

---

## Related

- [CK Quantized GEMM and MX Format](ck-quantization-mx.md) -- Dispatcher-managed quantized GEMM kernel
- [CK MoE / Norm / Conv Operators](ck-moe-norm-conv.md) -- More operator types to be covered by Dispatcher in the future
