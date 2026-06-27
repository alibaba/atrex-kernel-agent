# Verification Guide

## Verification Levels

### Level 1: Syntax Verification (Fastest)

```bash
python tools/check_syntax.py generated.py
```

**What it checks**:
- ✅ Python syntax is correct
- ✅ Correct imports are used
- ✅ No prohibited APIs are used
- ✅ All tensor creation operations have layout

**Limitations**: Cannot detect semantic errors

---

### Level 2: Compilation Verification

```bash
python -c "import generated; print('✅ Compilation passed')"
```

**What it checks**:
- ✅ Gluon compilation passes
- ✅ No compilation errors

**Limitations**: Cannot verify functional correctness

---

### Level 3: Functional Verification (Most Reliable)

```bash
python tools/validate.py generated.py reference.py
```

**What it checks**:
- ✅ Output matches the reference code
- ✅ `torch.allclose(atol=1e-3, rtol=1e-1)` passes

**Requirements**: Reference code and test code are needed

---

### Level 4: Performance Verification (Final Acceptance)

```bash
python tools/benchmark.py generated.py reference.py \
    --wrapper-name chunk_gated_delta_rule_fwd_h_wrapper \
    --setup-name test_chunk_gdn
```

**What it checks**:
- ✅ Gluon kernel execution time is comparable to the Triton kernel
- ✅ Performance ratio (Gluon/Triton) falls within the 85%–115% range

**Principle**:
- Use `triton.testing.do_bench` to benchmark the wrapper functions on both sides
- Compare p50 (median) execution times to eliminate noise
- Automatically capture wrapper call arguments from the setup function via monkey-patching to ensure identical inputs on both sides

**Requirements**: Functional verification (Level 3) must pass first

**Custom Parameters**:
```bash
# Widen performance range
python tools/benchmark.py generated.py reference.py \
    --wrapper-name my_wrapper --setup-name my_setup \
    --low 0.85 --high 1.15

# Increase sampling count
python tools/benchmark.py generated.py reference.py \
    --wrapper-name my_wrapper --setup-name my_setup \
    --warmup 50 --rep 200
# Widen performance range
python tools/benchmark.py generated.py reference.py \
    --wrapper-name my_wrapper --setup-name my_setup \
    --low 0.85 --high 1.15

# Increase sampling count
python tools/benchmark.py generated.py reference.py \
    --wrapper-name my_wrapper --setup-name my_setup \
    --warmup 50 --rep 200

# Extract TTGIR
python tools/extract_ttgir.py kernel.py

# Save to file
python tools/extract_ttgir.py kernel.py -o output.ttgir
```

---

## Handling Verification Failures

### Scenario 1: Syntax Verification Failure

**Symptom**: `check_syntax.py` reports errors

**Handling Process**:
```
1. Check error line number
   → Locate the specific code line

2. Analyze error type
   → Syntax error / Type error / Import error

3. Fix syntax
   → Refer to correct usage in conversion-guide.md

4. Re-verify
   → Run check_syntax.py again
```

**Common Errors**:

| Error Type | Cause | Fix |
|------------|-------|-----|
| `SyntaxError` | Python syntax error | Check parentheses and indentation |
| `ImportError` | Incorrect import statement | Use correct import |
| `NameError` | Undefined variable used | Check variable names |

**Example Fix**:
```python
# ❌ Wrong: Missing layout parameter
idx = gl.arange(0, BLOCK_SIZE)

# ✅ Correct: Add layout parameter
layout: gl.constexpr = gl.BlockedLayout(...)
idx = gl.arange(0, BLOCK_SIZE, layout=layout)
```

---

### Scenario 2: Compilation Passes but Functionally Incorrect

**Symptom**: Compilation succeeds, but output does not match expectations

**Handling Process**:
```
1. Compare Triton/Gluon behavior differences
   → Check if API semantics match

2. Check data types
   → Does dtype match

3. Check memory layout
   → Is layout definition correct, fully faithful to TTGIR Layouts

4. Update the relevant architecture-specific API mapping notes
   → Fix incorrect mapping relationships for the target backend

5. Regenerate code
   → Use updated knowledge
```

**Debugging Tips**:
```python
# Print intermediate result comparison
print("Triton output:", output_triton)
print("Gluon output:", output_gluon)
print("Difference:", (output_triton - output_gluon).abs().max())
```

**Common Causes**:
1. API mapping errors — re-check official documentation
2. dtype mismatch — check type conversions
3. Inconsistent layout — extract the correct layout from TTGIR

---

### Scenario 3: Numerical Precision Mismatch

**Symptom**: `torch.allclose()` fails with significant differences

**Handling Process**:
```
1. Check accumulator dtype
   → Ensure using float32 for accumulation

2. Check layout
   → Ensure memory layout is consistent

3. Check rounding mode
   → Compare Triton/Gluon default behavior

4. Adjust tolerance
   → If reasonable, can relax atol/rtol
```

**Correct Example**:
```python
# ❌ Wrong: Directly accumulate with float16
accumulator = gl.zeros((M, N), dtype=gl.float16)

# ✅ Correct: Accumulate with float32, convert at end
accumulator = gl.zeros((M, N), dtype=gl.float32)
# After computation is complete
c = accumulator.to(gl.float16)
```

**Debugging Code**:
```python
# Check maximum difference
max_diff = (output_triton - output_gluon).abs().max()
print(f"Maximum difference: {max_diff}")

# Check difference locations
diff_mask = (output_triton - output_gluon).abs() > 1e-3
print(f"Difference location count: {diff_mask.sum().item()}")
```

### Scenario 4: Performance Validation Failure

**Symptom**: `benchmark.py` reports performance ratio outside the 90%-110% range

**Gluon is slower than Triton (ratio < 0.90) handling flow**:
```
1. Check Layout definition
   → Ensure all Layouts fully faithful to TTGIR extraction results

2. Check if pipeline is missing
   → If Triton source num_stages > 1, Gluon needs manual pipeline implementation

3. Check shared memory usage
   → Avoid unnecessary allocate_shared_memory calls

4. Re-run to eliminate noise
   → Increase --rep 200 --warmup 50 and retry
```

**Gluon is faster than Triton (ratio > 1.10) handling flow**:
```
1. Usually measurement noise
   → Increase --rep 200 --warmup 50 and retry

2. Confirm functional correctness
   → Re-run validate.py to ensure precision hasn't degraded
   → Abnormally fast performance may mean some computations were skipped

3. If indeed faster and functionally correct
   → Relax --high parameter, mark as passed
```

---

## Experimental Validation Methods

### Method 1: Comparative Testing

```python
import torch
import triton_kernel  # Triton version
import gluon_kernel   # Gluon version

# Prepare input
input_tensor = torch.randn(1024)

# Run Triton version
output_triton = triton_kernel.test_func(input_tensor)

# Run Gluon version
output_gluon = gluon_kernel.test_func(input_tensor)

# Compare
if torch.allclose(output_triton, output_gluon, atol=1e-4, rtol=1e-4):
    print("✅ Functionally consistent")
else:
    print("❌ Functionally inconsistent")
    print(f"Maximum difference: {(output_triton - output_gluon).abs().max()}")
```

---

### Method 2: Incremental Validation

```python
# Verify each component incrementally

# 1. Verify imports
from triton.experimental import gluon
print("✅ Import successful")

# 2. Verify basic API
pid = gluon.program_id(0)
print("✅ program_id available")

# 3. Verify complex API
layout = gluon.BlockedLayout(...)
idx = gluon.arange(0, 1024, layout=layout)
print("✅ arange + layout available")

# 4. Verify complete kernel
@gluon.jit
def test_kernel(...):
    # ...
    pass
print("✅ kernel definition successful")
```

---

## Validation Checklist

### Must-Check After Conversion

- [ ] Import statements are correct
- [ ] Function signatures are consistent
- [ ] All `tl.*` have been replaced with `gl.*`
- [ ] All `gl.arange` have layouts
- [ ] `tl.make_block_ptr` is not used
- [ ] Passes `check_syntax.py`
- [ ] Passes `validate.py` (if reference code is available)
- [ ] Passes `benchmark.py` (after functional validation passes)

### New API Validation

- [ ] Check official documentation for confirmation
- [ ] Write minimal test code
- [ ] Compilation passes
- [ ] Functional test passes
- [ ] Record in the relevant architecture-specific API mapping notes

---

## Common Validation Errors

### Error 1: Compilation Passes but Functionality Is Incorrect

**Cause**: API mapping error

**Resolution**:
1. Re-check official documentation
2. Compare behavioral differences between Triton and Gluon
3. Update the relevant architecture-specific API mapping notes

---

### Error 2: Numerical Precision Mismatch

**Cause**: Incorrect accumulator dtype

**Resolution**:
```python
# ❌ Wrong
accumulator = gl.zeros((M, N), dtype=gl.float16)

# ✅ Correct
accumulator = gl.zeros((M, N), dtype=gl.float32)
# After computation is complete
c = accumulator.to(gl.float16)
```

---

### Error 3: Large Performance Variance

**Cause**: Improper layout definition

**Resolution**:
1. Extract TTGIR to view the original layout
2. Compare Gluon's layout definition
3. Adjust parameters

```bash
# TTGIR
python tools/extract_ttgir.py kernel.py
```

---

## Validation Tool Usage

### check_syntax.py

```bash
# Basic usage
python tools/check_syntax.py generated.py

# Verbose output
python tools/check_syntax.py generated.py -v
```

### validate.py

```bash
# Basic usage
python tools/validate.py generated.py reference.py

# Specify variable name
python tools/validate.py generated.py reference.py --var-name result_gold

# Adjust tolerance
python tools/validate.py generated.py reference.py --atol 1e-2 --rtol 1e-0
```

### benchmark.py

```bash
# English note
python tools/benchmark.py generated.py reference.py \
    --wrapper-name my_wrapper --setup-name my_setup

# relaxperformance
python tools/benchmark.py generated.py reference.py \
    --wrapper-name my_wrapper --setup-name my_setup \
    --low 0.85 --high 1.15

# increasesamplingcount
python tools/benchmark.py generated.py reference.py \
    --wrapper-name my_wrapper --setup-name my_setup \
    --warmup 50 --rep 200
```

# Extract TTGIR
python tools/extract_ttgir.py kernel.py

# Basic usage
python tools/check_syntax.py generated.py

# Verbose output
python tools/check_syntax.py generated.py -v
