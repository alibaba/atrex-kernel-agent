# Mamba / SSM State Management

## Pattern: Chunk-wise SSM State Update

**Source**: `flashinfer/ssd_chunk_state.py`

```python
@triton.jit
def chunk_state_kernel(B, C, dt, states, ...):
    """Mamba2 SSD: Update SSM state per chunk"""
    # Perform matrix operations within each chunk
    # Pass state between chunks (similar to RNN)
    
    # 1. Compute decay matrix within chunk
    decay = tl.exp(-dt * A)  # A is a diagonal matrix
    
    # 2. Accumulate state
    state = prev_state * decay + tl.dot(B_chunk.T, X_chunk)
    
    # 3. Compute output
    Y_chunk = tl.dot(C_chunk, state)
```

**Practical Experience**:
- Chunk size is typically set to 64-256, balancing parallelism and state precision
- The decay factor is computed using `exp` (not `exp2` as in attention), because SSM's A parameter does not require a log2 transform
- State dimension is typically 16-64, much smaller than attention's head_dim

---

## Related Documents

- **Same Series**: [Cascade / State Merge](cascade-state-merge.md) — attention state merge pattern
- **Same Series**: [Online Softmax and Flash Attention](online-softmax-flash-attention.md) — comparing chunk-wise computation of attention and SSM
- **Prerequisites**: [GPU Execution Model](../../../ref-docs/generic/gpu-execution-model.md) — chunk parallelism and warp scheduling
- **Hopper Practice**: [Hopper Optimization Practice](README.md) — implementation of linear attention on Hopper
- **Index**: [Triton Kernel Optimization Patterns Practice](README.md) — overview of all patterns
