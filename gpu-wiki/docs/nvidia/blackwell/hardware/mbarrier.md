# mbarrier


**Last updated**: 2026-06-30

## Overview

mbarriers are 64-bit shared memory primitives used for producer/consumer synchronization between asynchronous hardware units (TMA, tcgen05) and SM threads. Introduced on Hopper, essential for Blackwell warp-specialized kernels.

## Key Operations

```ptx
// Initialize: set expected arrival count
mbarrier.init.shared.b64 [mbar_addr], num_arrivals;

// Producer: arrive on barrier (decrements expected count)
mbarrier.arrive.shared.b64 _, [mbar_addr];

// Producer (with byte expectation): used by TMA
mbarrier.arrive.expect_tx.shared.b64 _, [mbar_addr], expected_bytes;
// After this, TMA hardware completes the transaction by arriving with
// the transferred byte count, and the mbarrier flips parity.

// Consumer: wait for parity flip
mbarrier.try_wait.parity.shared.b64 p, [mbar_addr], phase;
@!p bra WAIT_LOOP;
```

## Phase/Parity Semantics

mbarriers have a 1-bit **phase** that flips each time the arrival count reaches zero. Consumers track their expected phase and wait for it:

```cuda
int phase = 0;
for (int k = 0; k < num_iterations; k++) {
    int stage = k % NUM_STAGES;

    // Wait for this stage's producer to complete
    mbarrier_wait_parity(&mbar[stage], phase);
    phase ^= 1;  // Flip for next use of this stage slot

    // Use data...
    consume_data(stage);

    // Arrive on next-stage barrier when done
    mbarrier_arrive(&buffer_free[stage]);
}
```

## TMA Integration

TMA directly signals mbarriers when transfer completes, avoiding manual polling:

```cuda
// Producer warp issues TMA with mbarrier target
if (lane_id == 0) {
    uint32_t tx_bytes = TILE_A_BYTES + TILE_B_BYTES;
    mbarrier_arrive_expect_tx(&mbar[stage], tx_bytes);

    // TMA completion will fire the mbarrier automatically
    cp_async_bulk_tensor_2d(smem_A[stage], global_desc, x, y, &mbar[stage]);
    cp_async_bulk_tensor_2d(smem_B[stage], global_desc, x, y, &mbar[stage]);
}
// Do NOT manually arrive after TMA issue - it races with hardware
```

## Pitfalls

1. **Missing phase flip**: Reusing a stage slot without flipping parity causes stale arrivals to satisfy the new wait
2. **Mismatched init count**: If consumer expects N arrivals but producers only arrive M times, barrier never fires
3. **Manual arrive after async issue**: TMA/tcgen05 hardware arrives on completion -- extra manual arrive causes double-count
4. **Cross-cluster mbarriers**: Use `shared::cluster` qualifier for cluster-level sync (2-SM cooperative MMA)

## Related
- [TMA](tma.md) -- Primary producer for mbarriers
- [warp-specialization](../techniques/warp-specialization.md) -- Uses mbarriers for warp role handoff
