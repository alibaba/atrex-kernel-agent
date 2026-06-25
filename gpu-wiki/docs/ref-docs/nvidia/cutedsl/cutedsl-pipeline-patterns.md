# CuTeDSL Software Pipeline and Synchronization Patterns

## Pipeline State Machine

Each pipeline entry (mbarrier) goes through the following state transitions:

```
         Producer                Consumer
         ┌──────┐               ┌──────┐
empty ──→│acquire│──→ writing ──→│commit│──→ full
         └──────┘               └──────┘
  ↑                                          │
  │         ┌──────┐               ┌────┐    │
  └─────────│release│←── reading ←─│wait│←───┘
            └──────┘               └────┘
```

Loop buffer slot states: **X** (empty) → **W** (writing) → **D** (data ready) → **R** (reading) → **X**

---

## Producer/Consumer High-Level Wrappers

### PipelineProducer

```python
producer = pipeline.make_producer()

# usemode
try_acquire_token = producer.try_acquire()
handle = producer.acquire_and_advance(try_acquire_token)
# ... data ...
handle.commit()
```

**Methods:**
- `try_acquire() → Boolean` — Non-blocking try-acquire
- `acquire(try_acquire_token) → ImmutableResourceHandle` — Blocking acquire empty buffer
- `acquire_and_advance(...)` — Acquire and advance state
- `commit(handle)` — Mark buffer full
- `tail()` — Prevent dangling mbarrier arrive signals after kernel exit

### PipelineConsumer

```python
consumer = pipeline.make_consumer()

try_wait_token = consumer.try_wait()
handle = consumer.wait_and_advance(try_wait_token)
# ... data ...
handle.release()
```

**Methods:**
- `try_wait() → Boolean` — Non-blocking try-wait
- `wait(try_wait_token) → ImmutableResourceHandle` — Blocking wait for full buffer
- `wait_and_advance(...)` — Wait and advance state
- `release(handle)` — Mark buffer empty

---

## PipelineState

```python
state = make_pipeline_state(PipelineUserType.Producer, stages=4)
```

Tracks position in the loop buffer:
- `index: Int32` — Current slot index
- `phase: Int32` — Current phase bit (0 or 1, for mbarrier flip)
- `stages: int` — Total number of stages
- `advance()` — Advance to next slot
- `reverse()` — Retreat to previous slot
- `clone()` — Deep copy

**Producer starts with phase=1** (empty buffer), **Consumer starts with phase=0** (waiting for full buffer).

---

## Synchronization Primitives

### MbarrierArray

Array of barriers in shared memory for producer-consumer synchronization:

```python
mbar_array = MbarrierArray(
    barrier_storage=ptr,
    num_stages=4,
    agent=(PipelineOp.TmaLoad, CooperativeGroup(Agent.ThreadBlock)),
 tx_count=1024 # bytes
)

mbar_array.mbarrier_init # warp 0
mbar_array.arrive(index, dst) #
mbar_array.arrive_and_expect_tx(index, tx_count) #
mbar_array.wait(index, phase) # wait
mbar_array.try_wait(index, phase) -> Boolean # wait
```

### NamedBarrier

Hardware-managed named barrier (ID 0-15):

```python
named_bar = NamedBarrier(barrier_id=1, num_threads=128)
named_bar.arrive()
named_bar.wait()
named_bar.arrive_and_wait # coalesced
named_bar.sync()
```

**Key difference:** NamedBarrier counts all participating threads; MbarrierArray only counts one side (producer or consumer).

### CooperativeGroup and Agent

```python
# Agent type
Agent.Thread #
Agent.ThreadBlock      # CTA
Agent.ThreadBlockCluster  # Cluster

# create cooperative group
group = CooperativeGroup(Agent.ThreadBlock, size=128, alignment=32)
```

### PipelineOp

Maps operations to hardware features:

| PipelineOp | Meaning |
|------------|---------|
| `AsyncThread` | General async thread operation |
| `TCGen05Mma` | Blackwell tensor core MMA |
| `TmaLoad` | TMA load |
| `TmaStore` | TMA store |
| `ClcLoad` | Cluster Launch Control load |
| `Composite` | Combined operation |## Pipeline Variants

### PipelineAsync (Base Class)

General-purpose pipeline, both producer and consumer are `AsyncThread` type.

```python
pipeline = PipelineAsync.create(
    num_stages=4,
    producer_group=CooperativeGroup(Agent.ThreadBlock, size=32),
    consumer_group=CooperativeGroup(Agent.ThreadBlock, size=128),
    barrier_storage=smem_ptr,
    defer_sync=False
)
```

### PipelineTmaAsync — SM90 Mainloop

**TMA producer + AsyncThread consumer**, typical Hopper pattern.

```python
pipeline = PipelineTmaAsync.create(
    num_stages=4,
    producer_group=producer_group,
    consumer_group=consumer_group,
 tx_count=1024, # stage bytes
    barrier_storage=smem_ptr,
 cta_layout_vmnk=cta_layout, # cluster
 mcast_mode_mn=(1, 1), # multicast mode
)
```

Characteristics:
- `producer_commit` is noop (TMA instructions automatically update transaction count)
- `consumer_release` conditionally signals empty buffers to the producer

### PipelineTmaUmma — Blackwell Mainloop

**TMA producer + UMMA consumer**.

```python
pipeline = PipelineTmaUmma.create(
    num_stages=4,
    producer_group=producer_group,
    consumer_group=consumer_group,
    tx_count=1024,
    cta_layout_vmnk=cta_layout,
    mcast_mode_mn=(1, 1),
)
```

Additional attributes: `is_leader_cta`, `cta_group` (1CTA/2CTA).

### PipelineCpAsync

**CpAsync producer + AsyncThread consumer**.

### PipelineAsyncUmma — Blackwell Input Fusion

**AsyncThread producer + UMMA consumer**.

### PipelineUmmaAsync — Blackwell Accumulator Pipeline

**UMMA producer + AsyncThread consumer**.

### PipelineClcFetchAsync — Cluster Launch Control

**CLC dynamic scheduling** pipeline, supports dynamically canceling unlaunched clusters.

### PipelineTmaMultiConsumersAsync

**TMA producer + UMMA consumer + AsyncThread consumer** (dual consumer).

```python
pipeline = PipelineTmaMultiConsumersAsync.create(
    num_stages=4,
    producer_group=producer_group,
    consumer_group_umma=umma_group,
    consumer_group_async=async_group,
    tx_count=1024,
)
```

`consumer_release` requires an additional `op_type: PipelineOp` parameter to distinguish between UMMA/AsyncThread consumers.

### PipelineTmaStore — Epilogue

TMA store synchronization, does not use mbarrier.

---

## Automatic Software Pipelining

### prefetch_stages Loop Attribute

```python
for i in cutlass.range(bound, prefetch_stages=3):
    cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
    use(buffer[i % total_stages])
```

The compiler automatically generates:
1. **Prefetch loop**: fills the first `prefetch_stages` buffers
2. **Main loop**: consumes the current buffer and prefetches the next one in each iteration
3. **Drain**: consumes the remaining filled buffers

**SM90+ only, experimental feature.**

---

## PipelineOrder: Ordered Execution of Multiple Groups

Manages ordered execution of multiple groups within a pipeline stage.

```python
order = PipelineOrder.create(
    barrier_storage=smem_ptr,
 depth=4, # stage
 length=2, # group total count
    group_id=my_id,
    producer_group=group,
)

order.arrive # current stage
order.wait # wait group complete
```

Use case: controlling alternating execution order of two consumer warp groups in Ping-Pong designs.

---

## Pipeline Initialization Patterns

```python
# mbarrier synchronous
pipeline.init_arrive(cluster_shape_mn=None)  # fence + cluster arrive
pipeline.init_wait(cluster_shape_mn=None)    # sync threadblock/cluster

# Agent synchronous
agent_sync(Agent.ThreadBlock)  # __syncthreads()
agent_sync(Agent.ThreadBlockCluster)  # cluster barrier
```## Choosing the Correct Pipeline

| Scenario | Pipeline Type | Producer | Consumer |
|------|-------------|----------|----------|
| General async data movement | `PipelineAsync` | AsyncThread | AsyncThread |
| SM80 cp.async movement | `PipelineCpAsync` | CpAsync | AsyncThread |
| SM90 TMA mainloop | `PipelineTmaAsync` | TmaLoad | AsyncThread |
| SM100 TMA+UMMA mainloop | `PipelineTmaUmma` | TmaLoad | TCGen05Mma |
| SM100 input fusion | `PipelineAsyncUmma` | AsyncThread | TCGen05Mma |
| SM100 accumulator pipeline | `PipelineUmmaAsync` | TCGen05Mma | AsyncThread |
| Dynamic cluster scheduling | `PipelineClcFetchAsync` | ClcLoad | AsyncThread |
| Epilogue TMA store | `PipelineTmaStore` | AsyncThread | — |
