# ROCprof Trace Decoder

## Description

A plugin library for rocprofiler-sdk: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk

Thread trace is a profiling method that provides fine-grained insight into GPU kernel execution by collecting detailed traces of shader instructions executed by the GPU. This feature captures GPU occupancy, instruction execution times, fast performance counters, and other detailed performance data. Thread trace utilizes GPU hardware instrumentation to record events as they happen, resulting in precise timing information about wave (threads) execution behavior.

This library is responsible for transforming thread trace binary data (.att) into a tool consumable format.

## Usage

### rocprofv3 tool

```bash
rocprofv3 --att -- ./a.out
```

By default, rocprofv3 searches ``LD_LIBRARY_PATH`` and the rocprofiler-sdk install location, normally ``/opt/rocm/lib``. For custom install locations, use

```bash
rocprofv3 --att --att-library-path /path/to/lib -- ./a.out
```

For information on how to generate thread trace data, see the documentation on [using rocprofv3 to collect thread trace](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html)

### Rocprofiler-sdk API

Rocprofiler-sdk requires the library path to be provided at resource creation:

```bash
rocprofiler_thread_trace_decoder_handle_t decoder{};
# Notes: Passing null string "" searches in LD_LIBRARY_PATH. Passing nullptr is not allowed.
auto status = rocprofiler_thread_trace_decoder_create(&decoder, "/opt/rocm/lib");
```

For more API information, see the [ROCm docs](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/thread_trace.html), [SDK Samples](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-sdk/samples/thread_trace/agent.cpp) and [SDK API](https://github.com/ROCm/rocprofiler-sdk/tree/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace)

### Supported devices

- AMD Radeon: 6000, 7000, 9000 series
- AMD Instinct: MI200 and MI300 series

## End User License Agreement

- Headers are provided under the MIT license.
- The .so and .dylib binaries use a custom LICENSE
