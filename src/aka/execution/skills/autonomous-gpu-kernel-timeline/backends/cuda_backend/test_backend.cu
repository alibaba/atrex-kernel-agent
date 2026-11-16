// SPDX-License-Identifier: Apache-2.0
#include "atrex_timeline.cuh"

#include <cuda_runtime.h>
#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tl = atrex::timeline;

__global__ void probe_kernel(void* storage) {
  const uint32_t warp = tl::warp_id();
  const uint32_t lane = tl::thread_linear_id() & 31U;
  const bool selected = lane == 0 && (warp == 0 || warp == 2);
  const uint32_t ordinal = warp == 0 ? 0 : 1;
  tl::Recorder recorder(storage, tl::cta_writer_owner(ordinal, 2), selected);
  recorder.range_begin(7, warp);
  if (selected) {
    const uint64_t start = clock64();
    while (clock64() - start < 128) {
    }
  }
  recorder.range_end(7, warp);
  recorder.mark(9, blockIdx.x);
}

__global__ void duplicate_owner_kernel(void* storage) {
  const bool selected = threadIdx.x < 2;
  tl::Recorder recorder(storage, 0, selected);
  recorder.mark(9, threadIdx.x);
}

__global__ void zero_probe_kernel(void* storage) {
  const bool selected = threadIdx.x == 0;
  tl::Recorder recorder(storage, 0, selected);
}

__global__ void one_range_kernel(void* storage) {
  const bool selected = threadIdx.x == 0;
  tl::Recorder recorder(storage, 0, selected);
  recorder.range_begin(7);
  recorder.range_end(7);
}

__global__ void matrix_kernel(void* storage) {
  const bool selected = threadIdx.x == 0;
  tl::Recorder recorder(storage, tl::cta_writer_owner(0, 1), selected);
  for (uint32_t iteration = 0; iteration < 2; ++iteration) {
    recorder.range_begin(11, iteration);
    if ((tl::cta_linear_id() + iteration) & 1ULL) {
      recorder.mark(12, iteration);
    } else {
      recorder.count(13, iteration);
    }
    recorder.range_end(11, iteration);
  }
}

static void checked(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
    std::exit(2);
  }
}

enum class TestMode { normal, duplicate_owner, zero_probe, one_range, overflow, matrix };

int main(int argc, char** argv) {
  TestMode mode = TestMode::normal;
  const char* output_path = nullptr;
  if (argc == 2 && std::strcmp(argv[1], "--duplicate-owner") == 0) {
    mode = TestMode::duplicate_owner;
  } else if (argc == 2 && std::strcmp(argv[1], "--zero-probe") == 0) {
    mode = TestMode::zero_probe;
  } else if (argc == 2 && std::strcmp(argv[1], "--one-range") == 0) {
    mode = TestMode::one_range;
  } else if (argc == 2 && std::strcmp(argv[1], "--overflow") == 0) {
    mode = TestMode::overflow;
  } else if (argc == 3 && std::strcmp(argv[1], "--matrix") == 0) {
    mode = TestMode::matrix;
    output_path = argv[2];
  } else if (argc == 2) {
    output_path = argv[1];
  } else if (argc != 1) {
    std::fprintf(stderr, "unexpected arguments\n");
    return 1;
  }

  dim3 grid(2, 1, 1);
  dim3 block(128, 1, 1);
  uint32_t owners = 4;
  uint32_t records_per_owner = 3;
  if (mode == TestMode::duplicate_owner || mode == TestMode::zero_probe ||
      mode == TestMode::one_range || mode == TestMode::overflow) {
    grid = dim3(1, 1, 1);
    block = dim3(32, 1, 1);
    owners = 1;
    records_per_owner = mode == TestMode::one_range ? 2 : (mode == TestMode::overflow ? 5 : 1);
  } else if (mode == TestMode::matrix) {
    grid = dim3(2, 2, 2);
    block = dim3(32, 1, 1);
    owners = 8;
    records_per_owner = 6;
  }
  const size_t bytes = tl::allocation_bytes(owners, records_per_owner);
  void* storage = nullptr;
  checked(cudaMallocManaged(&storage, bytes), "cudaMallocManaged");
  if (!tl::initialize_host_buffer(storage, bytes, owners, records_per_owner,
                                  grid.x, grid.y, grid.z, block.x, block.y, block.z,
                                  0x1234)) {
    std::fprintf(stderr, "timeline host buffer initialization failed\n");
    return 9;
  }
  auto* header = static_cast<tl::TraceBufferHeader*>(storage);

  switch (mode) {
    case TestMode::duplicate_owner:
      duplicate_owner_kernel<<<grid, block>>>(storage);
      break;
    case TestMode::zero_probe:
      zero_probe_kernel<<<grid, block>>>(storage);
      break;
    case TestMode::one_range:
      one_range_kernel<<<grid, block>>>(storage);
      break;
    case TestMode::overflow:
    case TestMode::matrix:
      matrix_kernel<<<grid, block>>>(storage);
      break;
    default:
      probe_kernel<<<grid, block>>>(storage);
      break;
  }
  checked(cudaGetLastError(), "timeline test kernel launch");
  checked(cudaDeviceSynchronize(), "timeline test kernel synchronize");
#if defined(ATREX_TIMELINE_ENABLED)
  if (mode == TestMode::duplicate_owner) {
    if (header->status != tl::duplicate_owner) {
      std::fprintf(stderr, "duplicate owner status is 0x%x, expected 0x%x\n", header->status,
                   static_cast<unsigned>(tl::duplicate_owner));
      return 3;
    }
    checked(cudaFree(storage), "cudaFree");
    std::puts("cuda timeline duplicate-owner detection: ok");
    return 0;
  }
  if (mode == TestMode::overflow) {
    if (header->status != tl::overflow) {
      std::fprintf(stderr, "overflow status is 0x%x, expected 0x%x\n", header->status,
                   static_cast<unsigned>(tl::overflow));
      return 3;
    }
    checked(cudaFree(storage), "cudaFree");
    std::puts("cuda timeline capacity detection: ok");
    return 0;
  }
#endif
  if (header->status != 0) {
    std::fprintf(stderr, "device status is 0x%x\n", header->status);
    return 3;
  }
#if defined(ATREX_TIMELINE_ENABLED)
  const uint32_t capacity = owners * records_per_owner;
  const auto* records = reinterpret_cast<const tl::TraceRecord*>(
      static_cast<const uint8_t*>(storage) + sizeof(tl::TraceBufferHeader));
  const auto* claims = reinterpret_cast<const unsigned long long*>(records + capacity);
  if (mode == TestMode::zero_probe) {
    if (records[0].timestamp_ns || records[0].payload || records[0].tag || claims[0] == 0) {
      std::fprintf(stderr, "zero-probe mode changed records or failed to claim its owner\n");
      return 4;
    }
  } else {
    for (uint32_t owner = 0; owner < owners; ++owner) {
      const tl::TraceRecord* row = records + owner * records_per_owner;
      if (claims[owner] == 0) {
        std::fprintf(stderr, "owner %u was not claimed\n", owner);
        return 4;
      }
      for (uint32_t sequence = 0; sequence < records_per_owner; ++sequence) {
        if (((row[sequence].tag >> 28U) & tl::kCommitted) == 0 ||
            row[sequence].timestamp_ns == 0) {
          std::fprintf(stderr, "owner %u sequence %u was not recorded\n", owner, sequence);
          return 4;
        }
      }
      for (uint32_t sequence = 1; sequence < records_per_owner; ++sequence) {
        if (row[sequence - 1].timestamp_ns > row[sequence].timestamp_ns) {
          std::fprintf(stderr, "owner %u timestamps are not monotonic\n", owner);
          return 6;
        }
      }
      if (mode == TestMode::normal &&
          ((row[0].tag & 0xffffU) != 7 || ((row[0].tag >> 16U) & 3U) != 0 ||
           (row[1].tag & 0xffffU) != 7 || ((row[1].tag >> 16U) & 3U) != 1 ||
           (row[2].tag & 0xffffU) != 9 || ((row[2].tag >> 16U) & 3U) != 2)) {
        std::fprintf(stderr, "owner %u has unexpected site/kind tags\n", owner);
        return 5;
      }
    }
  }
#else
  const auto* tail = static_cast<const uint8_t*>(storage) + sizeof(tl::TraceBufferHeader);
  for (size_t offset = 0; offset < bytes - sizeof(tl::TraceBufferHeader); ++offset) {
    if (tail[offset]) {
      std::fprintf(stderr, "disabled backend wrote byte %zu\n", offset);
      return 7;
    }
  }
#endif
  if (output_path) {
    FILE* output = std::fopen(output_path, "wb");
    if (!output || std::fwrite(storage, 1, bytes, output) != bytes || std::fclose(output)) {
      std::fprintf(stderr, "could not write raw output\n");
      return 8;
    }
  }
  checked(cudaFree(storage), "cudaFree");
  std::puts("cuda timeline backend: ok");
  return 0;
}
