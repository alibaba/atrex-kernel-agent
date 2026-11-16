// SPDX-License-Identifier: Apache-2.0
#ifndef ATREX_TIMELINE_CUH_
#define ATREX_TIMELINE_CUH_

#if !defined(__CUDACC_RTC__)
#include <cstddef>
#include <cstring>
#endif

namespace atrex {
namespace timeline {

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

static_assert(sizeof(u8) == 1 && sizeof(u16) == 2 && sizeof(u32) == 4 && sizeof(u64) == 8,
              "unsupported CUDA integer widths");

constexpr u64 kMagic = 0x31544c5845525441ULL;  // byte order: "ATREXLT1"
constexpr u16 kAbiMajor = 1;
constexpr u16 kAbiMinor = 0;
constexpr u32 kCommitted = 0x8U;

enum class Kind : u32 { begin = 0, end = 1, instant = 2, counter = 3 };

enum Status : u32 {
  ok = 0,
  overflow = 1U << 0,
  bad_header = 1U << 1,
  bad_owner = 1U << 2,
  bad_sm = 1U << 3,
  duplicate_owner = 1U << 4,
};

struct alignas(16) TraceRecord {
  u64 timestamp_ns;
  u32 payload;
  u32 tag;
};

struct alignas(16) TraceBufferHeader {
  u64 magic;
  u16 abi_major;
  u16 abi_minor;
  u16 header_bytes;
  u16 record_bytes;
  u32 capacity;
  u32 owner_count;
  u32 records_per_owner;
  u32 status;
  u32 grid_x;
  u32 grid_y;
  u32 grid_z;
  u32 block_x;
  u32 block_y;
  u32 block_z;
  u64 launch_id;
};

static_assert(sizeof(TraceRecord) == 16, "trace record ABI changed");
static_assert(sizeof(TraceBufferHeader) == 64, "trace buffer header ABI changed");

#if !defined(__CUDACC_RTC__)

inline u64 allocation_bytes(u32 owner_count, u32 records_per_owner) {
  const u64 capacity = static_cast<u64>(owner_count) * records_per_owner;
  return sizeof(TraceBufferHeader) + capacity * sizeof(TraceRecord) +
         static_cast<u64>(owner_count) * sizeof(u64);
}

inline bool initialize_host_buffer(void* storage, u64 storage_bytes, u32 owner_count,
                                   u32 records_per_owner, u32 grid_x, u32 grid_y,
                                   u32 grid_z, u32 block_x, u32 block_y, u32 block_z,
                                   u64 launch_id) {
  const u64 capacity = static_cast<u64>(owner_count) * records_per_owner;
  const u64 required = allocation_bytes(owner_count, records_per_owner);
  if (storage == nullptr || owner_count == 0 || records_per_owner == 0 ||
      capacity > 0xffffffffULL || storage_bytes != required || grid_x == 0 ||
      grid_y == 0 || grid_z == 0 || block_x == 0 || block_y == 0 || block_z == 0) {
    return false;
  }
  std::memset(storage, 0, static_cast<std::size_t>(storage_bytes));
  auto* header = static_cast<TraceBufferHeader*>(storage);
  header->magic = kMagic;
  header->abi_major = kAbiMajor;
  header->abi_minor = kAbiMinor;
  header->header_bytes = sizeof(TraceBufferHeader);
  header->record_bytes = sizeof(TraceRecord);
  header->capacity = static_cast<u32>(capacity);
  header->owner_count = owner_count;
  header->records_per_owner = records_per_owner;
  header->grid_x = grid_x;
  header->grid_y = grid_y;
  header->grid_z = grid_z;
  header->block_x = block_x;
  header->block_y = block_y;
  header->block_z = block_z;
  header->launch_id = launch_id;
  return true;
}

#endif  // !defined(__CUDACC_RTC__)

#if defined(__CUDACC__)

__device__ __forceinline__ u64 read_globaltimer_ns() {
  u64 result;
  asm volatile("" ::: "memory");
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result) :: "memory");
  asm volatile("" ::: "memory");
  return result;
}

__device__ __forceinline__ u32 read_smid() {
  u32 result;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(result));
  return result;
}

__device__ __forceinline__ u64 cta_linear_id() {
  return (static_cast<u64>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x +
         blockIdx.x;
}

__device__ __forceinline__ u32 thread_linear_id() {
  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ u32 warp_id() { return thread_linear_id() / 32U; }

__device__ __forceinline__ u64 cta_writer_owner(u32 writer_ordinal, u32 writers_per_cta) {
  return cta_linear_id() * writers_per_cta + writer_ordinal;
}

#if defined(ATREX_TIMELINE_ENABLED)

class Recorder {
 public:
  __device__ __forceinline__ Recorder(void* storage, u64 owner, bool selected = true)
      : header_(reinterpret_cast<TraceBufferHeader*>(storage)),
        owner_(owner),
        sequence_(0),
        enabled_(storage != nullptr && selected) {
    if (!enabled_) return;
    if (header_->magic != kMagic || header_->abi_major != kAbiMajor ||
        header_->header_bytes != sizeof(TraceBufferHeader) ||
        header_->record_bytes != sizeof(TraceRecord)) {
      atomicOr(&header_->status, static_cast<u32>(bad_header));
      enabled_ = false;
    } else if (owner_ >= header_->owner_count) {
      atomicOr(&header_->status, static_cast<u32>(bad_owner));
      enabled_ = false;
    } else {
      const u64 cta = cta_linear_id();
      const u64 thread = thread_linear_id();
      if (cta >= 0xffffffffULL) {
        atomicOr(&header_->status, static_cast<u32>(bad_owner));
        enabled_ = false;
        return;
      }
      TraceRecord* records = reinterpret_cast<TraceRecord*>(
          reinterpret_cast<u8*>(header_) + header_->header_bytes);
      u64* claims = reinterpret_cast<u64*>(records + header_->capacity);
      const u64 token = ((cta + 1ULL) << 32U) | (thread + 1ULL);
      const u64 previous = atomicCAS(claims + owner_, 0ULL, token);
      if (previous != 0ULL) {
        atomicOr(&header_->status, static_cast<u32>(duplicate_owner));
        enabled_ = false;
      }
    }
  }

  __device__ __forceinline__ void range_begin(u16 site, u32 payload = 0, u32 flags = 0) {
    emit(site, Kind::begin, payload, flags);
  }
  __device__ __forceinline__ void range_end(u16 site, u32 payload = 0, u32 flags = 0) {
    emit(site, Kind::end, payload, flags);
  }
  __device__ __forceinline__ void mark(u16 site, u32 payload = 0, u32 flags = 0) {
    emit(site, Kind::instant, payload, flags);
  }
  __device__ __forceinline__ void count(u16 site, u32 value, u32 flags = 0) {
    emit(site, Kind::counter, value, flags);
  }

  __device__ __forceinline__ bool enabled() const { return enabled_; }
  __device__ __forceinline__ u32 next_sequence() const { return sequence_; }

 private:
  __device__ __forceinline__ void emit(u16 site, Kind kind, u32 payload, u32 user_flags) {
    if (!enabled_) return;
    const u32 sequence = sequence_++;
    if (sequence >= header_->records_per_owner) {
      atomicOr(&header_->status, static_cast<u32>(overflow));
      enabled_ = false;
      return;
    }
    const u64 slot = owner_ * header_->records_per_owner + sequence;
    if (slot >= header_->capacity) {
      atomicOr(&header_->status, static_cast<u32>(overflow));
      enabled_ = false;
      return;
    }
    const u32 sm = read_smid();
    if (sm >= 1024U) {
      atomicOr(&header_->status, static_cast<u32>(bad_sm));
      enabled_ = false;
      return;
    }
    const u32 tag = static_cast<u32>(site) | (static_cast<u32>(kind) << 16U) | (sm << 18U) |
                         (((user_flags & 0x7U) | kCommitted) << 28U);
    TraceRecord* records = reinterpret_cast<TraceRecord*>(
        reinterpret_cast<u8*>(header_) + header_->header_bytes);
    records[slot] = TraceRecord{read_globaltimer_ns(), payload, tag};
  }

  TraceBufferHeader* header_;
  u64 owner_;
  u32 sequence_;
  bool enabled_;
};

#else

class Recorder {
 public:
  __device__ __forceinline__ Recorder(void*, u64, bool = true) {}
  __device__ __forceinline__ void range_begin(u16, u32 = 0, u32 = 0) {}
  __device__ __forceinline__ void range_end(u16, u32 = 0, u32 = 0) {}
  __device__ __forceinline__ void mark(u16, u32 = 0, u32 = 0) {}
  __device__ __forceinline__ void count(u16, u32, u32 = 0) {}
  __device__ __forceinline__ bool enabled() const { return false; }
  __device__ __forceinline__ u32 next_sequence() const { return 0; }
};

#endif  // ATREX_TIMELINE_ENABLED
#endif  // __CUDACC__

}  // namespace timeline
}  // namespace atrex

#endif  // ATREX_TIMELINE_CUH_
