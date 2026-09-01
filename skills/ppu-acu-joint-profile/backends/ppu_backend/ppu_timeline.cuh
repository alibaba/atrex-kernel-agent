// SPDX-License-Identifier: BSD-3-Clause
#ifndef PPU_ACU_PROFILE_TIMELINE_CUH_
#define PPU_ACU_PROFILE_TIMELINE_CUH_

#if !defined(PPU_TIMELINE_DEVICE_ONLY)
#include <cstddef>
#include <cstring>
#endif

namespace ppu_acu_profile {
namespace timeline {

using u8 = unsigned char;
using u16 = unsigned short;
using u32 = unsigned int;
using u64 = unsigned long long;

static_assert(sizeof(u16) == 2 && sizeof(u32) == 4 && sizeof(u64) == 8,
              "unsupported integer widths");

// Little-endian bytes: "PPUTL1\0\0".
constexpr u64 kMagic = 0x0000314c54555050ULL;
constexpr u16 kAbiMajor = 1;
constexpr u16 kAbiMinor = 0;
constexpr u32 kCommitted = 1U << 31U;
constexpr u32 kUserFlagsMask = 0x1fffU;

enum class Kind : u32 { begin = 0, end = 1, instant = 2, counter = 3 };

enum Status : u32 {
  ok = 0,
  overflow = 1U << 0,
  bad_header = 1U << 1,
  bad_owner = 1U << 2,
  duplicate_owner = 1U << 3,
};

struct alignas(16) TraceRecord {
  u64 raw_timestamp;
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
static_assert(sizeof(TraceBufferHeader) == 64, "trace header ABI changed");

#if !defined(PPU_TIMELINE_DEVICE_ONLY)

inline u64 allocation_bytes(u32 owner_count, u32 records_per_owner) {
  const u64 capacity = static_cast<u64>(owner_count) * records_per_owner;
  return sizeof(TraceBufferHeader) + capacity * sizeof(TraceRecord) +
         static_cast<u64>(owner_count) * sizeof(u64);
}

inline bool initialize_host_buffer(void* storage, u64 storage_bytes,
                                   u32 owner_count, u32 records_per_owner,
                                   u32 grid_x, u32 grid_y, u32 grid_z,
                                   u32 block_x, u32 block_y, u32 block_z,
                                   u64 launch_id) {
  const u64 capacity = static_cast<u64>(owner_count) * records_per_owner;
  const u64 required = allocation_bytes(owner_count, records_per_owner);
  if (storage == nullptr || owner_count == 0 || records_per_owner == 0 ||
      capacity > 0xffffffffULL || storage_bytes != required || grid_x == 0 ||
      grid_y == 0 || grid_z == 0 || block_x == 0 || block_y == 0 ||
      block_z == 0) {
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

#endif  // !PPU_TIMELINE_DEVICE_ONLY

#if defined(__HGGCCC__) || (defined(__clang__) && defined(__HGGC__))

#if defined(__HGGC_DEVICE_COMPILE__)

__device__ __forceinline__ u64 read_globaltimer() {
  u64 value;
  asm volatile("" : : : "memory");
  asm volatile("ppu.mov.u64 %0, %%globaltimer;\n" : "=l"(value) : : "memory");
  asm volatile("" : : : "memory");
  return value;
}

__device__ __forceinline__ u64 block_linear_id() {
  return (static_cast<u64>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x +
         blockIdx.x;
}

__device__ __forceinline__ u32 thread_linear_id() {
  return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
}

#endif  // __HGGC_DEVICE_COMPILE__

#if defined(PPU_TIMELINE_ENABLED) && defined(__HGGC_DEVICE_COMPILE__)

class Recorder {
 public:
  __device__ __forceinline__ Recorder(void* storage, u64 owner,
                                      bool selected = true)
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
      return;
    }
    if (owner_ >= header_->owner_count) {
      atomicOr(&header_->status, static_cast<u32>(bad_owner));
      enabled_ = false;
      return;
    }
    auto* records = reinterpret_cast<TraceRecord*>(
        reinterpret_cast<u8*>(header_) + header_->header_bytes);
    auto* claims = reinterpret_cast<u64*>(records + header_->capacity);
    const u64 block = block_linear_id();
    const u64 thread = thread_linear_id();
    if (block >= 0xffffffffULL || thread >= 0xffffffffULL) {
      atomicOr(&header_->status, static_cast<u32>(bad_owner));
      enabled_ = false;
      return;
    }
    const u64 token = ((block + 1ULL) << 32U) | (thread + 1ULL);
    if (atomicCAS(claims + owner_, 0ULL, token) != 0ULL) {
      atomicOr(&header_->status, static_cast<u32>(duplicate_owner));
      enabled_ = false;
    }
  }

  __device__ __forceinline__ void range_begin(u16 site, u32 payload = 0,
                                               u32 flags = 0) {
    emit(site, Kind::begin, payload, flags);
  }
  __device__ __forceinline__ void range_end(u16 site, u32 payload = 0,
                                             u32 flags = 0) {
    emit(site, Kind::end, payload, flags);
  }
  __device__ __forceinline__ void mark(u16 site, u32 payload = 0,
                                        u32 flags = 0) {
    emit(site, Kind::instant, payload, flags);
  }
  __device__ __forceinline__ void count(u16 site, u32 value,
                                         u32 flags = 0) {
    emit(site, Kind::counter, value, flags);
  }
  __device__ __forceinline__ u64 timestamp() const {
    return enabled_ ? read_globaltimer() : 0ULL;
  }
  __device__ __forceinline__ void range_at(u16 site, u64 begin_timestamp,
                                            u64 end_timestamp,
                                            u32 begin_payload = 0,
                                            u32 end_payload = 0,
                                            u32 flags = 0) {
    if (!enabled_) return;
    emit_at(site, Kind::begin, begin_timestamp, begin_payload, flags);
    emit_at(site, Kind::end, end_timestamp, end_payload, flags);
  }
  __device__ __forceinline__ void mark_at(u16 site, u64 event_timestamp,
                                           u32 payload = 0, u32 flags = 0) {
    emit_at(site, Kind::instant, event_timestamp, payload, flags);
  }
  __device__ __forceinline__ void count_at(u16 site, u64 event_timestamp,
                                            u32 value, u32 flags = 0) {
    emit_at(site, Kind::counter, event_timestamp, value, flags);
  }
  __device__ __forceinline__ bool enabled() const { return enabled_; }

 private:
  __device__ __forceinline__ void emit(u16 site, Kind kind, u32 payload,
                                       u32 user_flags) {
    if (!enabled_) return;
    emit_at(site, kind, read_globaltimer(), payload, user_flags);
  }

  __device__ __forceinline__ void emit_at(u16 site, Kind kind, u64 timestamp,
                                          u32 payload, u32 user_flags) {
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
    auto* records = reinterpret_cast<TraceRecord*>(
        reinterpret_cast<u8*>(header_) + header_->header_bytes);
    volatile TraceRecord* record = records + slot;
    record->raw_timestamp = timestamp;
    record->payload = payload;
    asm volatile("" : : : "memory");
    record->tag = static_cast<u32>(site) |
                  (static_cast<u32>(kind) << 16U) |
                  ((user_flags & kUserFlagsMask) << 18U) | kCommitted;
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
  __device__ __forceinline__ u64 timestamp() const { return 0ULL; }
  __device__ __forceinline__ void range_at(u16, u64, u64, u32 = 0,
                                            u32 = 0, u32 = 0) {}
  __device__ __forceinline__ void mark_at(u16, u64, u32 = 0, u32 = 0) {}
  __device__ __forceinline__ void count_at(u16, u64, u32, u32 = 0) {}
  __device__ __forceinline__ bool enabled() const { return false; }
};

#endif
#endif

}  // namespace timeline
}  // namespace ppu_acu_profile

#endif
