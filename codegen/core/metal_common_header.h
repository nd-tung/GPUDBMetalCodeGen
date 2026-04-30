#pragma once
// ===================================================================
// Metal Common Header — SIMD reductions, bitmap ops, atomics
// ===================================================================
// This string is prepended to every generated Metal shader.
// Extracted from MetalCodegen::commonHeader() for maintainability.
//
// 64-BIT ATOMIC REPRESENTATION (lo/hi pair scheme)
// -------------------------------------------------
// Apple Metal exposes atomics only on 32-bit lanes (atomic_uint /
// atomic_int). To accumulate 64-bit values we store every logical
// `long` as TWO adjacent 32-bit slots:
//
//      slot[i+0]  =  lo word  (bits  0..31, unsigned)
//      slot[i+1]  =  hi word  (bits 32..63, signed when reconstructed)
//
//   atomic_add_long_pair(lo, hi, v)  — add the lo half, propagate the
//       carry to the hi half via a second atomic.
//   atomic_max_long_pair(lo, hi, v)  — CAS loop on the hi word, then
//       store the lo word once a strictly-greater hi is observed.
//   load_long_pair(lo, hi)           — non-atomic reconstruction
//       (((int64_t)hi << 32) | lo) for read-back on the host.
//
// The same scheme is used for SIMD reductions of `long`: each lane
// shuffles its two 32-bit halves separately and reassembles after the
// shuffle (see simd_reduce_add_long / simd_reduce_max_long below).
//
// SIMD WIDTH ASSUMPTION
// ---------------------
// All threadgroup reductions assume Apple's fixed 32-thread SIMD group
// (hence the `& 31u` lane mask, `>> 5u` group index, and the 32-slot
// `threadgroup` arrays in tg_reduce_*). This is correct for every
// Apple Silicon GPU we target; it would need revisiting on a port to a
// platform with a different SIMD width.
// ===================================================================

namespace codegen {

inline const char* kMetalCommonHeader = R"METAL(#include <metal_stdlib>
using namespace metal;

// --- SIMD reduction for long (int64) via 2×uint shuffle ---
inline long simd_reduce_add_long(long v) {
    for (uint d = 16; d >= 1; d >>= 1) {
        uint lo = simd_shuffle_down((uint)(v), d);
        uint hi = simd_shuffle_down((uint)((ulong)v >> 32), d);
        v += (long)(((ulong)hi << 32) | (ulong)lo);
    }
    return v;
}

inline float tg_reduce_float(float val, uint tid, uint tg_size,
                             threadgroup float* shared) {
    float sv = simd_sum(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    if (gid == 0u) {
        float v2 = (lane < ng) ? shared[lane] : 0.0f;
        r = simd_sum(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline uint tg_reduce_uint(uint val, uint tid, uint tg_size,
                           threadgroup uint* shared) {
    uint sv = simd_sum(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = 0u;
    if (gid == 0u) {
        uint v2 = (lane < ng) ? shared[lane] : 0u;
        r = simd_sum(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline long tg_reduce_long(long val, uint tid, uint tg_size,
                           threadgroup long* shared) {
    long sv = simd_reduce_add_long(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long r = 0;
    if (gid == 0u) {
        long v2 = (lane < ng) ? shared[lane] : 0;
        r = simd_reduce_add_long(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline bool bitmap_test(const device uint* bitmap, int key) {
    return (bitmap[(uint)key >> 5] >> ((uint)key & 31u)) & 1u;
}

inline void bitmap_set(device atomic_uint* bitmap, int key) {
    atomic_fetch_or_explicit(&bitmap[(uint)key >> 5],
                             1u << ((uint)key & 31u),
                             memory_order_relaxed);
}

inline void atomic_add_long_pair(device atomic_uint* lo,
                                 device atomic_uint* hi,
                                 long val) {
    ulong uval = as_type<ulong>(val);
    uint add_lo = (uint)(uval);
    uint add_hi = (uint)(uval >> 32);
    uint old_lo = atomic_fetch_add_explicit(lo, add_lo, memory_order_relaxed);
    uint new_lo = old_lo + add_lo;
    uint carry = (new_lo < old_lo) ? 1u : 0u;
    if (add_hi != 0 || carry != 0)
        atomic_fetch_add_explicit(hi, add_hi + carry, memory_order_relaxed);
}

inline long load_long_pair(const device uint* lo, const device uint* hi) {
    ulong v = ((ulong)(*hi) << 32) | (ulong)(*lo);
    return as_type<long>(v);
}

inline void atomic_add_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float new_f = as_type<float>(old_val) + val;
        uint new_val = as_type<uint>(new_f);
        if (atomic_compare_exchange_weak_explicit(addr, &old_val, new_val,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) break;
    }
}

inline float load_float_atomic(const device uint* addr) {
    return as_type<float>(*addr);
}

inline uint next_pow2(uint v) {
    v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// --- SIMD max reduction for long via 2×uint shuffle ---
inline long simd_reduce_max_long(long v) {
    for (uint d = 16; d >= 1; d >>= 1) {
        uint lo = simd_shuffle_down((uint)(v), d);
        uint hi = simd_shuffle_down((uint)((ulong)v >> 32), d);
        long other = (long)(((ulong)hi << 32) | (ulong)lo);
        v = (other > v) ? other : v;
    }
    return v;
}

inline long tg_reduce_max_long(long val, uint tid, uint tg_size,
                               threadgroup long* shared) {
    long sv = simd_reduce_max_long(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long r = 0;
    if (gid == 0u) {
        long v2 = (lane < ng) ? shared[lane] : 0;
        r = simd_reduce_max_long(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// --- Atomic max for long via CAS on lo/hi uint pair ---
inline void atomic_max_long_pair(device atomic_uint* lo,
                                 device atomic_uint* hi,
                                 long val) {
    // Simple approach: atomic CAS loop on hi word, then lo word
    // For correctness with 64-bit pairs, we use a two-word CAS pattern
    ulong uval = as_type<ulong>(val);
    uint new_lo = (uint)(uval);
    uint new_hi = (uint)(uval >> 32);

    while (true) {
        uint old_hi = atomic_load_explicit(hi, memory_order_relaxed);
        uint old_lo = atomic_load_explicit(lo, memory_order_relaxed);
        long old_val = as_type<long>(((ulong)old_hi << 32) | (ulong)old_lo);
        if (val <= old_val) return; // already bigger
        // Try to update hi first
        if (atomic_compare_exchange_weak_explicit(hi, &old_hi, new_hi,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
            atomic_store_explicit(lo, new_lo, memory_order_relaxed);
            return;
        }
    }
}

)METAL";

} // namespace codegen
