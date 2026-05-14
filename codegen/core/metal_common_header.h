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

inline float tg_reduce_min_float(float val, uint tid, uint tg_size,
                                 threadgroup float* shared) {
    float sv = simd_min(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 3.402823466e+38f;
    if (gid == 0u) {
        float v2 = (lane < ng) ? shared[lane] : 3.402823466e+38f;
        r = simd_min(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline float tg_reduce_max_float(float val, uint tid, uint tg_size,
                                 threadgroup float* shared) {
    float sv = simd_max(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = -3.402823466e+38f;
    if (gid == 0u) {
        float v2 = (lane < ng) ? shared[lane] : -3.402823466e+38f;
        r = simd_max(v2);
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

inline int tg_reduce_min_int(int val, uint tid, uint tg_size,
                             threadgroup int* shared) {
    int sv = simd_min(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int r = 2147483647;
    if (gid == 0u) {
        int v2 = (lane < ng) ? shared[lane] : 2147483647;
        r = simd_min(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline int tg_reduce_max_int(int val, uint tid, uint tg_size,
                             threadgroup int* shared) {
    int sv = simd_max(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int r = -2147483647;
    if (gid == 0u) {
        int v2 = (lane < ng) ? shared[lane] : -2147483647;
        r = simd_max(v2);
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

inline long simd_reduce_min_long(long v) {
    for (uint d = 16; d >= 1; d >>= 1) {
        uint lo = simd_shuffle_down((uint)(v), d);
        uint hi = simd_shuffle_down((uint)((ulong)v >> 32), d);
        long other = (long)(((ulong)hi << 32) | (ulong)lo);
        v = (other < v) ? other : v;
    }
    return v;
}

inline long tg_reduce_min_long(long val, uint tid, uint tg_size,
                               threadgroup long* shared) {
    long sv = simd_reduce_min_long(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long r = 9223372036854775807L;
    if (gid == 0u) {
        long v2 = (lane < ng) ? shared[lane] : 9223372036854775807L;
        r = simd_reduce_min_long(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline bool bitmap_test(const device uint* bitmap, int key) {
    return (bitmap[(uint)key >> 5] >> ((uint)key & 31u)) & 1u;
}
inline bool bitmap_test_atomic(const device atomic_uint* bitmap, int key) {
    uint word = atomic_load_explicit(&bitmap[(uint)key >> 5], memory_order_relaxed);
    return (word >> ((uint)key & 31u)) & 1u;
}

inline char packed_pattern_char(ulong lo, ulong hi, uint pos) {
    ulong word = (pos < 8u) ? lo : hi;
    uint shift = (pos < 8u) ? (pos * 8u) : ((pos - 8u) * 8u);
    return (char)((word >> shift) & 0xffu);
}

inline bool fixed_string_padding_ok(const device char* value, uint width, uint start) {
    for (uint i = start; i < width; i++) {
        char ch = value[i];
        if (ch != '\0' && ch != ' ') return false;
    }
    return true;
}

inline bool fixed_string_segment_eq(const device char* value,
                                    uint width,
                                    uint start,
                                    ulong seg_lo,
                                    ulong seg_hi,
                                    uint seg_len) {
    if (start + seg_len > width) return false;
    for (uint i = 0; i < seg_len; i++) {
        if (value[start + i] != packed_pattern_char(seg_lo, seg_hi, i)) return false;
    }
    return true;
}

inline bool fixed_like_one_segment(const device char* data,
                                   uint row,
                                   uint width,
                                   ulong seg_lo,
                                   ulong seg_hi,
                                   uint seg_len,
                                   bool leading_wildcard,
                                   bool trailing_wildcard) {
    if (seg_len > width) return false;
    const device char* value = data + row * width;
    uint last_start = leading_wildcard ? (width - seg_len) : 0u;
    for (uint start = 0u; start <= last_start; start++) {
        if (fixed_string_segment_eq(value, width, start, seg_lo, seg_hi, seg_len) &&
            (trailing_wildcard || fixed_string_padding_ok(value, width, start + seg_len))) {
            return true;
        }
    }
    return false;
}

inline bool fixed_like_two_segment(const device char* data,
                                   uint row,
                                   uint width,
                                   ulong first_lo,
                                   ulong first_hi,
                                   uint first_len,
                                   ulong second_lo,
                                   ulong second_hi,
                                   uint second_len,
                                   bool leading_wildcard,
                                   bool trailing_wildcard) {
    if (first_len + second_len > width) return false;
    const device char* value = data + row * width;
    uint first_last = leading_wildcard ? (width - first_len - second_len) : 0u;
    for (uint first_start = 0u; first_start <= first_last; first_start++) {
        if (!fixed_string_segment_eq(value, width, first_start, first_lo, first_hi, first_len)) {
            if (!leading_wildcard) break;
            continue;
        }
        uint second_start_min = first_start + first_len;
        uint second_start_max = width - second_len;
        for (uint second_start = second_start_min; second_start <= second_start_max; second_start++) {
            if (fixed_string_segment_eq(value, width, second_start, second_lo, second_hi, second_len) &&
                (trailing_wildcard || fixed_string_padding_ok(value, width, second_start + second_len))) {
                return true;
            }
        }
        if (!leading_wildcard) break;
    }
    return false;
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

inline void atomic_min_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float old_f = as_type<float>(old_val);
        if (val >= old_f) break;
        uint new_val = as_type<uint>(val);
        if (atomic_compare_exchange_weak_explicit(addr, &old_val, new_val,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) break;
    }
}

inline void atomic_max_float(device atomic_uint* addr, float val) {
    uint old_val = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float old_f = as_type<float>(old_val);
        if (val <= old_f) break;
        uint new_val = as_type<uint>(val);
        if (atomic_compare_exchange_weak_explicit(addr, &old_val, new_val,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) break;
    }
}

inline void atomic_min_float_seen(device atomic_uint* addr,
                                  device atomic_uint* state,
                                  float val) {
    uint expected = 0u;
    if (atomic_compare_exchange_weak_explicit(state, &expected, 1u,
                                               memory_order_relaxed,
                                               memory_order_relaxed)) {
        atomic_store_explicit(addr, as_type<uint>(val), memory_order_relaxed);
        atomic_store_explicit(state, 2u, memory_order_relaxed);
        return;
    }
    while (atomic_load_explicit(state, memory_order_relaxed) != 2u) {}
    atomic_min_float(addr, val);
}

inline void atomic_max_float_seen(device atomic_uint* addr,
                                  device atomic_uint* state,
                                  float val) {
    uint expected = 0u;
    if (atomic_compare_exchange_weak_explicit(state, &expected, 1u,
                                               memory_order_relaxed,
                                               memory_order_relaxed)) {
        atomic_store_explicit(addr, as_type<uint>(val), memory_order_relaxed);
        atomic_store_explicit(state, 2u, memory_order_relaxed);
        return;
    }
    while (atomic_load_explicit(state, memory_order_relaxed) != 2u) {}
    atomic_max_float(addr, val);
}

inline void atomic_min_int_seen(device atomic_int* addr,
                                device atomic_uint* state,
                                int val) {
    uint expected = 0u;
    if (atomic_compare_exchange_weak_explicit(state, &expected, 1u,
                                               memory_order_relaxed,
                                               memory_order_relaxed)) {
        atomic_store_explicit(addr, val, memory_order_relaxed);
        atomic_store_explicit(state, 2u, memory_order_relaxed);
        return;
    }
    while (atomic_load_explicit(state, memory_order_relaxed) != 2u) {}
    int old_val = atomic_load_explicit(addr, memory_order_relaxed);
    while (val < old_val) {
        if (atomic_compare_exchange_weak_explicit(addr, &old_val, val,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) break;
    }
}

inline void atomic_max_int_seen(device atomic_int* addr,
                                device atomic_uint* state,
                                int val) {
    uint expected = 0u;
    if (atomic_compare_exchange_weak_explicit(state, &expected, 1u,
                                               memory_order_relaxed,
                                               memory_order_relaxed)) {
        atomic_store_explicit(addr, val, memory_order_relaxed);
        atomic_store_explicit(state, 2u, memory_order_relaxed);
        return;
    }
    while (atomic_load_explicit(state, memory_order_relaxed) != 2u) {}
    int old_val = atomic_load_explicit(addr, memory_order_relaxed);
    while (val > old_val) {
        if (atomic_compare_exchange_weak_explicit(addr, &old_val, val,
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

inline void atomic_min_long_pair(device atomic_uint* lo,
                                 device atomic_uint* hi,
                                 long val) {
    ulong uval = as_type<ulong>(val);
    uint new_lo = (uint)(uval);
    uint new_hi = (uint)(uval >> 32);

    while (true) {
        uint old_hi = atomic_load_explicit(hi, memory_order_relaxed);
        uint old_lo = atomic_load_explicit(lo, memory_order_relaxed);
        long old_val = as_type<long>(((ulong)old_hi << 32) | (ulong)old_lo);
        if (val >= old_val) return;
        if (atomic_compare_exchange_weak_explicit(hi, &old_hi, new_hi,
                                                   memory_order_relaxed,
                                                   memory_order_relaxed)) {
            atomic_store_explicit(lo, new_lo, memory_order_relaxed);
            return;
        }
    }
}

inline void atomic_min_long_pair_seen(device atomic_uint* lo,
                                      device atomic_uint* hi,
                                      device atomic_uint* state,
                                      long val) {
    uint expected = 0u;
    if (atomic_compare_exchange_weak_explicit(state, &expected, 1u,
                                               memory_order_relaxed,
                                               memory_order_relaxed)) {
        ulong uval = as_type<ulong>(val);
        atomic_store_explicit(lo, (uint)(uval), memory_order_relaxed);
        atomic_store_explicit(hi, (uint)(uval >> 32), memory_order_relaxed);
        atomic_store_explicit(state, 2u, memory_order_relaxed);
        return;
    }
    while (atomic_load_explicit(state, memory_order_relaxed) != 2u) {}
    atomic_min_long_pair(lo, hi, val);
}

inline void atomic_max_long_pair_seen(device atomic_uint* lo,
                                      device atomic_uint* hi,
                                      device atomic_uint* state,
                                      long val) {
    uint expected = 0u;
    if (atomic_compare_exchange_weak_explicit(state, &expected, 1u,
                                               memory_order_relaxed,
                                               memory_order_relaxed)) {
        ulong uval = as_type<ulong>(val);
        atomic_store_explicit(lo, (uint)(uval), memory_order_relaxed);
        atomic_store_explicit(hi, (uint)(uval >> 32), memory_order_relaxed);
        atomic_store_explicit(state, 2u, memory_order_relaxed);
        return;
    }
    while (atomic_load_explicit(state, memory_order_relaxed) != 2u) {}
    atomic_max_long_pair(lo, hi, val);
}

// ------------------------------------------------------------------
// Linear-probing hash map helpers (composite-key, single-value).
// Layout: keys1[cap], keys2[cap] (atomic_uint, sentinel = 0xFFFFFFFFu),
//         values[cap] (atomic_uint, interpretation via as_type<>).
// `cap` MUST be a power of two so (slot & (cap-1)) wraps cleanly.
// Sentinel key encoding: (0xFFFFFFFFu, 0xFFFFFFFFu) means "empty".
// Real keys are zero-extended ints; we forbid -1 as a real key to keep
// the sentinel unambiguous.
// ------------------------------------------------------------------

inline uint hashmap_mix2(uint a, uint b) {
    uint h = a * 2654435761u;
    h ^= (b * 2246822519u);
    h ^= (h >> 16);
    h *= 2654435761u;
    h ^= (h >> 16);
    return h;
}

// Insert (key1, key2, value).  If the slot is empty, claim it via CAS
// and write key2 + value.  If already occupied with the same composite
// key, leave value as-is (first-writer-wins).  Returns slot index, or
// 0xFFFFFFFFu if the table is full (should not happen if cap is sized
// > 2x the build cardinality).
inline uint hashmap_insert_kv(device atomic_uint* keys1,
                              device atomic_uint* keys2,
                              device atomic_uint* values,
                              uint cap, uint key1, uint key2, uint value) {
    uint mask = cap - 1u;
    uint slot = hashmap_mix2(key1, key2) & mask;
    for (uint probe = 0u; probe < cap; ++probe) {
        uint expected = 0xFFFFFFFFu;
        if (atomic_compare_exchange_weak_explicit(&keys1[slot], &expected, key1,
                memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&keys2[slot], key2, memory_order_relaxed);
            atomic_store_explicit(&values[slot], value, memory_order_relaxed);
            return slot;
        }
        if (expected == 0xFFFFFFFFu) continue;
        if (expected == key1) {
            // Wait for keys2 to be initialised before comparing.
            uint k2 = atomic_load_explicit(&keys2[slot], memory_order_relaxed);
            while (k2 == 0xFFFFFFFFu) {
                k2 = atomic_load_explicit(&keys2[slot], memory_order_relaxed);
            }
            if (k2 == key2) return slot;
        }
        slot = (slot + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

// Insert + atomic-add aggregation on `value`.  Used for HashGroupJoin.
inline uint hashmap_insert_add(device atomic_uint* keys1,
                                device atomic_uint* keys2,
                                device atomic_uint* values,
                                uint cap, uint key1, uint key2, uint value) {
    uint mask = cap - 1u;
    uint slot = hashmap_mix2(key1, key2) & mask;
    for (uint probe = 0u; probe < cap; ++probe) {
        uint expected = 0xFFFFFFFFu;
        if (atomic_compare_exchange_weak_explicit(&keys1[slot], &expected, key1,
                memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&keys2[slot], key2, memory_order_relaxed);
            atomic_fetch_add_explicit(&values[slot], value, memory_order_relaxed);
            return slot;
        }
        if (expected == 0xFFFFFFFFu) continue;
        if (expected == key1) {
            uint k2 = atomic_load_explicit(&keys2[slot], memory_order_relaxed);
            while (k2 == 0xFFFFFFFFu) {
                k2 = atomic_load_explicit(&keys2[slot], memory_order_relaxed);
            }
            if (k2 == key2) {
                atomic_fetch_add_explicit(&values[slot], value, memory_order_relaxed);
                return slot;
            }
        }
        slot = (slot + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

inline void hashmap_insert_add_float(device atomic_uint* keys1,
                                      device atomic_uint* keys2,
                                      device atomic_uint* values,
                                      uint cap, uint key1, uint key2, float value) {
    uint mask = cap - 1u;
    uint slot = hashmap_mix2(key1, key2) & mask;
    for (uint probe = 0u; probe < cap; ++probe) {
        uint expected = 0xFFFFFFFFu;
        if (atomic_compare_exchange_weak_explicit(&keys1[slot], &expected, key1,
                memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&keys2[slot], key2, memory_order_relaxed);
            atomic_add_float(&values[slot], value);
            return;
        }
        if (expected == 0xFFFFFFFFu) continue;
        if (expected == key1) {
            uint k2 = atomic_load_explicit(&keys2[slot], memory_order_relaxed);
            while (k2 == 0xFFFFFFFFu) {
                k2 = atomic_load_explicit(&keys2[slot], memory_order_relaxed);
            }
            if (k2 == key2) {
                atomic_add_float(&values[slot], value);
                return;
            }
        }
        slot = (slot + 1u) & mask;
    }
}

// Lookup by composite key. Returns slot index, or 0xFFFFFFFFu if absent.
// Read by Probe phase (after a phase break) so plain non-atomic reads
// of keys/values are safe.
inline uint hashmap_lookup(const device uint* keys1,
                           const device uint* keys2,
                           uint cap, uint key1, uint key2) {
    uint mask = cap - 1u;
    uint slot = hashmap_mix2(key1, key2) & mask;
    for (uint probe = 0u; probe < cap; ++probe) {
        uint k1 = keys1[slot];
        if (k1 == 0xFFFFFFFFu) return 0xFFFFFFFFu;
        if (k1 == key1 && keys2[slot] == key2) return slot;
        slot = (slot + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

)METAL";

} // namespace codegen
