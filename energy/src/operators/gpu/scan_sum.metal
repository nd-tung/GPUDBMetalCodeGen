#include <metal_stdlib>

using namespace metal;

struct Sum64Parts {
    uint low;
    uint high;
};

inline Sum64Parts split_sum(long value) {
    const ulong bits = static_cast<ulong>(value);
    return Sum64Parts{static_cast<uint>(bits), static_cast<uint>(bits >> 32)};
}

inline long join_sum(Sum64Parts value) {
    const ulong bits = (static_cast<ulong>(value.high) << 32) |
                       static_cast<ulong>(value.low);
    return static_cast<long>(bits);
}

inline Sum64Parts add_sum(Sum64Parts left, Sum64Parts right) {
    const uint previous_low = left.low;
    left.low += right.low;
    left.high += right.high + static_cast<uint>(left.low < previous_low);
    return left;
}

// Metal SIMD reductions don't accept 64-bit integers on this target. Reduce an
// exact signed 64-bit sum as two 32-bit words using SIMD shuffle instructions.
inline long simd_sum_i64(long value, uint lane, uint simd_width) {
    Sum64Parts parts = split_sum(value);
    for (uint offset = simd_width / 2; offset > 0; offset /= 2) {
        const Sum64Parts other{
            simd_shuffle_down(parts.low, static_cast<ushort>(offset)),
            simd_shuffle_down(parts.high, static_cast<ushort>(offset))};
        if (lane < offset) {
            parts = add_sum(parts, other);
        }
    }
    return join_sum(parts);
}

inline void write_threadgroup_sum(
    long local_sum,
    device long* partial_sums,
    threadgroup long* simd_sums,
    uint group_id,
    uint lane,
    uint simd_group,
    uint simd_width,
    uint simd_group_count) {

    const long simd_total = simd_sum_i64(local_sum, lane, simd_width);
    if (lane == 0) {
        simd_sums[simd_group] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        const long group_value = lane < simd_group_count ? simd_sums[lane] : 0;
        const long group_total = simd_sum_i64(group_value, lane, simd_width);
        if (lane == 0) {
            partial_sums[group_id] = group_total;
        }
    }
}

inline void write_threadgroup_sum_tree(
    long local_sum,
    device long* partial_sums,
    threadgroup long* local_sums,
    uint group_id,
    uint local_id,
    uint group_width) {

    local_sums[local_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_width / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            local_sums[local_id] += local_sums[local_id + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_id == 0) {
        partial_sums[group_id] = local_sums[0];
    }
}

// Correctness baseline: one value per thread and a full threadgroup-memory tree.
kernel void scan_sum_i32_baseline(
    device const int* input [[buffer(0)]],
    device long* partial_sums [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {

    threadgroup long local_sums[512];
    const long local_sum = global_id < input_count
        ? static_cast<long>(input[global_id])
        : 0;
    write_threadgroup_sum_tree(
        local_sum, partial_sums, local_sums, group_id, local_id, group_width);
}

kernel void reduce_sum_i64_baseline(
    device const long* input [[buffer(0)]],
    device long* partial_sums [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {

    threadgroup long local_sums[512];
    const long local_sum = global_id < input_count ? input[global_id] : 0;
    write_threadgroup_sum_tree(
        local_sum, partial_sums, local_sums, group_id, local_id, group_width);
}

// Architecture-controlled middle variant: multi-item local accumulation with
// the baseline's full threadgroup-memory reduction tree.
kernel void scan_sum_i32_multi_item(
    device const int* input [[buffer(0)]],
    device long* partial_sums [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {

    constexpr uint vectors_per_thread = 4;
    const uint vector_count = input_count / 4;
    const uint group_vector_begin = group_id * group_width * vectors_per_thread;
    device const int4* vector_input = reinterpret_cast<device const int4*>(input);

    long local_sum = 0;
    for (uint iteration = 0; iteration < vectors_per_thread; ++iteration) {
        const uint vector_index = group_vector_begin + local_id + iteration * group_width;
        if (vector_index < vector_count) {
            const int4 values = vector_input[vector_index];
            local_sum += static_cast<long>(values.x);
            local_sum += static_cast<long>(values.y);
            local_sum += static_cast<long>(values.z);
            local_sum += static_cast<long>(values.w);
        }
    }

    const uint scalar_tail_begin = vector_count * 4;
    const uint group_scalar_end = min(
        input_count, (group_vector_begin + group_width * vectors_per_thread) * 4);
    if (group_scalar_end == input_count && local_id < input_count - scalar_tail_begin) {
        local_sum += static_cast<long>(input[scalar_tail_begin + local_id]);
    }

    threadgroup long local_sums[512];
    write_threadgroup_sum_tree(
        local_sum, partial_sums, local_sums, group_id, local_id, group_width);
}

kernel void reduce_sum_i64_multi_item(
    device const long* input [[buffer(0)]],
    device long* partial_sums [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {

    constexpr uint values_per_thread = 4;
    const uint group_begin = group_id * group_width * values_per_thread;
    long local_sum = 0;
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) {
            local_sum += input[index];
        }
    }

    threadgroup long local_sums[512];
    write_threadgroup_sum_tree(
        local_sum, partial_sums, local_sums, group_id, local_id, group_width);
}

// Tuned first pass: four coalesced int4 loads per thread (16 values),
// followed by an exact 64-bit two-level SIMD-group/threadgroup reduction.
kernel void scan_sum_i32_simdgroup(
    device const int* input [[buffer(0)]],
    device long* partial_sums [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {

    constexpr uint vectors_per_thread = 4;
    const uint vector_count = input_count / 4;
    const uint group_vector_begin = group_id * group_width * vectors_per_thread;
    device const int4* vector_input = reinterpret_cast<device const int4*>(input);

    long local_sum = 0;
    for (uint iteration = 0; iteration < vectors_per_thread; ++iteration) {
        const uint vector_index = group_vector_begin + local_id + iteration * group_width;
        if (vector_index < vector_count) {
            const int4 values = vector_input[vector_index];
            local_sum += static_cast<long>(values.x);
            local_sum += static_cast<long>(values.y);
            local_sum += static_cast<long>(values.z);
            local_sum += static_cast<long>(values.w);
        }
    }

    // At most three scalar tail values remain. Only the final threadgroup can own them.
    const uint scalar_tail_begin = vector_count * 4;
    const uint group_scalar_end = min(
        input_count, (group_vector_begin + group_width * vectors_per_thread) * 4);
    if (group_scalar_end == input_count && local_id < input_count - scalar_tail_begin) {
        local_sum += static_cast<long>(input[scalar_tail_begin + local_id]);
    }

    threadgroup long simd_sums[16];
    write_threadgroup_sum(
        local_sum, partial_sums, simd_sums, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

// Optimized later passes: four 64-bit partials per thread and the same
// one-barrier SIMD/threadgroup reduction.
kernel void reduce_sum_i64_simdgroup(
    device const long* input [[buffer(0)]],
    device long* partial_sums [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {

    constexpr uint values_per_thread = 4;
    const uint group_begin = group_id * group_width * values_per_thread;
    long local_sum = 0;
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) {
            local_sum += input[index];
        }
    }

    threadgroup long simd_sums[16];
    write_threadgroup_sum(
        local_sum, partial_sums, simd_sums, group_id, lane, simd_group,
        simd_width, simd_group_count);
}
