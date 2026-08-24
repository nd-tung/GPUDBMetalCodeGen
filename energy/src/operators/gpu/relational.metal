#include <metal_stdlib>

using namespace metal;

kernel void bitmap_popcounts_u32(
    device const uint* bitmap [[buffer(0)]],
    device uint* counts [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        counts[index] = popcount(bitmap[index]);
    }
}

kernel void exclusive_scan_u32_blocks(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device uint* block_sums [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint global_id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {
    threadgroup uint values[512];
    values[local_id] = global_id < count ? input[global_id] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 1; stride < group_width; stride *= 2) {
        const uint index = (local_id + 1) * stride * 2 - 1;
        if (index < group_width) {
            values[index] += values[index - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_id == 0) {
        block_sums[group_id] = values[group_width - 1];
        values[group_width - 1] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_width / 2; stride > 0; stride /= 2) {
        const uint index = (local_id + 1) * stride * 2 - 1;
        if (index < group_width) {
            const uint left = values[index - stride];
            values[index - stride] = values[index];
            values[index] += left;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (global_id < count) {
        output[global_id] = values[local_id];
    }
}

kernel void add_scan_block_offsets_u32(
    device uint* offsets [[buffer(0)]],
    device const uint* block_offsets [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& block_width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        offsets[index] += block_offsets[index / block_width];
    }
}

kernel void materialize_bitmap_rows(
    device const uint* bitmap [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device uint* output_rows [[buffer(2)]],
    constant uint& word_count [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    uint word [[thread_position_in_grid]]) {
    if (word >= word_count) return;
    const uint bits = bitmap[word];
    uint output = offsets[word];
    const uint row_begin = word * 32;
    for (uint bit = 0; bit < 32 && row_begin + bit < row_count; ++bit) {
        if ((bits & (1u << bit)) != 0) {
            output_rows[output++] = row_begin + bit;
        }
    }
}

kernel void bitmap_materialized_count(
    device const uint* bitmap [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device uint* output_count [[buffer(2)]],
    constant uint& word_count [[buffer(3)]]) {
    if (word_count == 0) {
        output_count[0] = 0;
    } else {
        output_count[0] = offsets[word_count - 1] + popcount(bitmap[word_count - 1]);
    }
}
