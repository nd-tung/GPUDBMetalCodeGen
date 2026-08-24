#include <metal_stdlib>

using namespace metal;

constant uint hash_values_per_probe_thread = 16;
constant uint hash_values_per_materialize_thread = 8;

inline bool hash_part_lookup(
    int key,
    device const int* hash_keys,
    device const uint* hash_promo,
    uint hash_mask,
    thread bool& promo) {
    if (key <= 0) return false;
    uint slot = static_cast<uint>(key) * 2654435761u & hash_mask;
    while (hash_keys[slot] != 0) {
        if (hash_keys[slot] == key) {
            promo = hash_promo[slot] != 0;
            return true;
        }
        slot = (slot + 1) & hash_mask;
    }
    return false;
}

inline bool hash_is_promo(device const char* type) {
    return type[0] == 'P' && type[1] == 'R' && type[2] == 'O' &&
           type[3] == 'M' && type[4] == 'O';
}

kernel void part_hash_clear(
    device atomic_int* hash_keys [[buffer(0)]],
    device uint* hash_promo [[buffer(1)]],
    constant uint& capacity [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < capacity) {
        atomic_store_explicit(&hash_keys[index], 0, memory_order_relaxed);
        hash_promo[index] = 0;
    }
}

inline void hash_write_count_pair(
    uint2 local,
    device uint2* output,
    threadgroup uint2* simd_partials,
    uint output_index,
    uint lane,
    uint simd_group,
    uint simd_group_count) {
    const uint2 simd_total{simd_sum(local.x), simd_sum(local.y)};
    if (lane == 0) simd_partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const uint2 value =
            lane < simd_group_count ? simd_partials[lane] : uint2(0);
        const uint2 total{simd_sum(value.x), simd_sum(value.y)};
        if (lane == 0) output[output_index] = total;
    }
}

kernel void part_hash_build_atomic(
    device const int* part_key [[buffer(0)]],
    device const char* part_type [[buffer(1)]],
    device atomic_int* hash_keys [[buffer(2)]],
    device uint* hash_promo [[buffer(3)]],
    device uint2* group_counts [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& hash_mask [[buffer(6)]],
    uint row [[thread_position_in_grid]],
    uint group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    uint2 local(0);
    if (row < row_count) {
        const int key = part_key[row];
        const bool promo = hash_is_promo(part_type + row * 25);
        uint slot = static_cast<uint>(key) * 2654435761u & hash_mask;
        while (true) {
            int observed =
                atomic_load_explicit(&hash_keys[slot], memory_order_relaxed);
            if (observed == key) {
                // Unique build keys are the public contract. This assignment
                // also keeps duplicate-but-equal input deterministic.
                hash_promo[slot] = static_cast<uint>(promo);
                break;
            }
            if (observed == 0) {
                int expected = 0;
                if (atomic_compare_exchange_weak_explicit(
                        &hash_keys[slot], &expected, key,
                        memory_order_relaxed, memory_order_relaxed)) {
                    hash_promo[slot] = static_cast<uint>(promo);
                    local = uint2(1, static_cast<uint>(promo));
                    break;
                }
                if (expected == key) {
                    hash_promo[slot] = static_cast<uint>(promo);
                    break;
                }
                // A weak compare-exchange may fail spuriously while the slot
                // is still empty. Retry this slot; skipping it would break the
                // open-addressing lookup invariant.
                if (expected == 0) continue;
            }
            slot = (slot + 1) & hash_mask;
        }
    }
    threadgroup uint2 simd_partials[16];
    hash_write_count_pair(
        local, group_counts, simd_partials, group_id, lane, simd_group,
        simd_group_count);
}

kernel void part_hash_probe_count_first(
    device const int* probe_keys [[buffer(0)]],
    device const int* hash_keys [[buffer(1)]],
    device const uint* hash_promo [[buffer(2)]],
    device uint2* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& hash_mask [[buffer(5)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    const uint group_begin =
        group_id * group_width * hash_values_per_probe_thread;
    uint2 local(0);
    for (uint iteration = 0; iteration < hash_values_per_probe_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row >= row_count) continue;
        bool promo = false;
        if (hash_part_lookup(
                probe_keys[row], hash_keys, hash_promo, hash_mask, promo)) {
            ++local.x;
            local.y += static_cast<uint>(promo);
        }
    }
    threadgroup uint2 simd_partials[16];
    hash_write_count_pair(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_group_count);
}

kernel void part_hash_probe_count_reduce(
    device const uint2* input [[buffer(0)]],
    device uint2* output [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint values_per_thread = 4;
    const uint group_begin = group_id * group_width * values_per_thread;
    uint2 local(0);
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) local += input[index];
    }
    threadgroup uint2 simd_partials[16];
    hash_write_count_pair(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_group_count);
}

kernel void part_hash_probe_block_counts(
    device const int* probe_keys [[buffer(0)]],
    device const int* hash_keys [[buffer(1)]],
    device const uint* hash_promo [[buffer(2)]],
    device uint* block_counts [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& hash_mask [[buffer(5)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    const uint row_begin =
        group_id * group_width * hash_values_per_materialize_thread +
        local_id * hash_values_per_materialize_thread;
    uint local_count = 0;
    for (uint iteration = 0;
         iteration < hash_values_per_materialize_thread;
         ++iteration) {
        const uint row = row_begin + iteration;
        if (row >= row_count) continue;
        bool promo = false;
        local_count += hash_part_lookup(
            probe_keys[row], hash_keys, hash_promo, hash_mask, promo);
    }
    const uint simd_total = simd_sum(local_count);
    threadgroup uint simd_partials[16];
    if (lane == 0) simd_partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const uint value =
            lane < simd_group_count ? simd_partials[lane] : 0;
        const uint total = simd_sum(value);
        if (lane == 0) block_counts[group_id] = total;
    }
}

kernel void part_hash_exclusive_scan_u32(
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
        if (index < group_width) values[index] += values[index - stride];
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
    if (global_id < count) output[global_id] = values[local_id];
}

kernel void part_hash_add_scan_offsets(
    device uint* offsets [[buffer(0)]],
    device const uint* block_offsets [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& block_width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) offsets[index] += block_offsets[index / block_width];
}

struct HashMatchRecord {
    uint probe_row;
    uint promo;
};

kernel void part_hash_probe_scatter(
    device const int* probe_keys [[buffer(0)]],
    device const int* hash_keys [[buffer(1)]],
    device const uint* hash_promo [[buffer(2)]],
    device const uint* block_offsets [[buffer(3)]],
    device HashMatchRecord* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& hash_mask [[buffer(6)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {
    const uint row_begin =
        group_id * group_width * hash_values_per_materialize_thread +
        local_id * hash_values_per_materialize_thread;
    bool matched[hash_values_per_materialize_thread];
    bool promos[hash_values_per_materialize_thread];
    uint local_count = 0;
    for (uint iteration = 0;
         iteration < hash_values_per_materialize_thread;
         ++iteration) {
        const uint row = row_begin + iteration;
        bool promo = false;
        matched[iteration] =
            row < row_count &&
            hash_part_lookup(
                probe_keys[row], hash_keys, hash_promo, hash_mask, promo);
        promos[iteration] = promo;
        local_count += static_cast<uint>(matched[iteration]);
    }

    // Exclusive prefix over the per-thread counts. Each thread owns a
    // contiguous run of eight rows, so this preserves complete probe order.
    threadgroup uint prefixes[512];
    prefixes[local_id] = local_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < group_width; stride *= 2) {
        const uint index = (local_id + 1) * stride * 2 - 1;
        if (index < group_width) prefixes[index] += prefixes[index - stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_id == 0) prefixes[group_width - 1] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_width / 2; stride > 0; stride /= 2) {
        const uint index = (local_id + 1) * stride * 2 - 1;
        if (index < group_width) {
            const uint left = prefixes[index - stride];
            prefixes[index - stride] = prefixes[index];
            prefixes[index] += left;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint write = block_offsets[group_id] + prefixes[local_id];
    for (uint iteration = 0;
         iteration < hash_values_per_materialize_thread;
         ++iteration) {
        if (matched[iteration]) {
            output[write++] = HashMatchRecord{
                row_begin + iteration,
                static_cast<uint>(promos[iteration])};
        }
    }
}

kernel void part_hash_materialized_count(
    device const uint* block_counts [[buffer(0)]],
    device const uint* block_offsets [[buffer(1)]],
    device uint* output_count [[buffer(2)]],
    constant uint& block_count [[buffer(3)]]) {
    output_count[0] = block_count == 0
        ? 0
        : block_offsets[block_count - 1] + block_counts[block_count - 1];
}
