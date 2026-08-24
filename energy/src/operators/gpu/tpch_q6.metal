#include <metal_stdlib>

using namespace metal;

struct Q6Pair {
    long count;
    long revenue;
};

struct Q6Parts {
    uint low;
    uint high;
};

inline Q6Parts q6_split(long value) {
    const ulong bits = static_cast<ulong>(value);
    return Q6Parts{static_cast<uint>(bits), static_cast<uint>(bits >> 32)};
}

inline long q6_join(Q6Parts value) {
    return static_cast<long>(
        (static_cast<ulong>(value.high) << 32) | static_cast<ulong>(value.low));
}

inline Q6Parts q6_add(Q6Parts left, Q6Parts right) {
    const uint previous_low = left.low;
    left.low += right.low;
    left.high += right.high + static_cast<uint>(left.low < previous_low);
    return left;
}

inline long q6_simd_sum(long value, uint lane, uint simd_width) {
    Q6Parts parts = q6_split(value);
    for (uint offset = simd_width / 2; offset > 0; offset /= 2) {
        const Q6Parts other{
            simd_shuffle_down(parts.low, static_cast<ushort>(offset)),
            simd_shuffle_down(parts.high, static_cast<ushort>(offset))};
        if (lane < offset) {
            parts = q6_add(parts, other);
        }
    }
    return q6_join(parts);
}

inline bool q6_qualifies(float quantity, float discount, int ship_date) {
    return ship_date >= 19940101 && ship_date < 19950101 &&
           discount >= 0.05f && discount <= 0.07f && quantity < 24.0f;
}

inline long q6_scaled_revenue(float price, float discount) {
    const long price_cents = static_cast<long>(round(price * 100.0f));
    const long discount_hundredths = static_cast<long>(round(discount * 100.0f));
    return price_cents * discount_hundredths;
}

inline void q6_write_pair(
    Q6Pair local,
    device Q6Pair* partials,
    threadgroup Q6Pair* simd_partials,
    uint group_id,
    uint lane,
    uint simd_group,
    uint simd_width,
    uint simd_group_count) {
    const Q6Pair simd_total{
        q6_simd_sum(local.count, lane, simd_width),
        q6_simd_sum(local.revenue, lane, simd_width)};
    if (lane == 0) {
        simd_partials[simd_group] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const Q6Pair value = lane < simd_group_count
            ? simd_partials[lane]
            : Q6Pair{0, 0};
        const Q6Pair total{
            q6_simd_sum(value.count, lane, simd_width),
            q6_simd_sum(value.revenue, lane, simd_width)};
        if (lane == 0) {
            partials[group_id] = total;
        }
    }
}

inline void q6_write_count(
    long local,
    device long* partials,
    threadgroup long* simd_partials,
    uint group_id,
    uint lane,
    uint simd_group,
    uint simd_width,
    uint simd_group_count) {
    const long simd_total = q6_simd_sum(local, lane, simd_width);
    if (lane == 0) {
        simd_partials[simd_group] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const long value = lane < simd_group_count
            ? simd_partials[lane]
            : 0;
        const long total = q6_simd_sum(value, lane, simd_width);
        if (lane == 0) {
            partials[group_id] = total;
        }
    }
}

kernel void tpch_q6_filter_count(
    device const float* quantity [[buffer(0)]],
    device const float* discount [[buffer(1)]],
    device const int* ship_date [[buffer(2)]],
    device long* partials [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    long count = 0;
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < row_count &&
            q6_qualifies(quantity[index], discount[index], ship_date[index])) {
            ++count;
        }
    }
    threadgroup long simd_partials[16];
    q6_write_count(
        count, partials, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void tpch_q6_revenue(
    device const float* quantity [[buffer(0)]],
    device const float* extended_price [[buffer(1)]],
    device const float* discount [[buffer(2)]],
    device const int* ship_date [[buffer(3)]],
    device Q6Pair* partials [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    Q6Pair local{0, 0};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < row_count &&
            q6_qualifies(quantity[index], discount[index], ship_date[index])) {
            ++local.count;
            local.revenue += q6_scaled_revenue(extended_price[index], discount[index]);
        }
    }
    threadgroup Q6Pair simd_partials[16];
    q6_write_pair(
        local, partials, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void tpch_q6_reduce_pair(
    device const Q6Pair* input [[buffer(0)]],
    device Q6Pair* partials [[buffer(1)]],
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
    Q6Pair local{0, 0};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) {
            local.count += input[index].count;
            local.revenue += input[index].revenue;
        }
    }
    threadgroup Q6Pair simd_partials[16];
    q6_write_pair(
        local, partials, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void tpch_q6_reduce_count(
    device const long* input [[buffer(0)]],
    device long* partials [[buffer(1)]],
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
    long local = 0;
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) {
            local += input[index];
        }
    }
    threadgroup long simd_partials[16];
    q6_write_count(
        local, partials, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void tpch_q6_filter_bitmap(
    device const float* quantity [[buffer(0)]],
    device const float* discount [[buffer(1)]],
    device const int* ship_date [[buffer(2)]],
    device uint* bitmap [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    uint word [[thread_position_in_grid]]) {
    const uint row_begin = word * 32;
    if (row_begin >= row_count) {
        return;
    }
    uint bits = 0;
    const uint row_end = min(row_begin + 32, row_count);
    for (uint row = row_begin; row < row_end; ++row) {
        bits |= static_cast<uint>(
            q6_qualifies(quantity[row], discount[row], ship_date[row])) << (row - row_begin);
    }
    bitmap[word] = bits;
}

kernel void tpch_q6_revenue_from_bitmap(
    device const uint* bitmap [[buffer(0)]],
    device const float* extended_price [[buffer(1)]],
    device const float* discount [[buffer(2)]],
    device Q6Pair* partials [[buffer(3)]],
    constant uint& bitmap_word_count [[buffer(4)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint words_per_thread = 4;
    const uint group_begin = group_id * group_width * words_per_thread;
    Q6Pair local{0, 0};
    for (uint iteration = 0; iteration < words_per_thread; ++iteration) {
        const uint word = group_begin + local_id + iteration * group_width;
        if (word >= bitmap_word_count) {
            continue;
        }
        uint bits = bitmap[word];
        local.count += popcount(bits);
        while (bits != 0) {
            const uint bit = ctz(bits);
            const uint row = word * 32 + bit;
            local.revenue += q6_scaled_revenue(
                extended_price[row], discount[row]);
            bits &= bits - 1;
        }
    }
    threadgroup Q6Pair simd_partials[16];
    q6_write_pair(
        local, partials, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}
