#include <metal_stdlib>

using namespace metal;

struct Q1Group {
    long count;
    long sum_quantity_1e2;
    long sum_base_price_1e2;
    long sum_discount_price_1e4_usd;
    long sum_charge_1e6_usd;
    long sum_discount_1e2;
};

struct LongParts {
    uint low;
    uint high;
};

inline LongParts q1_split(long value) {
    const ulong bits = static_cast<ulong>(value);
    return LongParts{static_cast<uint>(bits), static_cast<uint>(bits >> 32)};
}

inline long q1_join(LongParts value) {
    return static_cast<long>(
        (static_cast<ulong>(value.high) << 32) | static_cast<ulong>(value.low));
}

inline LongParts q1_add(LongParts left, LongParts right) {
    const uint previous_low = left.low;
    left.low += right.low;
    left.high += right.high + static_cast<uint>(left.low < previous_low);
    return left;
}

inline long q1_simd_sum(long value, uint lane, uint simd_width) {
    LongParts parts = q1_split(value);
    for (uint offset = simd_width / 2; offset > 0; offset /= 2) {
        const LongParts other{
            simd_shuffle_down(parts.low, static_cast<ushort>(offset)),
            simd_shuffle_down(parts.high, static_cast<ushort>(offset))};
        if (lane < offset) parts = q1_add(parts, other);
    }
    return q1_join(parts);
}

inline Q1Group q1_add_group(Q1Group left, Q1Group right) {
    left.count += right.count;
    left.sum_quantity_1e2 += right.sum_quantity_1e2;
    left.sum_base_price_1e2 += right.sum_base_price_1e2;
    left.sum_discount_price_1e4_usd += right.sum_discount_price_1e4_usd;
    left.sum_charge_1e6_usd += right.sum_charge_1e6_usd;
    left.sum_discount_1e2 += right.sum_discount_1e2;
    return left;
}

inline void q1_write_group(
    Q1Group local,
    device Q1Group* output,
    threadgroup Q1Group* simd_partials,
    uint output_index,
    uint lane,
    uint simd_group,
    uint simd_width,
    uint simd_group_count) {
    Q1Group simd_total{
        q1_simd_sum(local.count, lane, simd_width),
        q1_simd_sum(local.sum_quantity_1e2, lane, simd_width),
        q1_simd_sum(local.sum_base_price_1e2, lane, simd_width),
        q1_simd_sum(local.sum_discount_price_1e4_usd, lane, simd_width),
        q1_simd_sum(local.sum_charge_1e6_usd, lane, simd_width),
        q1_simd_sum(local.sum_discount_1e2, lane, simd_width)};
    if (lane == 0) simd_partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const Q1Group value = lane < simd_group_count
            ? simd_partials[lane]
            : Q1Group{0, 0, 0, 0, 0, 0};
        const Q1Group total{
            q1_simd_sum(value.count, lane, simd_width),
            q1_simd_sum(value.sum_quantity_1e2, lane, simd_width),
            q1_simd_sum(value.sum_base_price_1e2, lane, simd_width),
            q1_simd_sum(value.sum_discount_price_1e4_usd, lane, simd_width),
            q1_simd_sum(value.sum_charge_1e6_usd, lane, simd_width),
            q1_simd_sum(value.sum_discount_1e2, lane, simd_width)};
        if (lane == 0) output[output_index] = total;
    }
    // q1_write_group is invoked repeatedly for the six Q1 keys while reusing
    // the same threadgroup scratch buffer.  Keep every SIMD group here until
    // SIMD group 0 has consumed the current partials; otherwise another SIMD
    // group can overwrite them with the next key before the second reduction
    // has finished reading them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline int q1_group_index(char return_flag, char line_status) {
    const int first = return_flag == 'A' ? 0 : return_flag == 'N' ? 1 :
                      return_flag == 'R' ? 2 : -1;
    const int second = line_status == 'F' ? 0 : line_status == 'O' ? 1 : -1;
    return first < 0 || second < 0 ? -1 : first * 2 + second;
}

kernel void tpch_q1_first(
    device const float* quantity [[buffer(0)]],
    device const float* extended_price [[buffer(1)]],
    device const float* discount [[buffer(2)]],
    device const float* tax [[buffer(3)]],
    device const char* return_flag [[buffer(4)]],
    device const char* line_status [[buffer(5)]],
    device const int* ship_date [[buffer(6)]],
    device Q1Group* output [[buffer(7)]],
    constant uint& row_count [[buffer(8)]],
    constant uint& output_count [[buffer(9)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    Q1Group local[6];
    for (uint group = 0; group < 6; ++group) {
        local[group] = Q1Group{0, 0, 0, 0, 0, 0};
    }
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row >= row_count || ship_date[row] > 19980902) continue;
        const int target = q1_group_index(return_flag[row], line_status[row]);
        if (target < 0) continue;
        const long quantity_scaled = static_cast<long>(round(quantity[row] * 100.0f));
        const long price_scaled = static_cast<long>(round(extended_price[row] * 100.0f));
        const long discount_scaled = static_cast<long>(round(discount[row] * 100.0f));
        const long tax_scaled = static_cast<long>(round(tax[row] * 100.0f));
        ++local[target].count;
        local[target].sum_quantity_1e2 += quantity_scaled;
        local[target].sum_base_price_1e2 += price_scaled;
        local[target].sum_discount_price_1e4_usd +=
            price_scaled * (100 - discount_scaled);
        local[target].sum_charge_1e6_usd +=
            price_scaled * (100 - discount_scaled) * (100 + tax_scaled);
        local[target].sum_discount_1e2 += discount_scaled;
    }
    threadgroup Q1Group simd_partials[16];
    for (uint target = 0; target < 6; ++target) {
        q1_write_group(
            local[target], output, simd_partials, target * output_count + group_id,
            lane, simd_group, simd_width, simd_group_count);
    }
}

kernel void tpch_q1_reduce(
    device const Q1Group* input [[buffer(0)]],
    device Q1Group* output [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    constant uint& output_count [[buffer(3)]],
    uint3 local_position [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_size [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint values_per_thread = 4;
    const uint group_begin = group.x * group_size.x * values_per_thread;
    Q1Group local{0, 0, 0, 0, 0, 0};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_position.x + iteration * group_size.x;
        if (index < input_count) {
            local = q1_add_group(local, input[group.y * input_count + index]);
        }
    }
    threadgroup Q1Group simd_partials[16];
    q1_write_group(
        local, output, simd_partials, group.y * output_count + group.x,
        lane, simd_group, simd_width, simd_group_count);
}

struct Q14Pair {
    long promo;
    long total;
};

inline void q14_write_pair(
    Q14Pair local,
    device Q14Pair* output,
    threadgroup Q14Pair* simd_partials,
    uint output_index,
    uint lane,
    uint simd_group,
    uint simd_width,
    uint simd_group_count) {
    const Q14Pair simd_total{
        q1_simd_sum(local.promo, lane, simd_width),
        q1_simd_sum(local.total, lane, simd_width)};
    if (lane == 0) simd_partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const Q14Pair value = lane < simd_group_count
            ? simd_partials[lane] : Q14Pair{0, 0};
        const Q14Pair total{
            q1_simd_sum(value.promo, lane, simd_width),
            q1_simd_sum(value.total, lane, simd_width)};
        if (lane == 0) output[output_index] = total;
    }
}

inline bool q14_hash_lookup(
    int key,
    device const int* hash_keys,
    device const uchar* hash_promo,
    uint hash_mask,
    thread bool& promo) {
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

kernel void tpch_q14_first(
    device const int* part_key [[buffer(0)]],
    device const float* extended_price [[buffer(1)]],
    device const float* discount [[buffer(2)]],
    device const int* ship_date [[buffer(3)]],
    device const int* hash_keys [[buffer(4)]],
    device const uchar* hash_promo [[buffer(5)]],
    device Q14Pair* output [[buffer(6)]],
    constant uint& row_count [[buffer(7)]],
    constant uint& hash_mask [[buffer(8)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    Q14Pair local{0, 0};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row >= row_count || ship_date[row] < 19950901 || ship_date[row] >= 19951001) {
            continue;
        }
        bool promo = false;
        if (!q14_hash_lookup(part_key[row], hash_keys, hash_promo, hash_mask, promo)) continue;
        const long price = static_cast<long>(round(extended_price[row] * 100.0f));
        const long discount_scaled = static_cast<long>(round(discount[row] * 100.0f));
        const long revenue = price * (100 - discount_scaled);
        local.total += revenue;
        if (promo) local.promo += revenue;
    }
    threadgroup Q14Pair simd_partials[16];
    q14_write_pair(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void tpch_q14_reduce(
    device const Q14Pair* input [[buffer(0)]],
    device Q14Pair* output [[buffer(1)]],
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
    Q14Pair local{0, 0};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) {
            local.promo += input[index].promo;
            local.total += input[index].total;
        }
    }
    threadgroup Q14Pair simd_partials[16];
    q14_write_pair(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

struct TopKEntry {
    long total_price_cents;
    int order_key;
};

inline TopKEntry topk_sentinel() {
    return TopKEntry{static_cast<long>(0x8000000000000000ul), 0x7fffffff};
}

inline bool topk_better(TopKEntry left, TopKEntry right) {
    return left.total_price_cents > right.total_price_cents ||
           (left.total_price_cents == right.total_price_cents &&
            left.order_key < right.order_key);
}

inline void topk_initialize(thread TopKEntry* result) {
    for (uint index = 0; index < 10; ++index) result[index] = topk_sentinel();
}

inline void topk_insert(thread TopKEntry* result, TopKEntry candidate) {
    if (!topk_better(candidate, result[9])) return;
    uint position = 9;
    while (position > 0 && topk_better(candidate, result[position - 1])) {
        result[position] = result[position - 1];
        --position;
    }
    result[position] = candidate;
}

kernel void orders_topk_first(
    device const int* order_key [[buffer(0)]],
    device const float* total_price [[buffer(1)]],
    device TopKEntry* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {
    constexpr uint values_per_thread = 64;
    const uint output_index = group_id * group_width + local_id;
    const uint group_begin = group_id * group_width * values_per_thread;
    TopKEntry local[10];
    topk_initialize(local);
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row < row_count) {
            topk_insert(local, TopKEntry{
                static_cast<long>(round(total_price[row] * 100.0f)), order_key[row]});
        }
    }
    for (uint index = 0; index < 10; ++index) {
        output[output_index * 10 + index] = local[index];
    }
}

kernel void orders_topk_reduce(
    device const TopKEntry* input [[buffer(0)]],
    device TopKEntry* output [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    constant uint& output_count [[buffer(3)]],
    uint output_index [[thread_position_in_grid]]) {
    if (output_index >= output_count) return;
    TopKEntry local[10];
    topk_initialize(local);
    const uint first = output_index * 4;
    for (uint list = first; list < min(first + 4, input_count); ++list) {
        for (uint index = 0; index < 10; ++index) {
            topk_insert(local, input[list * 10 + index]);
        }
    }
    for (uint index = 0; index < 10; ++index) {
        output[output_index * 10 + index] = local[index];
    }
}

struct PriceSum {
    long sum;
};

struct PriceMinMax {
    int minimum;
    int maximum;
};

struct PriceStats {
    long sum;
    int minimum;
    int maximum;
};

inline int price_simd_min(int value, uint lane, uint simd_width) {
    for (uint offset = simd_width / 2; offset > 0; offset /= 2) {
        const int other =
            simd_shuffle_down(value, static_cast<ushort>(offset));
        if (lane < offset) value = min(value, other);
    }
    return value;
}

inline int price_simd_max(int value, uint lane, uint simd_width) {
    for (uint offset = simd_width / 2; offset > 0; offset /= 2) {
        const int other =
            simd_shuffle_down(value, static_cast<ushort>(offset));
        if (lane < offset) value = max(value, other);
    }
    return value;
}

inline PriceMinMax price_add_minmax(
    PriceMinMax left,
    PriceMinMax right) {
    left.minimum = min(left.minimum, right.minimum);
    left.maximum = max(left.maximum, right.maximum);
    return left;
}

inline PriceStats price_add_stats(PriceStats left, PriceStats right) {
    left.sum += right.sum;
    left.minimum = min(left.minimum, right.minimum);
    left.maximum = max(left.maximum, right.maximum);
    return left;
}

inline void price_write_sum(
    PriceSum local,
    device PriceSum* output,
    threadgroup PriceSum* simd_partials,
    uint output_index,
    uint lane,
    uint simd_group,
    uint simd_width,
    uint simd_group_count) {
    const PriceSum simd_total{
        q1_simd_sum(local.sum, lane, simd_width)};
    if (lane == 0) simd_partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const PriceSum value = lane < simd_group_count
            ? simd_partials[lane]
            : PriceSum{0};
        const PriceSum total{
            q1_simd_sum(value.sum, lane, simd_width)};
        if (lane == 0) output[output_index] = total;
    }
}

inline void price_write_minmax(
    PriceMinMax local,
    device PriceMinMax* output,
    threadgroup PriceMinMax* simd_partials,
    uint output_index,
    uint lane,
    uint simd_group,
    uint simd_width,
    uint simd_group_count) {
    const PriceMinMax simd_total{
        price_simd_min(local.minimum, lane, simd_width),
        price_simd_max(local.maximum, lane, simd_width)};
    if (lane == 0) simd_partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const PriceMinMax value = lane < simd_group_count
            ? simd_partials[lane]
            : PriceMinMax{0x7fffffff, static_cast<int>(0x80000000u)};
        const PriceMinMax total{
            price_simd_min(value.minimum, lane, simd_width),
            price_simd_max(value.maximum, lane, simd_width)};
        if (lane == 0) output[output_index] = total;
    }
}

inline void price_write_stats(
    PriceStats local,
    device PriceStats* output,
    threadgroup PriceStats* simd_partials,
    uint output_index,
    uint lane,
    uint simd_group,
    uint simd_width,
    uint simd_group_count) {
    const PriceStats simd_total{
        q1_simd_sum(local.sum, lane, simd_width),
        price_simd_min(local.minimum, lane, simd_width),
        price_simd_max(local.maximum, lane, simd_width)};
    if (lane == 0) simd_partials[simd_group] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        const PriceStats value = lane < simd_group_count
            ? simd_partials[lane]
            : PriceStats{
                  0, 0x7fffffff, static_cast<int>(0x80000000u)};
        const PriceStats total{
            q1_simd_sum(value.sum, lane, simd_width),
            price_simd_min(value.minimum, lane, simd_width),
            price_simd_max(value.maximum, lane, simd_width)};
        if (lane == 0) output[output_index] = total;
    }
}

kernel void price_sum_first(
    device const float* input [[buffer(0)]],
    device PriceSum* output [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    PriceSum local{0};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row >= row_count) continue;
        local.sum += static_cast<int>(round(input[row] * 100.0f));
    }
    threadgroup PriceSum simd_partials[16];
    price_write_sum(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void price_sum_reduce(
    device const PriceSum* input [[buffer(0)]],
    device PriceSum* output [[buffer(1)]],
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
    PriceSum local{0};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) local.sum += input[index].sum;
    }
    threadgroup PriceSum simd_partials[16];
    price_write_sum(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void price_minmax_first(
    device const float* input [[buffer(0)]],
    device PriceMinMax* output [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    PriceMinMax local{0x7fffffff, static_cast<int>(0x80000000u)};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row >= row_count) continue;
        const int value = static_cast<int>(round(input[row] * 100.0f));
        local.minimum = min(local.minimum, value);
        local.maximum = max(local.maximum, value);
    }
    threadgroup PriceMinMax simd_partials[16];
    price_write_minmax(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void price_minmax_reduce(
    device const PriceMinMax* input [[buffer(0)]],
    device PriceMinMax* output [[buffer(1)]],
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
    PriceMinMax local{0x7fffffff, static_cast<int>(0x80000000u)};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) {
            local = price_add_minmax(local, input[index]);
        }
    }
    threadgroup PriceMinMax simd_partials[16];
    price_write_minmax(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void price_stats_first(
    device const float* input [[buffer(0)]],
    device PriceStats* output [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]],
    uint simd_group_count [[simdgroups_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    PriceStats local{0, 0x7fffffff, static_cast<int>(0x80000000u)};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row >= row_count) continue;
        const int value = static_cast<int>(round(input[row] * 100.0f));
        local.sum += value;
        local.minimum = min(local.minimum, value);
        local.maximum = max(local.maximum, value);
    }
    threadgroup PriceStats simd_partials[16];
    price_write_stats(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void price_stats_reduce(
    device const PriceStats* input [[buffer(0)]],
    device PriceStats* output [[buffer(1)]],
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
    PriceStats local{0, 0x7fffffff, static_cast<int>(0x80000000u)};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) {
            local = price_add_stats(local, input[index]);
        }
    }
    threadgroup PriceStats simd_partials[16];
    price_write_stats(
        local, output, simd_partials, group_id, lane, simd_group,
        simd_width, simd_group_count);
}

kernel void groupby_count_clear(
    device uint* counts [[buffer(0)]],
    constant uint& group_count [[buffer(1)]],
    uint group [[thread_position_in_grid]]) {
    if (group < group_count) counts[group] = 0;
}

kernel void groupby_count_i32(
    device const int* keys [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& group_count [[buffer(3)]],
    uint work_item [[thread_position_in_grid]],
    uint work_item_count [[threads_per_grid]]) {
    const uint mask = group_count - 1;
    for (uint iteration = 0; iteration < 16; ++iteration) {
        const uint row = work_item + iteration * work_item_count;
        if (row < row_count) {
            const uint group = (static_cast<uint>(keys[row]) - 1u) & mask;
            atomic_fetch_add_explicit(
                counts + group, 1u, memory_order_relaxed);
        }
    }
}

kernel void price_sum_first_threadgroup(
    device const float* input [[buffer(0)]],
    device PriceSum* output [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    PriceSum local{0};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row >= row_count) continue;
        local.sum += static_cast<int>(round(input[row] * 100.0f));
    }
    threadgroup PriceSum values[512];
    values[local_id] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_width / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            values[local_id].sum += values[local_id + stride].sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_id == 0) output[group_id] = values[0];
}

kernel void price_sum_reduce_threadgroup(
    device const PriceSum* input [[buffer(0)]],
    device PriceSum* output [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {
    constexpr uint values_per_thread = 4;
    const uint group_begin = group_id * group_width * values_per_thread;
    PriceSum local{0};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) local.sum += input[index].sum;
    }
    threadgroup PriceSum values[512];
    values[local_id] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_width / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            values[local_id].sum += values[local_id + stride].sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_id == 0) output[group_id] = values[0];
}

kernel void price_minmax_first_threadgroup(
    device const float* input [[buffer(0)]],
    device PriceMinMax* output [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    PriceMinMax local{0x7fffffff, static_cast<int>(0x80000000u)};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row >= row_count) continue;
        const int value = static_cast<int>(round(input[row] * 100.0f));
        local.minimum = min(local.minimum, value);
        local.maximum = max(local.maximum, value);
    }
    threadgroup PriceMinMax values[512];
    values[local_id] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_width / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            values[local_id] = price_add_minmax(
                values[local_id], values[local_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_id == 0) output[group_id] = values[0];
}

kernel void price_minmax_reduce_threadgroup(
    device const PriceMinMax* input [[buffer(0)]],
    device PriceMinMax* output [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {
    constexpr uint values_per_thread = 4;
    const uint group_begin = group_id * group_width * values_per_thread;
    PriceMinMax local{0x7fffffff, static_cast<int>(0x80000000u)};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) {
            local = price_add_minmax(local, input[index]);
        }
    }
    threadgroup PriceMinMax values[512];
    values[local_id] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_width / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            values[local_id] = price_add_minmax(
                values[local_id], values[local_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_id == 0) output[group_id] = values[0];
}

kernel void price_stats_first_threadgroup(
    device const float* input [[buffer(0)]],
    device PriceStats* output [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {
    constexpr uint values_per_thread = 16;
    const uint group_begin = group_id * group_width * values_per_thread;
    PriceStats local{0, 0x7fffffff, static_cast<int>(0x80000000u)};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint row = group_begin + local_id + iteration * group_width;
        if (row >= row_count) continue;
        const int value = static_cast<int>(round(input[row] * 100.0f));
        local.sum += value;
        local.minimum = min(local.minimum, value);
        local.maximum = max(local.maximum, value);
    }
    threadgroup PriceStats values[512];
    values[local_id] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_width / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            values[local_id] =
                price_add_stats(values[local_id], values[local_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_id == 0) output[group_id] = values[0];
}

kernel void price_stats_reduce_threadgroup(
    device const PriceStats* input [[buffer(0)]],
    device PriceStats* output [[buffer(1)]],
    constant uint& input_count [[buffer(2)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint group_width [[threads_per_threadgroup]]) {
    constexpr uint values_per_thread = 4;
    const uint group_begin = group_id * group_width * values_per_thread;
    PriceStats local{0, 0x7fffffff, static_cast<int>(0x80000000u)};
    for (uint iteration = 0; iteration < values_per_thread; ++iteration) {
        const uint index = group_begin + local_id + iteration * group_width;
        if (index < input_count) {
            local = price_add_stats(local, input[index]);
        }
    }
    threadgroup PriceStats values[512];
    values[local_id] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_width / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            values[local_id] =
                price_add_stats(values[local_id], values[local_id + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local_id == 0) output[group_id] = values[0];
}

kernel void groupby_count_i32_threadgroup(
    device const int* keys [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& group_count [[buffer(3)]],
    threadgroup atomic_uint* local_counts [[threadgroup(0)]],
    uint work_item [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_width [[threads_per_threadgroup]],
    uint work_item_count [[threads_per_grid]]) {
    for (uint group = local_id; group < group_count; group += group_width) {
        atomic_store_explicit(
            local_counts + group, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint mask = group_count - 1;
    for (uint iteration = 0; iteration < 16; ++iteration) {
        const uint row = work_item + iteration * work_item_count;
        if (row < row_count) {
            const uint group = (static_cast<uint>(keys[row]) - 1u) & mask;
            atomic_fetch_add_explicit(
                local_counts + group, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint group = local_id; group < group_count; group += group_width) {
        const uint count = atomic_load_explicit(
            local_counts + group, memory_order_relaxed);
        if (count != 0) {
            atomic_fetch_add_explicit(
                counts + group, count, memory_order_relaxed);
        }
    }
}
