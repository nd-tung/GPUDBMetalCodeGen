#include <metal_stdlib>

using namespace metal;

struct FilterProjectRecord {
    uint row_id;
    int part_key;
    long revenue_1e4_usd;
};

static_assert(sizeof(FilterProjectRecord) == 16, "record ABI must match C++");

kernel void materialize_q6_project_records(device const uint* bitmap [[buffer(0)]],
                                           device const uint* offsets [[buffer(1)]],
                                           device const int* part_key [[buffer(2)]],
                                           device const float* extended_price [[buffer(3)]],
                                           device const float* discount [[buffer(4)]],
                                           device FilterProjectRecord* output_records [[buffer(5)]],
                                           constant uint& word_count [[buffer(6)]],
                                           constant uint& row_count [[buffer(7)]],
                                           uint word [[thread_position_in_grid]]) {
    if (word >= word_count) return;

    uint bits = bitmap[word];
    uint output = offsets[word];
    const uint row_begin = word * 32;
    while (bits != 0) {
        const uint bit = ctz(bits);
        const uint row = row_begin + bit;
        if (row < row_count) {
            const long price_cents = static_cast<long>(round(extended_price[row] * 100.0f));
            const long discount_hundredths = static_cast<long>(round(discount[row] * 100.0f));
            output_records[output++] =
                FilterProjectRecord{row, part_key[row],
                                    price_cents * (100 - discount_hundredths)};
        }
        bits &= bits - 1;
    }
}
