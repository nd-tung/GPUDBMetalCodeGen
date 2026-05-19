#include "generic/lowering/generic_scalar_preagg_ops.h"

#include <algorithm>
#include <cctype>
#include <utility>
#include <vector>

namespace codegen {

namespace {

std::string sanitizeIdentifier(std::string name) {
    if (name.empty()) name = "expr";
    for (char& ch : name) {
        unsigned char uch = static_cast<unsigned char>(ch);
        if (!std::isalnum(uch) && ch != '_') ch = '_';
    }
    if (std::isdigit(static_cast<unsigned char>(name.front()))) {
        name = "c_" + name;
    }
    return name;
}

class ScalarGlobalFloatAgg : public MetalUnaryOperator {
public:
    ScalarGlobalFloatAgg(std::unique_ptr<MetalOperator> child,
                         std::string op,
                         std::string buffer,
                         std::string state,
                         std::string value)
        : MetalUnaryOperator(std::move(child)),
          op_(std::move(op)),
          buffer_(std::move(buffer)),
          state_(std::move(state)),
          value_(std::move(value)) {}

    void produce(MetalCodegen& cg, ConsumerFn) override {
        cg.addAtomicBufferParam(buffer_,
                                op_ == "sum" ? "atomic_float" : "atomic_uint",
                                "1");
        if (!state_.empty()) cg.addAtomicBufferParam(state_, "atomic_uint", "1");
        child_->produce(cg, [&]() {
            std::string value = "(float)(" + value_ + ")";
            if (op_ == "sum") {
                cg.addLine("atomic_fetch_add_explicit(&" + buffer_ + "[0], " +
                           value + ", memory_order_relaxed);");
            } else if (op_ == "min") {
                cg.addLine("atomic_min_float_seen(&" + buffer_ + "[0], &" +
                           state_ + "[0], " + value + ");");
            } else if (op_ == "max") {
                cg.addLine("atomic_max_float_seen(&" + buffer_ + "[0], &" +
                           state_ + "[0], " + value + ");");
            }
        });
    }

    std::string describe() const override {
        return "ScalarGlobalFloatAgg(" + op_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(value_, out);
    }

private:
    std::string op_;
    std::string buffer_;
    std::string state_;
    std::string value_;
};

class ScalarDirectFloatAgg : public MetalUnaryOperator {
public:
    ScalarDirectFloatAgg(std::unique_ptr<MetalOperator> child,
                         std::string op,
                         std::string buffer,
                         std::string state,
                         std::string key,
                         std::string value,
                         std::string size)
        : MetalUnaryOperator(std::move(child)),
          op_(std::move(op)),
          buffer_(std::move(buffer)),
          state_(std::move(state)),
          key_(std::move(key)),
          value_(std::move(value)),
          size_(std::move(size)) {}

    void produce(MetalCodegen& cg, ConsumerFn) override {
        cg.addAtomicBufferParam(buffer_, "atomic_uint", size_);
        if (!state_.empty()) cg.addAtomicBufferParam(state_, "atomic_uint", size_);
        child_->produce(cg, [&]() {
            std::string key = "(uint)(" + key_ + ")";
            std::string value = "(float)(" + value_ + ")";
            if (!state_.empty()) {
                cg.addLine("atomic_store_explicit(&" + state_ + "[" + key +
                           "], 1u, memory_order_relaxed);");
            }
            if (op_ == "sum") {
                cg.addLine("atomic_add_float(&" + buffer_ + "[" + key + "], " + value + ");");
            } else if (op_ == "min") {
                cg.addLine("atomic_min_float(&" + buffer_ + "[" + key + "], " + value + ");");
            } else if (op_ == "max") {
                cg.addLine("atomic_max_float(&" + buffer_ + "[" + key + "], " + value + ");");
            }
        });
    }

    std::string describe() const override {
        return "ScalarDirectFloatAgg(" + op_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(key_, out);
        appendIUsFromExpr(value_, out);
    }

private:
    std::string op_;
    std::string buffer_;
    std::string state_;
    std::string key_;
    std::string value_;
    std::string size_;
};

class ScalarDirectAvgAgg : public MetalUnaryOperator {
public:
    ScalarDirectAvgAgg(std::unique_ptr<MetalOperator> child,
                       std::string countBuffer,
                       std::string sumBuffer,
                       std::string key,
                       std::string value,
                       std::string size)
        : MetalUnaryOperator(std::move(child)),
          countBuffer_(std::move(countBuffer)),
          sumBuffer_(std::move(sumBuffer)),
          key_(std::move(key)),
          value_(std::move(value)),
          size_(std::move(size)) {}

    void produce(MetalCodegen& cg, ConsumerFn) override {
        cg.addAtomicBufferParam(countBuffer_, "atomic_uint", size_);
        cg.addAtomicBufferParam(sumBuffer_, "atomic_uint", size_);
        child_->produce(cg, [&]() {
            const std::string key = "(uint)(" + key_ + ")";
            const std::string value = "(float)(" + value_ + ")";
            cg.addLine("atomic_fetch_add_explicit(&" + countBuffer_ +
                       "[" + key + "], 1u, memory_order_relaxed);");
            cg.addLine("atomic_add_float(&" + sumBuffer_ +
                       "[" + key + "], " + value + ");");
        });
    }

    std::string describe() const override {
        return "ScalarDirectAvgAgg";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(key_, out);
        appendIUsFromExpr(value_, out);
    }

private:
    std::string countBuffer_;
    std::string sumBuffer_;
    std::string key_;
    std::string value_;
    std::string size_;
};

class ScalarFillFloatBuffer : public MetalOperator {
public:
    ScalarFillFloatBuffer(std::string buffer, std::string size, std::string fill)
        : buffer_(std::move(buffer)),
          size_(std::move(size)),
          fill_(std::move(fill)) {}

    void produce(MetalCodegen& cg, ConsumerFn) override {
        std::string n = "n_" + sanitizeIdentifier(buffer_);
        cg.addAtomicBufferParam(buffer_, "atomic_uint", size_);
        cg.addResolvedScalarParam(n, "uint", size_);
        cg.addBlock("for (uint i = tid; i < " + n + "; i += tpg)", [&]() {
            cg.addLine("atomic_store_explicit(&" + buffer_ +
                       "[i], as_type<uint>(" + fill_ + "), memory_order_relaxed);");
        });
    }

    std::string describe() const override {
        return "ScalarFillFloatBuffer";
    }

private:
    std::string buffer_;
    std::string size_;
    std::string fill_;
};

class ScalarCompositeHashAgg : public MetalUnaryOperator {
public:
    ScalarCompositeHashAgg(std::unique_ptr<MetalOperator> child,
                           std::string map,
                           std::string key1,
                           std::string key2,
                           std::string value,
                           std::string capacity,
                           bool valueIsFloat)
        : MetalUnaryOperator(std::move(child)),
          map_(std::move(map)),
          key1_(std::move(key1)),
          key2_(std::move(key2)),
          value_(std::move(value)),
          capacity_(std::move(capacity)),
          valueIsFloat_(valueIsFloat) {}

    void produce(MetalCodegen& cg, ConsumerFn consume) override {
        cg.addAtomicBufferParam(map_ + "_states", "atomic_uint", capacity_, 0);
        cg.addAtomicBufferParam(map_ + "_keys1", "atomic_uint", capacity_, 0);
        cg.addAtomicBufferParam(map_ + "_keys2", "atomic_uint", capacity_, 0);
        cg.addAtomicBufferParam(map_ + "_values", "atomic_uint", capacity_, 0);
        cg.addResolvedScalarParam("n_" + map_, "uint", capacity_);
        child_->produce(cg, [&]() {
            if (valueIsFloat_) {
                cg.addLine("scalar_hash_insert_add_float_64(" + map_ + "_states, " +
                           map_ + "_keys1, " + map_ + "_keys2, " +
                           map_ + "_values, n_" + map_ +
                           ", (uint)(" + key1_ + "), (uint)(" + key2_ +
                           "), (float)(" + value_ + "));");
            } else {
                cg.addLine("scalar_hash_insert_add_u32_64(" + map_ + "_states, " +
                           map_ + "_keys1, " + map_ + "_keys2, " +
                           map_ + "_values, n_" + map_ +
                           ", (uint)(" + key1_ + "), (uint)(" + key2_ +
                           "), (uint)(" + value_ + "));");
            }
            consume();
        });
    }

    std::string describe() const override {
        return "ScalarCompositeHashAgg(" + map_ + ")";
    }

    void iusUsed(std::vector<IU>& out) const override {
        appendIUsFromExpr(key1_, out);
        appendIUsFromExpr(key2_, out);
        appendIUsFromExpr(value_, out);
    }

private:
    std::string map_;
    std::string key1_;
    std::string key2_;
    std::string value_;
    std::string capacity_;
    bool valueIsFloat_;
};

const std::string& scalarCompositeHashHelpers() {
    static const std::string helpers = R"(
	static ulong scalar_hash_pack2(uint k1, uint k2) {
	    return ((ulong)k1 << 32) | (ulong)k2;
	}

	static uint scalar_hash_mix64(ulong x) {
	    x ^= x >> 33;
	    x *= 0xff51afd7ed558ccdUL;
	    x ^= x >> 33;
	    x *= 0xc4ceb9fe1a85ec53UL;
	    x ^= x >> 33;
	    return (uint)x ^ (uint)(x >> 32);
	}

	static void scalar_hash_insert_add_u32_64(device atomic_uint* states,
	                                          device atomic_uint* keys1,
	                                          device atomic_uint* keys2,
	                                          device atomic_uint* vals,
	                                          uint cap, uint k1, uint k2, uint value) {
	    ulong key = scalar_hash_pack2(k1, k2);
	    uint mask = cap - 1u;
	    uint slot = scalar_hash_mix64(key) & mask;
	    for (uint probe = 0u; probe < cap; ++probe) {
	        uint state = atomic_load_explicit(&states[slot], memory_order_relaxed);
	        if (state == 0u) {
	            uint expected = 0u;
	            if (atomic_compare_exchange_weak_explicit(&states[slot], &expected, 1u,
	                    memory_order_relaxed, memory_order_relaxed)) {
	                atomic_store_explicit(&keys1[slot], k1, memory_order_relaxed);
	                atomic_store_explicit(&keys2[slot], k2, memory_order_relaxed);
	                atomic_store_explicit(&vals[slot], 0u, memory_order_relaxed);
	                atomic_store_explicit(&states[slot], 2u, memory_order_relaxed);
	                atomic_fetch_add_explicit(&vals[slot], value, memory_order_relaxed);
	                return;
	            }
	            continue;
	        }
	        while (state == 1u) {
	            state = atomic_load_explicit(&states[slot], memory_order_relaxed);
	        }
	        if (state == 2u) {
	            for (uint retry = 0u; retry < 32u; ++retry) {
	                uint slot_k1 = atomic_load_explicit(&keys1[slot], memory_order_relaxed);
	                uint slot_k2 = atomic_load_explicit(&keys2[slot], memory_order_relaxed);
	                if (slot_k1 == k1 && slot_k2 == k2) {
	                    atomic_fetch_add_explicit(&vals[slot], value, memory_order_relaxed);
	                    return;
	                }
	            }
	        }
	        slot = (slot + 1u) & mask;
	    }
	}

	static void scalar_hash_insert_add_float_64(device atomic_uint* states,
	                                            device atomic_uint* keys1,
	                                            device atomic_uint* keys2,
	                                            device atomic_uint* vals,
	                                            uint cap, uint k1, uint k2, float value) {
	    ulong key = scalar_hash_pack2(k1, k2);
	    uint mask = cap - 1u;
	    uint slot = scalar_hash_mix64(key) & mask;
	    for (uint probe = 0u; probe < cap; ++probe) {
	        uint state = atomic_load_explicit(&states[slot], memory_order_relaxed);
	        if (state == 0u) {
	            uint expected = 0u;
	            if (atomic_compare_exchange_weak_explicit(&states[slot], &expected, 1u,
	                    memory_order_relaxed, memory_order_relaxed)) {
	                atomic_store_explicit(&keys1[slot], k1, memory_order_relaxed);
	                atomic_store_explicit(&keys2[slot], k2, memory_order_relaxed);
	                atomic_store_explicit(&vals[slot], 0u, memory_order_relaxed);
	                atomic_store_explicit(&states[slot], 2u, memory_order_relaxed);
	                atomic_add_float(&vals[slot], value);
	                return;
	            }
	            continue;
	        }
	        while (state == 1u) {
	            state = atomic_load_explicit(&states[slot], memory_order_relaxed);
	        }
	        if (state == 2u) {
	            for (uint retry = 0u; retry < 32u; ++retry) {
	                uint slot_k1 = atomic_load_explicit(&keys1[slot], memory_order_relaxed);
	                uint slot_k2 = atomic_load_explicit(&keys2[slot], memory_order_relaxed);
	                if (slot_k1 == k1 && slot_k2 == k2) {
	                    atomic_add_float(&vals[slot], value);
	                    return;
	                }
	            }
	        }
	        slot = (slot + 1u) & mask;
	    }
	}

	static uint scalar_hash_lookup_raw64(const device uint* states,
	                                     const device uint* keys1,
	                                     const device uint* keys2,
	                                     const device uint* vals,
	                                     uint cap, uint k1, uint k2) {
	    ulong key = scalar_hash_pack2(k1, k2);
	    uint mask = cap - 1u;
	    uint slot = scalar_hash_mix64(key) & mask;
	    for (uint probe = 0u; probe < cap; ++probe) {
	        uint state = states[slot];
	        if (state == 0u) return 0u;
	        if (state == 2u && keys1[slot] == k1 && keys2[slot] == k2) return vals[slot];
	        slot = (slot + 1u) & mask;
	    }
	    return 0u;
	}

	static float scalar_hash_lookup_float_or_nan64(const device uint* states,
	                                               const device uint* keys1,
	                                               const device uint* keys2,
	                                               const device uint* vals,
	                                               uint cap, uint k1, uint k2) {
	    ulong key = scalar_hash_pack2(k1, k2);
	    uint mask = cap - 1u;
	    uint slot = scalar_hash_mix64(key) & mask;
	    for (uint probe = 0u; probe < cap; ++probe) {
	        uint state = states[slot];
	        if (state == 0u) return as_type<float>(0x7fc00000u);
	        if (state == 2u && keys1[slot] == k1 && keys2[slot] == k2) return as_type<float>(vals[slot]);
	        slot = (slot + 1u) & mask;
	    }
	    return as_type<float>(0x7fc00000u);
	}
	)";
    return helpers;
}

} // namespace

std::unique_ptr<MetalOperator> makeScalarGlobalFloatAgg(
    std::unique_ptr<MetalOperator> child,
    std::string op,
    std::string buffer,
    std::string state,
    std::string value) {
    return std::make_unique<ScalarGlobalFloatAgg>(
        std::move(child), std::move(op), std::move(buffer),
        std::move(state), std::move(value));
}

std::unique_ptr<MetalOperator> makeScalarDirectFloatAgg(
    std::unique_ptr<MetalOperator> child,
    std::string op,
    std::string buffer,
    std::string state,
    std::string key,
    std::string value,
    std::string size) {
    return std::make_unique<ScalarDirectFloatAgg>(
        std::move(child), std::move(op), std::move(buffer),
        std::move(state), std::move(key), std::move(value),
        std::move(size));
}

std::unique_ptr<MetalOperator> makeScalarDirectAvgAgg(
    std::unique_ptr<MetalOperator> child,
    std::string countBuffer,
    std::string sumBuffer,
    std::string key,
    std::string value,
    std::string size) {
    return std::make_unique<ScalarDirectAvgAgg>(
        std::move(child), std::move(countBuffer), std::move(sumBuffer),
        std::move(key), std::move(value), std::move(size));
}

std::unique_ptr<MetalOperator> makeScalarFillFloatBuffer(
    std::string buffer,
    std::string size,
    std::string fill) {
    return std::make_unique<ScalarFillFloatBuffer>(
        std::move(buffer), std::move(size), std::move(fill));
}

std::unique_ptr<MetalOperator> makeScalarCompositeHashAgg(
    std::unique_ptr<MetalOperator> child,
    std::string map,
    std::string key1,
    std::string key2,
    std::string value,
    std::string capacity,
    bool valueIsFloat) {
    return std::make_unique<ScalarCompositeHashAgg>(
        std::move(child), std::move(map), std::move(key1),
        std::move(key2), std::move(value), std::move(capacity),
        valueIsFloat);
}

void ensureScalarCompositeHashHelpers(MetalQueryPlan& plan) {
    const std::string& helpers = scalarCompositeHashHelpers();
    if (std::find(plan.helpers.begin(), plan.helpers.end(), helpers) ==
        plan.helpers.end()) {
        plan.helpers.push_back(helpers);
    }
}

} // namespace codegen
