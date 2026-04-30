#include "metal_operators.h"
#include <sstream>
#include <cstdlib>

namespace codegen {

// Ablation: when GPUDB_SCALAR_ATOMIC=1, MetalTGReduce skips the
// SIMD+threadgroup reduction and has every thread issue a global atomic.
// This isolates the value of the existing reduction strategy.
static bool scalarAtomicMode() {
    const char* e = std::getenv("GPUDB_SCALAR_ATOMIC");
    return e && e[0] && e[0] != '0';
}

// ===================================================================
// MetalGridStrideScan
// ===================================================================

MetalGridStrideScan::MetalGridStrideScan(const std::string& table,
                                         const std::string& rowVar,
                                         const std::string& idxVar)
    : tableName_(table), rowVar_(rowVar), idxVar_(idxVar) {}

void MetalGridStrideScan::addColumn(const std::string& paramName, const std::string& metalType) {
    columns_.push_back({paramName, metalType});
}

void MetalGridStrideScan::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.setPhaseScannedTable(tableName_);

    // All TPC-H scans are columnar (addColumn() called by the planner before
    // produce()). The legacy AoS struct path was removed — no current query
    // exercises it.
    if (columns_.empty()) {
        throw std::runtime_error(
            "MetalGridStrideScan(" + tableName_ +
            "): no columns registered. Call addColumn() before produce().");
    }
    for (const auto& col : columns_) {
        cg.addColumnParam(col.paramName, col.metalType, tableName_);
    }
    cg.addTableSizeParam(tableName_);

    // Emit grid-stride loop
    cg.addBlock("for (uint " + idxVar_ + " = tid; " + idxVar_ + " < " +
                tableSizeName(tableName_) + "; " + idxVar_ + " += tpg)", [&]() {
        consume();
    });
}

std::string MetalGridStrideScan::describe() const {
    return "GridStrideScan(" + tableName_ + ")";
}

// ===================================================================
// MetalSelection
// ===================================================================

MetalSelection::MetalSelection(std::unique_ptr<MetalOperator> child,
                               const std::string& predicate)
    : MetalUnaryOperator(std::move(child)), predicate_(predicate) {}

void MetalSelection::produce(MetalCodegen& cg, ConsumerFn consume) {
    child_->produce(cg, [&]() {
        cg.addIf(predicate_, [&]() {
            consume();
        });
    });
}

std::string MetalSelection::describe() const {
    return "Selection(" + predicate_ + ")";
}

// ===================================================================
// MetalComputeExpr
// ===================================================================

MetalComputeExpr::MetalComputeExpr(std::unique_ptr<MetalOperator> child,
                                   const std::string& varName,
                                   const std::string& varType,
                                   const std::string& expression)
    : MetalUnaryOperator(std::move(child)),
      varName_(varName), varType_(varType), expression_(expression) {}

void MetalComputeExpr::produce(MetalCodegen& cg, ConsumerFn consume) {
    child_->produce(cg, [&]() {
        cg.addLine(varType_ + " " + varName_ + " = " + expression_ + ";");
        consume();
    });
}

std::string MetalComputeExpr::describe() const {
    return "ComputeExpr(" + varName_ + " = " + expression_ + ")";
}

// ===================================================================
// MetalBitmapBuild
// ===================================================================

MetalBitmapBuild::MetalBitmapBuild(std::unique_ptr<MetalOperator> child,
                                   const std::string& bitmapName,
                                   const std::string& keyExpr,
                                   const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr), sizeExpr_(sizeExpr) {}

void MetalBitmapBuild::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register bitmap buffer as atomic for writes
    cg.addBitmapWriteParam(bitmapName_, sizeExpr_);

    child_->produce(cg, [&]() {
        cg.addLine("bitmap_set(" + bitmapName_ + ", " + keyExpr_ + ");");
        consume();  // allow chaining after bitmap set
    });
}

std::string MetalBitmapBuild::describe() const {
    return "BitmapBuild(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalBitmapProbe
// ===================================================================

MetalBitmapProbe::MetalBitmapProbe(std::unique_ptr<MetalOperator> child,
                                   const std::string& bitmapName,
                                   const std::string& keyExpr)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr) {}

void MetalBitmapProbe::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register bitmap as read-only in probe phase
    cg.addBitmapReadParam(bitmapName_, "");  // size comes from build phase

    child_->produce(cg, [&]() {
        cg.addIf("bitmap_test(" + bitmapName_ + ", " + keyExpr_ + ")", [&]() {
            consume();
        });
    });
}

std::string MetalBitmapProbe::describe() const {
    return "BitmapProbe(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalAntiBitmapProbe
// ===================================================================

MetalAntiBitmapProbe::MetalAntiBitmapProbe(std::unique_ptr<MetalOperator> child,
                                           const std::string& bitmapName,
                                           const std::string& keyExpr)
    : MetalUnaryOperator(std::move(child)),
      bitmapName_(bitmapName), keyExpr_(keyExpr) {}

void MetalAntiBitmapProbe::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addBitmapReadParam(bitmapName_, "");

    child_->produce(cg, [&]() {
        cg.addIf("!bitmap_test(" + bitmapName_ + ", " + keyExpr_ + ")", [&]() {
            consume();
        });
    });
}

std::string MetalAntiBitmapProbe::describe() const {
    return "AntiBitmapProbe(" + bitmapName_ + ", key=" + keyExpr_ + ")";
}

// ===================================================================
// MetalArrayStore
// ===================================================================

MetalArrayStore::MetalArrayStore(std::unique_ptr<MetalOperator> child,
                                 const std::string& arrayName,
                                 const std::string& keyExpr,
                                 const std::string& valueExpr,
                                 const std::string& valueType,
                                 const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), keyExpr_(keyExpr), valueExpr_(valueExpr),
      valueType_(valueType), sizeExpr_(sizeExpr) {}

void MetalArrayStore::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register array as device buffer, initialized to sentinel (-1 = 0xFF fill for int)
    cg.addBufferParam(arrayName_, valueType_, sizeExpr_, true, 0xFF);

    child_->produce(cg, [&]() {
        cg.addLine(arrayName_ + "[" + keyExpr_ + "] = " + valueExpr_ + ";");
        consume();
    });
}

std::string MetalArrayStore::describe() const {
    return "ArrayStore(" + arrayName_ + "[" + keyExpr_ + "] = " + valueExpr_ + ")";
}

// ===================================================================
// MetalArrayLookup
// ===================================================================

MetalArrayLookup::MetalArrayLookup(std::unique_ptr<MetalOperator> child,
                                   const std::string& arrayName,
                                   const std::string& keyExpr,
                                   const std::string& resultVar,
                                   const std::string& resultType,
                                   int sentinel)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), keyExpr_(keyExpr), resultVar_(resultVar),
      resultType_(resultType), sentinel_(sentinel) {}

void MetalArrayLookup::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Read-only access to array
    cg.addBufferParam(arrayName_, resultType_, "", false);

    child_->produce(cg, [&]() {
        cg.addLine(resultType_ + " " + resultVar_ + " = " + arrayName_ +
                   "[" + keyExpr_ + "];");
        // Guard: skip sentinel values (e.g., -1 means not found)
        cg.addIf(resultVar_ + " != " + std::to_string(sentinel_), [&]() {
            consume();
        });
    });
}

std::string MetalArrayLookup::describe() const {
    return "ArrayLookup(" + resultVar_ + " = " + arrayName_ + "[" + keyExpr_ + "])";
}

// ===================================================================
// MetalTGReduce
// ===================================================================

MetalTGReduce::MetalTGReduce(std::unique_ptr<MetalOperator> child,
                             const std::string& outputPrefix)
    : MetalUnaryOperator(std::move(child)), outputPrefix_(outputPrefix) {}

void MetalTGReduce::addAccumulator(const std::string& name,
                                    const std::string& valueExpr,
                                    const std::string& type,
                                    const std::string& loBuffer,
                                    const std::string& hiBuffer) {
    Accumulator acc;
    acc.name = name;
    acc.valueExpr = valueExpr;
    acc.type = type;
    acc.loBuffer = loBuffer.empty() ? (outputPrefix_ + "_" + name + "_lo") : loBuffer;
    acc.hiBuffer = hiBuffer.empty() ? (type == "long" ? (outputPrefix_ + "_" + name + "_hi") : "") : hiBuffer;
    acc.binIndex = static_cast<int>(accumulators_.size());
    accumulators_.push_back(acc);
}

void MetalTGReduce::setResultAlias(const std::string& displayName, int scaleDown) {
    resultInfos_.push_back({displayName, scaleDown});
}

void MetalTGReduce::produce(MetalCodegen& cg, ConsumerFn consume) {
    const bool scalar = scalarAtomicMode();

    // Register output buffers
    for (const auto& acc : accumulators_) {
        if (acc.type == "float") {
            // Float path: single atomic_uint buffer (reinterpreted as float via CAS)
            cg.addAtomicBufferParam(acc.loBuffer, "atomic_uint", "1");
        } else {
            // Long path: lo/hi atomic_uint pair
            cg.addAtomicBufferParam(acc.loBuffer, "atomic_uint", "1");
            cg.addAtomicBufferParam(acc.hiBuffer, "atomic_uint", "1");
        }
    }

    if (scalar) {
        // ===== SCALAR-ATOMIC ABLATION =====
        // Each thread issues a global atomic per row consumed. No local
        // accumulation, no SIMD/TG reduction, no shared memory.
        cg.addComment("--- Scalar-atomic mode: per-row global atomic ---");
        child_->produce(cg, [&]() {
            for (const auto& acc : accumulators_) {
                if (acc.type == "float") {
                    cg.addLine("atomic_add_float(" + acc.loBuffer + ", (float)("
                               + acc.valueExpr + "));");
                } else {
                    cg.addLine("atomic_add_long_pair(" + acc.loBuffer + ", "
                               + acc.hiBuffer + ", (long)(" + acc.valueExpr + "));");
                }
            }
            consume();
        });
    } else {
        // Declare local accumulators
        for (const auto& acc : accumulators_) {
            if (acc.type == "float") {
                cg.addLine("float local_" + acc.name + " = 0.0f;");
            } else {
                cg.addLine("long local_" + acc.name + " = 0;");
            }
        }

        // Child produces rows; inside the loop we accumulate
        child_->produce(cg, [&]() {
            for (const auto& acc : accumulators_) {
                if (acc.type == "float") {
                    cg.addLine("local_" + acc.name + " += (float)(" + acc.valueExpr + ");");
                } else {
                    cg.addLine("local_" + acc.name + " += (long)(" + acc.valueExpr + ");");
                }
            }
            consume();
        });

        // After the loop: SIMD + threadgroup reduction → atomic write
        cg.addComment("--- Threadgroup reduction ---");
        for (const auto& acc : accumulators_) {
            std::string localVar = "local_" + acc.name;
            std::string tgVar = "tg_" + acc.name;

            if (acc.type == "float") {
                cg.addLine("threadgroup float tg_shared_" + acc.name + "[32];");
                cg.addLine("float " + tgVar + " = tg_reduce_float(" + localVar +
                           ", lid, tg_size, tg_shared_" + acc.name + ");");
                cg.addIf("lid == 0", [&]() {
                    cg.addLine("atomic_add_float(" + acc.loBuffer + ", " + tgVar + ");");
                });
            } else {
                cg.addLine("threadgroup long tg_shared_" + acc.name + "[32];");
                cg.addLine("long " + tgVar + " = tg_reduce_long(" + localVar +
                           ", lid, tg_size, tg_shared_" + acc.name + ");");
                cg.addIf("lid == 0", [&]() {
                    cg.addLine("atomic_add_long_pair(" + acc.loBuffer + ", " +
                               acc.hiBuffer + ", " + tgVar + ");");
                });
            }
        }
    }

    // Register result schema
    if (!resultInfos_.empty()) {
        for (size_t i = 0; i < accumulators_.size() && i < resultInfos_.size(); i++) {
            const auto& acc = accumulators_[i];
            const auto& info = resultInfos_[i];
            cg.registerScalarAggOutput(acc.loBuffer, acc.hiBuffer, acc.type);
            cg.registerScalarAggColumn(info.displayName, (int)i, info.scaleDown);
        }
    }
}

std::string MetalTGReduce::describe() const {
    return "TGReduce(" + std::to_string(accumulators_.size()) + " accumulators)";
}

// ===================================================================
// MetalKeyedAgg
// ===================================================================

MetalKeyedAgg::MetalKeyedAgg(std::unique_ptr<MetalOperator> child,
                             const std::string& outputArrayName,
                             const std::string& bucketExpr,
                             int numBuckets,
                             int valuesPerBucket,
                             const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      outputArrayName_(outputArrayName), bucketExpr_(bucketExpr),
      numBuckets_(numBuckets), valuesPerBucket_(valuesPerBucket),
      sizeExpr_(sizeExpr) {}

void MetalKeyedAgg::addAggregate(const std::string& name, int offset,
                                  const std::string& valueExpr,
                                  const std::string& atomicOp,
                                  bool isLongPair,
                                  int scaleDown) {
    aggregates_.push_back({name, offset, valueExpr, atomicOp, isLongPair, scaleDown});
}

void MetalKeyedAgg::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register output buffer
    std::string sz = sizeExpr_.empty()
        ? std::to_string(numBuckets_ * valuesPerBucket_)
        : sizeExpr_;
    cg.addAtomicBufferParam(outputArrayName_, "atomic_uint", sz);

    // --- Thread-local accumulation + TG reduction strategy ---
    // Instead of per-row global atomics, accumulate in thread-local arrays,
    // then do threadgroup SIMD reduction and a single atomic per TG per bucket.
    // This reduces atomic operations from O(rows) to O(threadgroups × buckets).

    // Check if all aggregates are "add" (reduction-compatible)
    bool allAdds = true;
    for (const auto& agg : aggregates_) {
        if (agg.atomicOp != "add") { allAdds = false; break; }
    }

    // Only use TG reduction when there are enough aggregates per row to justify
    // the reduction overhead. With few aggs (e.g. 1 count), the barrier cost
    // exceeds the atomic savings, especially for low-selectivity joins.
    //
    // Tuning knobs (empirical, see /memories/repo/metal_codegen_optimizations.md):
    //   - kMaxBucketsForTGReduce: per-thread accumulator array length cap.
    //     Above this, register pressure and TG-shared memory dominate.
    //   - kMinAggsForTGReduce: minimum aggs per row to amortise the two-level
    //     reduction barrier cost.
    constexpr int kMaxBucketsForTGReduce = 64;
    constexpr int kMinAggsForTGReduce    = 3;
    if (allAdds && numBuckets_ <= kMaxBucketsForTGReduce &&
        (int)aggregates_.size() >= kMinAggsForTGReduce) {
        // === OPTIMIZED PATH: thread-local + TG reduction ===
        cg.setPhaseMaxThreadgroups(1024);

        // Declare thread-local accumulator arrays (before the scan loop)
        for (const auto& agg : aggregates_) {
            if (agg.isLongPair) {
                cg.addLine("long _local_" + agg.name + "[" + std::to_string(numBuckets_) + "];");
                cg.addBlock("for (int _b = 0; _b < " + std::to_string(numBuckets_) + "; _b++)", [&]() {
                    cg.addLine("_local_" + agg.name + "[_b] = 0;");
                });
            } else {
                cg.addLine("uint _local_" + agg.name + "[" + std::to_string(numBuckets_) + "];");
                cg.addBlock("for (int _b = 0; _b < " + std::to_string(numBuckets_) + "; _b++)", [&]() {
                    cg.addLine("_local_" + agg.name + "[_b] = 0;");
                });
            }
        }

        // Child produces rows; inside the loop we accumulate locally (no atomics)
        child_->produce(cg, [&]() {
            cg.addLine("int _bucket = " + bucketExpr_ + ";");
            for (const auto& agg : aggregates_) {
                if (agg.isLongPair) {
                    cg.addLine("_local_" + agg.name + "[_bucket] += (long)(" + agg.valueExpr + ");");
                } else {
                    cg.addLine("_local_" + agg.name + "[_bucket] += (uint)(" + agg.valueExpr + ");");
                }
            }
            consume();
        });

        // After the loop: TG reduction per bucket per aggregate, then single atomic
        cg.addComment("--- Threadgroup reduction for keyed aggregation ---");
        for (const auto& agg : aggregates_) {
            if (agg.isLongPair) {
                cg.addLine("threadgroup long _tg_shared_" + agg.name + "[32];");
                cg.addBlock("for (int _b = 0; _b < " + std::to_string(numBuckets_) + "; _b++)", [&]() {
                    cg.addLine("long _tg_val = tg_reduce_long(_local_" + agg.name +
                               "[_b], lid, tg_size, _tg_shared_" + agg.name + ");");
                    cg.addIf("lid == 0 && _tg_val != 0", [&]() {
                        std::string loIdx = "_b * " + std::to_string(valuesPerBucket_) + " + " + std::to_string(agg.offset);
                        std::string hiIdx = "_b * " + std::to_string(valuesPerBucket_) + " + " + std::to_string(agg.offset + 1);
                        cg.addLine("atomic_add_long_pair(&" + outputArrayName_ + "[" + loIdx +
                                   "], &" + outputArrayName_ + "[" + hiIdx +
                                   "], _tg_val);");
                    });
                });
            } else {
                cg.addLine("threadgroup uint _tg_shared_" + agg.name + "[32];");
                cg.addBlock("for (int _b = 0; _b < " + std::to_string(numBuckets_) + "; _b++)", [&]() {
                    cg.addLine("uint _tg_val = tg_reduce_uint(_local_" + agg.name +
                               "[_b], lid, tg_size, _tg_shared_" + agg.name + ");");
                    cg.addIf("lid == 0 && _tg_val != 0", [&]() {
                        std::string idx = "_b * " + std::to_string(valuesPerBucket_) + " + " + std::to_string(agg.offset);
                        cg.addLine("atomic_fetch_add_explicit(&" + outputArrayName_ + "[" + idx +
                                   "], _tg_val, memory_order_relaxed);");
                    });
                });
            }
        }
    } else {
        // === FALLBACK: per-row global atomics (for min/max or high-cardinality) ===
        child_->produce(cg, [&]() {
            cg.addLine("int _bucket = " + bucketExpr_ + ";");
            for (const auto& agg : aggregates_) {
                std::string base = "_bucket * " + std::to_string(valuesPerBucket_);
                if (agg.isLongPair && agg.atomicOp == "add") {
                    std::string loIdx = base + " + " + std::to_string(agg.offset);
                    std::string hiIdx = base + " + " + std::to_string(agg.offset + 1);
                    cg.addLine("atomic_add_long_pair(&" + outputArrayName_ + "[" + loIdx +
                               "], &" + outputArrayName_ + "[" + hiIdx +
                               "], (long)(" + agg.valueExpr + "));");
                } else {
                    std::string idx = base + " + " + std::to_string(agg.offset);
                    if (agg.atomicOp == "add") {
                        cg.addLine("atomic_fetch_add_explicit(&" + outputArrayName_ + "[" + idx +
                                   "], (uint)(" + agg.valueExpr + "), memory_order_relaxed);");
                    } else if (agg.atomicOp == "min") {
                        cg.addLine("atomic_fetch_min_explicit(&" + outputArrayName_ + "[" + idx +
                                   "], (uint)(" + agg.valueExpr + "), memory_order_relaxed);");
                    } else if (agg.atomicOp == "max") {
                        cg.addLine("atomic_fetch_max_explicit(&" + outputArrayName_ + "[" + idx +
                                   "], (uint)(" + agg.valueExpr + "), memory_order_relaxed);");
                    }
                }
            }
            consume();
        });
    }

    // Register result schema with slot layout
    std::vector<MetalResultSchema::KeyedAggSlot> slots;
    for (const auto& agg : aggregates_) {
        slots.push_back({agg.name, agg.offset, agg.isLongPair, agg.scaleDown});
    }
    cg.registerKeyedAggOutput(outputArrayName_, numBuckets_, valuesPerBucket_, slots);
}

std::string MetalKeyedAgg::describe() const {
    return "KeyedAgg(" + outputArrayName_ + ", " + std::to_string(numBuckets_) +
           " buckets, " + std::to_string(aggregates_.size()) + " aggs)";
}

// ===================================================================
// MetalAtomicAgg
// ===================================================================

MetalAtomicAgg::MetalAtomicAgg(std::unique_ptr<MetalOperator> child,
                               const std::string& arrayName,
                               const std::string& bucketExpr,
                               const std::string& valueExpr,
                               const std::string& sizeExpr,
                               const std::string& atomicType,
                               const std::string& castType)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), bucketExpr_(bucketExpr),
      valueExpr_(valueExpr), sizeExpr_(sizeExpr),
      atomicType_(atomicType), castType_(castType) {}

void MetalAtomicAgg::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addAtomicBufferParam(arrayName_, atomicType_, sizeExpr_);

    child_->produce(cg, [&]() {
        cg.addLine("atomic_fetch_add_explicit(&" + arrayName_ + "[" + bucketExpr_ +
                   "], (" + castType_ + ")(" + valueExpr_ + "), memory_order_relaxed);");
        consume();
    });
}

std::string MetalAtomicAgg::describe() const {
    return "AtomicAgg(" + arrayName_ + "[" + bucketExpr_ + "])";
}

// ===================================================================
// MetalAtomicCount
// ===================================================================

MetalAtomicCount::MetalAtomicCount(std::unique_ptr<MetalOperator> child,
                                   const std::string& arrayName,
                                   const std::string& bucketExpr,
                                   const std::string& sizeExpr)
    : MetalUnaryOperator(std::move(child)),
      arrayName_(arrayName), bucketExpr_(bucketExpr), sizeExpr_(sizeExpr) {}

void MetalAtomicCount::produce(MetalCodegen& cg, ConsumerFn consume) {
    cg.addAtomicBufferParam(arrayName_, "atomic_uint", sizeExpr_);

    // Parse sizeExpr to determine if we can use threadgroup-local histogram.
    // For small, statically-known bucket counts (≤ 256), use TG-local histogram
    // to drastically reduce global atomic contention.
    int staticSize = 0;
    try { staticSize = std::stoi(sizeExpr_); } catch (...) {}

    if (staticSize > 0 && staticSize <= 256) {
        // === OPTIMIZED: Threadgroup-local histogram ===
        cg.setPhaseMaxThreadgroups(1024);
        std::string szStr = std::to_string(staticSize);

        // Declare threadgroup-local histogram and zero-initialize
        cg.addLine("threadgroup uint _tg_hist_" + arrayName_ + "[" + szStr + "];");
        cg.addBlock("for (uint _h = lid; _h < " + szStr + "u; _h += tg_size)", [&]() {
            cg.addLine("_tg_hist_" + arrayName_ + "[_h] = 0;");
        });
        cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");

        // Child scan loop — accumulate into threadgroup-local histogram
        child_->produce(cg, [&]() {
            cg.addLine("atomic_fetch_add_explicit((threadgroup atomic_uint*)&_tg_hist_" +
                       arrayName_ + "[" + bucketExpr_ + "], 1u, memory_order_relaxed);");
            consume();
        });

        // Barrier, then flush non-zero bins to global
        cg.addLine("threadgroup_barrier(mem_flags::mem_threadgroup);");
        cg.addBlock("for (uint _h = lid; _h < " + szStr + "u; _h += tg_size)", [&]() {
            cg.addIf("_tg_hist_" + arrayName_ + "[_h] > 0", [&]() {
                cg.addLine("atomic_fetch_add_explicit(&" + arrayName_ + "[_h], " +
                           "_tg_hist_" + arrayName_ + "[_h], memory_order_relaxed);");
            });
        });
    } else {
        // === FALLBACK: per-row global atomics ===
        child_->produce(cg, [&]() {
            cg.addLine("atomic_fetch_add_explicit(&" + arrayName_ + "[" + bucketExpr_ +
                       "], 1u, memory_order_relaxed);");
            consume();
        });
    }
}

std::string MetalAtomicCount::describe() const {
    return "AtomicCount(" + arrayName_ + "[" + bucketExpr_ + "])";
}

// ===================================================================
// MetalMaterialize
// ===================================================================

MetalMaterialize::MetalMaterialize(std::unique_ptr<MetalOperator> child,
                                   const std::string& counterName,
                                   const std::string& counterSizeExpr)
    : MetalUnaryOperator(std::move(child)),
      counterName_(counterName), counterSizeExpr_(counterSizeExpr) {}

void MetalMaterialize::addColumn(const std::string& arrayName, const std::string& type,
                                  const std::string& valueExpr,
                                  const std::string& displayName,
                                  const std::string& sizeExpr) {
    columns_.push_back({arrayName, type, valueExpr,
                        displayName.empty() ? arrayName : displayName, sizeExpr});
}

void MetalMaterialize::produce(MetalCodegen& cg, ConsumerFn consume) {
    // Register counter
    cg.addAtomicBufferParam(counterName_, "atomic_uint", counterSizeExpr_);

    // Register output column buffers
    for (const auto& col : columns_) {
        cg.addBufferParam(col.arrayName, col.type, col.sizeExpr, false);
    }

    // Register result schema
    cg.registerMaterializeOutput(counterName_);
    for (const auto& col : columns_) {
        cg.registerOutputColumn(col.displayName, col.arrayName, col.type);
    }

    child_->produce(cg, [&]() {
        // Atomic increment counter to get output position
        cg.addLine("uint _pos = atomic_fetch_add_explicit(&" + counterName_ +
                   "[0], 1u, memory_order_relaxed);");
        // Scatter values to output arrays
        for (const auto& col : columns_) {
            cg.addLine(col.arrayName + "[_pos] = " + col.valueExpr + ";");
        }
        consume();
    });
}

std::string MetalMaterialize::describe() const {
    return "Materialize(" + std::to_string(columns_.size()) + " columns)";
}

} // namespace codegen
