#pragma once

#include "../../src/infra.h"

#include <Metal/Metal.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace codegen {

class ChunkedColbinTable {
public:
    ChunkedColbinTable() = default;
    ~ChunkedColbinTable();

    ChunkedColbinTable(const ChunkedColbinTable&) = delete;
    ChunkedColbinTable& operator=(const ChunkedColbinTable&) = delete;

    bool open(MTL::Device* device,
              const std::string& tblPath,
              const std::vector<ColSpec>& specs,
              size_t chunkRows,
              int slotCount,
              std::string& error);

    bool loadChunk(int slot, size_t startRow, size_t rowCount, std::string& error);
    MTL::Buffer* buffer(int slot, int columnIndex) const;

    size_t rows() const { return rows_; }
    size_t chunkRows() const { return chunkRows_; }
    size_t bytesLoaded() const { return bytesLoaded_; }
    int slotCount() const { return (int)slots_.size(); }

    void close();

private:
    struct ColumnInfo {
        ColSpec spec;
        colbin::ColDesc desc{};
        size_t elemBytes = 0;
    };
    struct Slot {
        std::unordered_map<int, MTL::Buffer*> buffers;
    };

    MTL::Device* device_ = nullptr;
    void* mapBase_ = nullptr;
    size_t mapSize_ = 0;
    size_t rows_ = 0;
    size_t chunkRows_ = 0;
    size_t bytesLoaded_ = 0;
    std::vector<ColumnInfo> columns_;
    std::vector<Slot> slots_;
};

} // namespace codegen