#include "chunked_colbin_loader.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace codegen {

namespace {

std::string errnoMessage(const std::string& path) {
    return path + ": " + std::strerror(errno);
}

} // namespace

ChunkedColbinTable::~ChunkedColbinTable() {
    close();
}

bool ChunkedColbinTable::open(MTL::Device* device,
                              const std::string& dataPath,
                              const std::vector<ColSpec>& specs,
                              size_t chunkRows,
                              int slotCount,
                              std::string& error) {
    close();
    if (!device) {
        error = "no Metal device";
        return false;
    }
    if (specs.empty()) {
        error = "no columns requested";
        return false;
    }
    if (chunkRows == 0) {
        error = "chunk row count must be positive";
        return false;
    }
    if (slotCount < 1) slotCount = 1;

    size_t tblSize = 0;
    int64_t tblMtime = 0;
    const bool checkSource = !colbin::isBinaryPath(dataPath);
    const bool tblPresent = checkSource &&
        colbin::statFile(colbin::textPath(dataPath), tblSize, tblMtime);
    // .tbl is optional once .colbin has been generated; if absent we skip
    // the source-size/mtime cross-check and trust the .colbin header.

    const std::string colbinPath = colbin::binaryPath(dataPath);
    int fd = ::open(colbinPath.c_str(), O_RDONLY);
    if (fd < 0) {
        error = "missing .colbin for chunked execution: " + errnoMessage(colbinPath);
        return false;
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        error = "cannot stat .colbin " + errnoMessage(colbinPath);
        ::close(fd);
        return false;
    }
    const size_t fileSize = (size_t)st.st_size;
    if (fileSize < sizeof(colbin::FileHeader)) {
        error = "invalid .colbin header in " + colbinPath;
        ::close(fd);
        return false;
    }

    void* base = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
        error = "cannot mmap .colbin " + errnoMessage(colbinPath);
        return false;
    }
    ::madvise(base, fileSize, MADV_SEQUENTIAL);

    colbin::FileHeader header{};
    std::memcpy(&header, base, sizeof(header));
    if (!colbin::validateHeader(header, fileSize, tblPresent, tblSize, tblMtime)) {
        ::munmap(base, fileSize);
        error = "stale or invalid .colbin for " + dataPath + " (run make colbin-sfN)";
        return false;
    }

    const auto* descs = reinterpret_cast<const colbin::ColDesc*>(
        static_cast<const char*>(base) + sizeof(colbin::FileHeader));
    std::unordered_map<int, const colbin::ColDesc*> byIndex;
    byIndex.reserve(header.n_cols);
    for (uint32_t i = 0; i < header.n_cols; ++i) {
        byIndex[descs[i].columnIndex] = &descs[i];
    }

    std::vector<ColumnInfo> columns;
    columns.reserve(specs.size());
    for (const auto& spec : specs) {
        auto it = byIndex.find(spec.columnIndex);
        if (it == byIndex.end() ||
            !colbin::descriptorMatches(*it->second, spec, header.n_rows, fileSize)) {
            ::munmap(base, fileSize);
            error = "column descriptor mismatch in " + colbinPath + " (run make colbin-sfN)";
            return false;
        }
        columns.push_back(ColumnInfo{spec, *it->second, colbin::elementBytes(spec)});
    }

    std::vector<Slot> slots((size_t)slotCount);
    const size_t maxRowsInSlot = std::min(chunkRows, (size_t)header.n_rows);
    for (auto& slot : slots) {
        for (const auto& col : columns) {
            const size_t bytes = std::max((size_t)1, maxRowsInSlot * col.elemBytes);
            MTL::Buffer* buffer = device->newBuffer(bytes, MTL::ResourceStorageModeShared);
            if (!buffer) {
                for (auto& cleanupSlot : slots) {
                    for (auto& [_, owned] : cleanupSlot.buffers) if (owned) owned->release();
                }
                ::munmap(base, fileSize);
                error = "Metal buffer allocation failed for chunked table " + dataPath;
                return false;
            }
            slot.buffers[col.spec.columnIndex] = buffer;
        }
    }

    device_ = device;
    mapBase_ = base;
    mapSize_ = fileSize;
    rows_ = (size_t)header.n_rows;
    chunkRows_ = chunkRows;
    columns_ = std::move(columns);
    slots_ = std::move(slots);
    bytesLoaded_ = 0;
    return true;
}

bool ChunkedColbinTable::loadChunk(int slot, size_t startRow, size_t rowCount, std::string& error) {
    if (!mapBase_) {
        error = "chunked table is not open";
        return false;
    }
    if (slot < 0 || slot >= (int)slots_.size()) {
        error = "chunk slot index out of range";
        return false;
    }
    if (rowCount > chunkRows_ || startRow > rows_ || startRow + rowCount > rows_) {
        error = "chunk row range out of bounds";
        return false;
    }

    const char* base = static_cast<const char*>(mapBase_);
    for (const auto& col : columns_) {
        auto it = slots_[(size_t)slot].buffers.find(col.spec.columnIndex);
        if (it == slots_[(size_t)slot].buffers.end() || !it->second) {
            error = "missing chunk slot buffer";
            return false;
        }
        const size_t bytes = rowCount * col.elemBytes;
        const char* src = base + col.desc.offset + startRow * col.elemBytes;
        std::memcpy(it->second->contents(), src, bytes);
        bytesLoaded_ += bytes;
    }
    return true;
}

MTL::Buffer* ChunkedColbinTable::buffer(int slot, int columnIndex) const {
    if (slot < 0 || slot >= (int)slots_.size()) return nullptr;
    const auto& buffers = slots_[(size_t)slot].buffers;
    auto it = buffers.find(columnIndex);
    return it == buffers.end() ? nullptr : it->second;
}

void ChunkedColbinTable::close() {
    for (auto& slot : slots_) {
        for (auto& [_, buffer] : slot.buffers) {
            if (buffer) buffer->release();
        }
        slot.buffers.clear();
    }
    slots_.clear();
    columns_.clear();
    if (mapBase_ && mapSize_) {
        ::munmap(mapBase_, mapSize_);
    }
    mapBase_ = nullptr;
    mapSize_ = 0;
    rows_ = 0;
    chunkRows_ = 0;
    bytesLoaded_ = 0;
    device_ = nullptr;
}

} // namespace codegen
