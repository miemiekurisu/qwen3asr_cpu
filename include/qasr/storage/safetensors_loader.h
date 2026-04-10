#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "qasr/core/status.h"

namespace qasr {

/// Cross-platform mmap wrapper. RAII: unmaps on destruction.
/// Pre: path must be a regular file.
/// Post: data() returns read-only pointer; size() returns file size.
/// Thread-safe: immutable after construction.
class MappedFile {
public:
    MappedFile() noexcept = default;
    ~MappedFile();

    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
    MappedFile(MappedFile && other) noexcept;
    MappedFile & operator=(MappedFile && other) noexcept;

    static Status Open(const std::string & path, MappedFile * out);

    const void * data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    bool is_open() const noexcept { return data_ != nullptr; }

private:
    void Close() noexcept;
    void * data_ = nullptr;
    std::size_t size_ = 0;
#ifdef _WIN32
    void * file_handle_ = nullptr;
    void * mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

enum class TensorDtype {
    kFloat32 = 0,
    kFloat16,
    kBFloat16,
    kInt32,
    kInt64,
    kBool,
    kUnknown = -1,
};

struct TensorView {
    std::string name;
    TensorDtype dtype = TensorDtype::kUnknown;
    std::vector<std::int64_t> shape;
    const void * data = nullptr;
    std::size_t data_size = 0;
};

/// Single-file safetensors index.
/// Pre: file must be mapped via MappedFile.
/// Post: tensors can be looked up by name.
/// Thread-safe: immutable after construction.
class SafeTensorIndex {
public:
    static Status Build(const MappedFile & file, SafeTensorIndex * out);

    const TensorView * Find(std::string_view name) const;
    std::size_t tensor_count() const noexcept { return tensors_.size(); }

private:
    std::vector<TensorView> tensors_;
};

/// Multi-shard consistency registry.
/// Pre: model directory must exist and contain at least one .safetensors file.
/// Post: provides unified tensor lookup across all shards.
/// Thread-safe: immutable after construction.
class ShardRegistry {
public:
    static Status Open(const std::string & model_dir, ShardRegistry * out);

    const TensorView * Find(std::string_view name) const;
    std::size_t shard_count() const noexcept { return files_.size(); }
    std::size_t tensor_count() const noexcept;

private:
    std::vector<MappedFile> files_;
    std::vector<SafeTensorIndex> indices_;
};

/// Pre: registry must be open. name must exist.
/// Post: returns read-only tensor view or error.
/// Thread-safe: yes (read-only).
Status LoadTensorView(const ShardRegistry & registry, std::string_view name, TensorView * out);

/// Pre: model_dir must contain model.safetensors.index.json.
/// Post: returns Ok if all indexed shards exist and have valid headers.
/// Thread-safe: yes.
Status ValidateShardChecksums(const std::string & model_dir);

std::string_view TensorDtypeName(TensorDtype dtype) noexcept;
std::size_t TensorDtypeSize(TensorDtype dtype) noexcept;
std::int64_t TensorElementCount(const TensorView & view) noexcept;

}  // namespace qasr
