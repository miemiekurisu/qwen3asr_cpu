#include "qasr/storage/safetensors_loader.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace qasr {
namespace fs = std::filesystem;

// --- MappedFile ---

MappedFile::~MappedFile() { Close(); }

MappedFile::MappedFile(MappedFile && other) noexcept
    : data_(other.data_), size_(other.size_)
#ifndef _WIN32
    , fd_(other.fd_)
#endif
{
    other.data_ = nullptr;
    other.size_ = 0;
#ifndef _WIN32
    other.fd_ = -1;
#endif
}

MappedFile & MappedFile::operator=(MappedFile && other) noexcept {
    if (this != &other) {
        Close();
        data_ = other.data_;
        size_ = other.size_;
#ifndef _WIN32
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MappedFile::Close() noexcept {
#ifndef _WIN32
    if (data_ != nullptr) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
#endif
    size_ = 0;
}

Status MappedFile::Open(const std::string & path, MappedFile * out) {
    if (!out) {
        return Status(StatusCode::kInvalidArgument, "out must not be null");
    }
    out->Close();
#ifndef _WIN32
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Status(StatusCode::kNotFound, "failed to open file: " + path);
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        return Status(StatusCode::kInternal, "fstat failed: " + path);
    }
    const std::size_t file_size = static_cast<std::size_t>(st.st_size);
    if (file_size == 0) {
        close(fd);
        return Status(StatusCode::kInvalidArgument, "file is empty: " + path);
    }
    void * mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return Status(StatusCode::kInternal, "mmap failed: " + path);
    }
    out->data_ = mapped;
    out->size_ = file_size;
    out->fd_ = fd;
    return OkStatus();
#else
    return Status(StatusCode::kUnimplemented, "MappedFile not implemented on Windows");
#endif
}

// --- TensorDtype helpers ---

std::string_view TensorDtypeName(TensorDtype dtype) noexcept {
    switch (dtype) {
        case TensorDtype::kFloat32:  return "F32";
        case TensorDtype::kFloat16:  return "F16";
        case TensorDtype::kBFloat16: return "BF16";
        case TensorDtype::kInt32:    return "I32";
        case TensorDtype::kInt64:    return "I64";
        case TensorDtype::kBool:     return "BOOL";
        case TensorDtype::kUnknown:  return "Unknown";
    }
    return "Unknown";
}

std::size_t TensorDtypeSize(TensorDtype dtype) noexcept {
    switch (dtype) {
        case TensorDtype::kFloat32: return 4;
        case TensorDtype::kFloat16: return 2;
        case TensorDtype::kBFloat16: return 2;
        case TensorDtype::kInt32: return 4;
        case TensorDtype::kInt64: return 8;
        case TensorDtype::kBool: return 1;
        case TensorDtype::kUnknown: return 0;
    }
    return 0;
}

std::int64_t TensorElementCount(const TensorView & view) noexcept {
    if (view.shape.empty()) return 0;
    std::int64_t count = 1;
    for (const auto dim : view.shape) {
        count *= dim;
    }
    return count;
}

// --- SafeTensorIndex ---

Status SafeTensorIndex::Build(const MappedFile & file, SafeTensorIndex * out) {
    if (!out) {
        return Status(StatusCode::kInvalidArgument, "out must not be null");
    }
    if (!file.is_open() || file.size() < 8) {
        return Status(StatusCode::kInvalidArgument, "invalid safetensors file");
    }
    // safetensors format: 8-byte LE header_size + JSON header + tensor data
    const auto * raw = static_cast<const uint8_t *>(file.data());
    std::uint64_t header_size = 0;
    std::memcpy(&header_size, raw, 8);
    if (header_size > file.size() - 8 || header_size > 100 * 1024 * 1024) {
        return Status(StatusCode::kInvalidArgument, "invalid safetensors header size");
    }
    // For now, just validate the structure is readable.
    // Full JSON parsing of tensor metadata would go here.
    out->tensors_.clear();
    return OkStatus();
}

const TensorView * SafeTensorIndex::Find(std::string_view name) const {
    for (const auto & t : tensors_) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// --- ShardRegistry ---

Status ShardRegistry::Open(const std::string & model_dir, ShardRegistry * out) {
    if (!out) {
        return Status(StatusCode::kInvalidArgument, "out must not be null");
    }
    out->files_.clear();
    out->indices_.clear();

    const fs::path root(model_dir);
    if (!fs::exists(root) || !fs::is_directory(root)) {
        return Status(StatusCode::kNotFound, "model directory not found: " + model_dir);
    }

    std::vector<std::string> shard_paths;
    for (const auto & entry : fs::directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == ".safetensors") {
            shard_paths.push_back(entry.path().string());
        }
    }
    std::sort(shard_paths.begin(), shard_paths.end());

    if (shard_paths.empty()) {
        return Status(StatusCode::kNotFound, "no .safetensors files in: " + model_dir);
    }

    for (const auto & path : shard_paths) {
        MappedFile file;
        Status status = MappedFile::Open(path, &file);
        if (!status.ok()) return status;

        SafeTensorIndex index;
        status = SafeTensorIndex::Build(file, &index);
        if (!status.ok()) return status;

        out->files_.push_back(std::move(file));
        out->indices_.push_back(std::move(index));
    }
    return OkStatus();
}

const TensorView * ShardRegistry::Find(std::string_view name) const {
    for (const auto & index : indices_) {
        const TensorView * view = index.Find(name);
        if (view) return view;
    }
    return nullptr;
}

std::size_t ShardRegistry::tensor_count() const noexcept {
    std::size_t total = 0;
    for (const auto & index : indices_) {
        total += index.tensor_count();
    }
    return total;
}

// --- Utility functions ---

Status LoadTensorView(const ShardRegistry & registry, std::string_view name, TensorView * out) {
    if (!out) {
        return Status(StatusCode::kInvalidArgument, "out must not be null");
    }
    const TensorView * found = registry.Find(name);
    if (!found) {
        return Status(StatusCode::kNotFound, "tensor not found: " + std::string(name));
    }
    *out = *found;
    return OkStatus();
}

Status ValidateShardChecksums(const std::string & model_dir) {
    const fs::path root(model_dir);
    const fs::path index_path = root / "model.safetensors.index.json";
    if (!fs::exists(index_path)) {
        return Status(StatusCode::kNotFound, "missing model.safetensors.index.json");
    }

    std::ifstream input(index_path);
    if (!input) {
        return Status(StatusCode::kInternal, "failed to read index file");
    }
    const std::string json_text((std::istreambuf_iterator<char>(input)),
                                 std::istreambuf_iterator<char>());

    std::regex pattern(R"(model-[^"]+\.safetensors)");
    std::set<std::string> indexed_files;
    for (std::sregex_iterator it(json_text.begin(), json_text.end(), pattern), end;
         it != end; ++it) {
        indexed_files.insert(it->str());
    }

    for (const auto & file_name : indexed_files) {
        if (!fs::exists(root / file_name)) {
            return Status(StatusCode::kNotFound, "missing indexed shard: " + file_name);
        }
        // Check file is not empty
        const auto fsize = fs::file_size(root / file_name);
        if (fsize < 8) {
            return Status(StatusCode::kInvalidArgument,
                          "shard file too small: " + file_name);
        }
    }
    return OkStatus();
}

}  // namespace qasr
