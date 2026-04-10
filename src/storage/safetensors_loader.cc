#include "qasr/storage/safetensors_loader.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace qasr {
namespace fs = std::filesystem;

// --- MappedFile ---

MappedFile::~MappedFile() { Close(); }

MappedFile::MappedFile(MappedFile && other) noexcept
    : data_(other.data_), size_(other.size_)
#ifdef _WIN32
    , file_handle_(other.file_handle_), mapping_handle_(other.mapping_handle_)
#else
    , fd_(other.fd_)
#endif
{
    other.data_ = nullptr;
    other.size_ = 0;
#ifdef _WIN32
    other.file_handle_ = nullptr;
    other.mapping_handle_ = nullptr;
#else
    other.fd_ = -1;
#endif
}

MappedFile & MappedFile::operator=(MappedFile && other) noexcept {
    if (this != &other) {
        Close();
        data_ = other.data_;
        size_ = other.size_;
#ifdef _WIN32
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MappedFile::Close() noexcept {
#ifdef _WIN32
    if (data_ != nullptr) {
        UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_handle_ != nullptr) {
        CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }
    if (file_handle_ != nullptr && file_handle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(file_handle_);
        file_handle_ = nullptr;
    }
#else
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
#ifdef _WIN32
    HANDLE file_handle = CreateFileA(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle == INVALID_HANDLE_VALUE) {
        return Status(StatusCode::kNotFound, "failed to open file: " + path);
    }
    LARGE_INTEGER file_size_li;
    if (!GetFileSizeEx(file_handle, &file_size_li)) {
        CloseHandle(file_handle);
        return Status(StatusCode::kInternal, "GetFileSizeEx failed: " + path);
    }
    const std::size_t file_size = static_cast<std::size_t>(file_size_li.QuadPart);
    if (file_size == 0) {
        CloseHandle(file_handle);
        return Status(StatusCode::kInvalidArgument, "file is empty: " + path);
    }
    HANDLE mapping_handle = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_handle == nullptr) {
        CloseHandle(file_handle);
        return Status(StatusCode::kInternal, "CreateFileMapping failed: " + path);
    }
    void * mapped = MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
    if (mapped == nullptr) {
        CloseHandle(mapping_handle);
        CloseHandle(file_handle);
        return Status(StatusCode::kInternal, "MapViewOfFile failed: " + path);
    }
    out->data_ = mapped;
    out->size_ = file_size;
    out->file_handle_ = file_handle;
    out->mapping_handle_ = mapping_handle;
    return OkStatus();
#else
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
