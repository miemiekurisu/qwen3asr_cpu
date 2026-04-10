#include "tests/test_registry.h"
#include "qasr/storage/safetensors_loader.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string CreateTempFile(const std::string & name, const void * data, std::size_t size) {
    const std::string path = (std::filesystem::temp_directory_path() / ("qasr_test_" + name)).string();
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    return path;
}

void RemoveTempFile(const std::string & path) {
    std::remove(path.c_str());
}

}  // namespace

// --- Normal ---

QASR_TEST(TensorDtypeNameCoversAllTypes) {
    QASR_EXPECT_EQ(qasr::TensorDtypeName(qasr::TensorDtype::kFloat32), "F32");
    QASR_EXPECT_EQ(qasr::TensorDtypeName(qasr::TensorDtype::kFloat16), "F16");
    QASR_EXPECT_EQ(qasr::TensorDtypeName(qasr::TensorDtype::kBFloat16), "BF16");
    QASR_EXPECT_EQ(qasr::TensorDtypeName(qasr::TensorDtype::kInt32), "I32");
    QASR_EXPECT_EQ(qasr::TensorDtypeName(qasr::TensorDtype::kInt64), "I64");
    QASR_EXPECT_EQ(qasr::TensorDtypeName(qasr::TensorDtype::kBool), "BOOL");
    QASR_EXPECT_EQ(qasr::TensorDtypeName(qasr::TensorDtype::kUnknown), "Unknown");
}

QASR_TEST(TensorDtypeSizeCorrect) {
    QASR_EXPECT_EQ(qasr::TensorDtypeSize(qasr::TensorDtype::kFloat32), std::size_t(4));
    QASR_EXPECT_EQ(qasr::TensorDtypeSize(qasr::TensorDtype::kFloat16), std::size_t(2));
    QASR_EXPECT_EQ(qasr::TensorDtypeSize(qasr::TensorDtype::kBFloat16), std::size_t(2));
    QASR_EXPECT_EQ(qasr::TensorDtypeSize(qasr::TensorDtype::kInt32), std::size_t(4));
    QASR_EXPECT_EQ(qasr::TensorDtypeSize(qasr::TensorDtype::kInt64), std::size_t(8));
    QASR_EXPECT_EQ(qasr::TensorDtypeSize(qasr::TensorDtype::kBool), std::size_t(1));
    QASR_EXPECT_EQ(qasr::TensorDtypeSize(qasr::TensorDtype::kUnknown), std::size_t(0));
}

QASR_TEST(TensorElementCountEmpty) {
    qasr::TensorView view;
    QASR_EXPECT_EQ(qasr::TensorElementCount(view), std::int64_t(0));
}

QASR_TEST(TensorElementCountMultiDim) {
    qasr::TensorView view;
    view.shape = {2, 3, 4};
    QASR_EXPECT_EQ(qasr::TensorElementCount(view), std::int64_t(24));
}

// --- MappedFile ---

QASR_TEST(MappedFileOpenNonexistent) {
    qasr::MappedFile file;
    const std::string nonexistent = (std::filesystem::temp_directory_path() / "qasr_does_not_exist_12345").string();
    qasr::Status s = qasr::MappedFile::Open(nonexistent, &file);
    QASR_EXPECT(!s.ok());
    QASR_EXPECT(!file.is_open());
}

QASR_TEST(MappedFileOpenValid) {
    const std::string data = "hello safetensors";
    const std::string path = CreateTempFile("mapped.bin", data.data(), data.size());
    qasr::MappedFile file;
    qasr::Status s = qasr::MappedFile::Open(path, &file);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(file.is_open());
    QASR_EXPECT_EQ(file.size(), data.size());
    RemoveTempFile(path);
}

QASR_TEST(MappedFileOpenEmpty) {
    const std::string path = CreateTempFile("empty.bin", "", 0);
    // Write a zero-length file
    { std::ofstream out(path, std::ios::binary); }
    qasr::MappedFile file;
    qasr::Status s = qasr::MappedFile::Open(path, &file);
    QASR_EXPECT(!s.ok());  // Empty files are rejected
    RemoveTempFile(path);
}

QASR_TEST(MappedFileMoveSemantics) {
    const std::string data = "test data for move";
    const std::string path = CreateTempFile("move.bin", data.data(), data.size());
    qasr::MappedFile file;
    qasr::MappedFile::Open(path, &file);
    QASR_EXPECT(file.is_open());

    qasr::MappedFile moved = std::move(file);
    QASR_EXPECT(moved.is_open());
    QASR_EXPECT(!file.is_open());
    QASR_EXPECT_EQ(moved.size(), data.size());
    RemoveTempFile(path);
}

// --- Error: null output ---

QASR_TEST(MappedFileOpenNullOut) {
    qasr::Status s = qasr::MappedFile::Open("/tmp/test", nullptr);
    QASR_EXPECT(!s.ok());
}

// --- ShardRegistry ---

QASR_TEST(ShardRegistryOpenNonexistentDir) {
    qasr::ShardRegistry reg;
    qasr::Status s = qasr::ShardRegistry::Open("/tmp/qasr_nonexistent_dir_12345", &reg);
    QASR_EXPECT(!s.ok());
}

// --- LoadTensorView ---

QASR_TEST(LoadTensorViewNotFound) {
    qasr::ShardRegistry reg;
    qasr::TensorView view;
    qasr::Status s = qasr::LoadTensorView(reg, "nonexistent.weight", &view);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(LoadTensorViewNullOut) {
    qasr::ShardRegistry reg;
    qasr::Status s = qasr::LoadTensorView(reg, "test", nullptr);
    QASR_EXPECT(!s.ok());
}

// --- ValidateShardChecksums ---

QASR_TEST(ValidateShardChecksumsNonexistentDir) {
    qasr::Status s = qasr::ValidateShardChecksums("/tmp/qasr_missing_dir_12345");
    QASR_EXPECT(!s.ok());
}
