#pragma once

#include <filesystem>
#include <string>
#include <system_error>

namespace qasr_test {

inline std::filesystem::path TempDir(const char * source_file) {
    const std::filesystem::path source(source_file);
    const std::filesystem::path dir = source.parent_path() / "tmp";
    std::filesystem::create_directories(dir);
    return dir;
}

inline std::filesystem::path TempPath(const char * source_file, const std::string & name) {
    return TempDir(source_file) / name;
}

inline std::filesystem::path FreshTempDir(const char * source_file, const std::string & name) {
    const std::filesystem::path dir = TempPath(source_file, name);
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir);
    return dir;
}

inline std::filesystem::path MissingTempPath(const char * source_file, const std::string & name) {
    const std::filesystem::path path = TempPath(source_file, name);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return path;
}

}  // namespace qasr_test
