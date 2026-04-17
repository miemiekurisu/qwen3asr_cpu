#include "tests/test_registry.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>

#ifdef _WIN32
#  include <crtdbg.h>
#  include <cstdlib>
#endif

namespace qasr::test {

std::vector<TestCase> & Registry() {
    static std::vector<TestCase> registry;
    return registry;
}

TestRegistrar::TestRegistrar(std::string name, TestFunction function) {
    Registry().push_back(TestCase{std::move(name), std::move(function)});
}

}  // namespace qasr::test

static std::vector<std::string> ParseExcludeList(const char *arg) {
    std::vector<std::string> result;
    std::string s(arg);
    std::string::size_type pos = 0;
    while (pos < s.size()) {
        auto comma = s.find(',', pos);
        if (comma == std::string::npos) comma = s.size();
        if (comma > pos) result.push_back(s.substr(pos, comma - pos));
        pos = comma + 1;
    }
    return result;
}

int main(int argc, char *argv[]) {
#ifdef _WIN32
    // Redirect debug assertions to stderr and suppress modal dialogs.
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif

    std::vector<std::string> excludes;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--exclude") == 0 && i + 1 < argc) {
            excludes = ParseExcludeList(argv[++i]);
        }
    }

    int failed = 0;
    int skipped = 0;
    for (const auto & test_case : qasr::test::Registry()) {
        if (std::find(excludes.begin(), excludes.end(), test_case.name) != excludes.end()) {
            std::cout << "[SKIP] " << test_case.name << "\n";
            ++skipped;
            continue;
        }
        try {
            test_case.function();
            std::cout << "[PASS] " << test_case.name << "\n";
        } catch (const std::exception & ex) {
            ++failed;
            std::cout << "[FAIL] " << test_case.name << " :: " << ex.what() << "\n";
        }
    }

    auto total = qasr::test::Registry().size();
    if (failed != 0) {
        std::cout << failed << " test(s) failed";
        if (skipped > 0) std::cout << ", " << skipped << " skipped";
        std::cout << "\n";
        return 1;
    }

    std::cout << "all tests passed: " << total - static_cast<std::size_t>(skipped);
    if (skipped > 0) std::cout << " (" << skipped << " skipped)";
    std::cout << "\n";
    return 0;
}
