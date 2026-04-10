#include "tests/test_registry.h"

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

int main() {
#ifdef _WIN32
    // Redirect debug assertions to stderr and suppress modal dialogs.
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
    int failed = 0;
    for (const auto & test_case : qasr::test::Registry()) {
        try {
            test_case.function();
            std::cout << "[PASS] " << test_case.name << "\n";
        } catch (const std::exception & ex) {
            ++failed;
            std::cout << "[FAIL] " << test_case.name << " :: " << ex.what() << "\n";
        }
    }

    if (failed != 0) {
        std::cout << failed << " test(s) failed\n";
        return 1;
    }

    std::cout << "all tests passed: " << qasr::test::Registry().size() << "\n";
    return 0;
}
