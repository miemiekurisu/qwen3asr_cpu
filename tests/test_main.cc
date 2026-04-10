#include "tests/test_registry.h"

#include <exception>
#include <iostream>

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
