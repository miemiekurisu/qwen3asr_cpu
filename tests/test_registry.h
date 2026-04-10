#pragma once

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qasr::test {

using TestFunction = std::function<void()>;

struct TestCase {
    std::string name;
    TestFunction function;
};

std::vector<TestCase> & Registry();

struct TestRegistrar {
    TestRegistrar(std::string name, TestFunction function);
};

inline std::string FailureMessage(const char * expr, const char * file, int line) {
    std::ostringstream builder;
    builder << file << ":" << line << ": expectation failed: " << expr;
    return builder.str();
}

template <typename Left, typename Right>
inline std::string EqualityFailureMessage(
    const char * lhs_expr,
    const char * rhs_expr,
    const Left &,
    const Right &,
    const char * file,
    int line) {
    std::ostringstream builder;
    builder << file << ":" << line << ": expectation failed: " << lhs_expr
            << " == " << rhs_expr;
    return builder.str();
}

}  // namespace qasr::test

#define QASR_TEST(name)                                                                  \
    static void name();                                                                  \
    static ::qasr::test::TestRegistrar registrar_##name(#name, &name);                  \
    static void name()

#define QASR_EXPECT(expr)                                                                \
    do {                                                                                 \
        if (!(expr)) {                                                                   \
            throw std::runtime_error(::qasr::test::FailureMessage(#expr, __FILE__, __LINE__)); \
        }                                                                                \
    } while (0)

#define QASR_EXPECT_EQ(lhs, rhs)                                                         \
    do {                                                                                 \
        const auto lhs_value = (lhs);                                                    \
        const auto rhs_value = (rhs);                                                    \
        if (!(lhs_value == rhs_value)) {                                                 \
            throw std::runtime_error(::qasr::test::EqualityFailureMessage(               \
                #lhs, #rhs, lhs_value, rhs_value, __FILE__, __LINE__));                  \
        }                                                                                \
    } while (0)
