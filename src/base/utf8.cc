#include "qasr/base/utf8.h"

namespace qasr {
namespace base {

bool IsUtf8Continuation(unsigned char byte) noexcept {
    return (byte & 0xC0U) == 0x80U;
}

std::size_t CountUtf8Codepoints(std::string_view text) noexcept {
    std::size_t count = 0;
    for (const char ch : text) {
        if (!IsUtf8Continuation(static_cast<unsigned char>(ch))) {
            ++count;
        }
    }
    return count;
}

}  // namespace base
}  // namespace qasr
