#pragma once

#include <cstddef>
#include <string_view>

namespace qasr {
namespace base {

/// UTF-8 utilities shared across modules.
///
/// Why this lives in `base/`:
///   - `runtime/model_bridge.cc` and `service/realtime.cc` previously
///     each defined their own copy of `IsUtf8Continuation` and
///     `CountUtf8Codepoints` (byte-for-byte identical).  Drift between
///     those copies would have produced silently inconsistent text
///     boundary handling, so we keep a single implementation here.
///
/// Pre/Post: no preconditions.  All functions are pure and read-only.

/// Returns true iff `byte` is a UTF-8 continuation byte (0b10xxxxxx).
bool IsUtf8Continuation(unsigned char byte) noexcept;

/// Counts the number of Unicode code points in `text` by counting
/// every byte that is *not* a UTF-8 continuation byte.  Invalid UTF-8
/// sequences are not rejected; the count matches what a permissive
/// UTF-8 decoder would produce.
std::size_t CountUtf8Codepoints(std::string_view text) noexcept;

}  // namespace base
}  // namespace qasr
