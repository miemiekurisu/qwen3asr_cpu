#pragma once

#include <set>
#include <string>

namespace qasr {

/// Scan a safetensors index file and return the unique shard names
/// referenced in it.  A shard name is a substring that starts with
/// `model-` and ends with `.safetensors` (matching the original
/// `model-[^"]+\.safetensors` regex used in earlier versions of this
/// project).  The scan is performed in O(n) over the input without
/// instantiating `std::regex`.
///
/// This helper is exposed for unit testing; production code should
/// call `ValidateModelDirectory`, which uses the same scan
/// internally.
std::set<std::string> ExtractIndexedSafetensors(const std::string & json_text);

}  // namespace qasr
