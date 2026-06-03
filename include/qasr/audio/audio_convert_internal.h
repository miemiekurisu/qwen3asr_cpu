#pragma once

#include <string>
#include <vector>

namespace qasr {

/// Build the argv that ConvertToWav passes to ffmpeg.  Exposed for
/// unit testing.  Each element of the returned vector is a separate
/// argument to the spawned ffmpeg process; nothing here is parsed by
/// a shell, so shell metacharacters in `input_path` or
/// `output_wav_path` are preserved verbatim and cannot be interpreted
/// as commands.
std::vector<std::string> BuildFfmpegArgv(const std::string & input_path,
                                         const std::string & output_wav_path);

}  // namespace qasr
