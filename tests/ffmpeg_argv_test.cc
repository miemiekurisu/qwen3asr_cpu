#include "tests/test_registry.h"

#include <string>
#include <vector>

#include "qasr/audio/audio_convert_internal.h"

namespace {

QASR_TEST(BuildFfmpegArgvReturnsExpectedArgv) {
    const std::vector<std::string> args = qasr::BuildFfmpegArgv(
        "in.mp3", "out.wav");
    // Expected: ffmpeg -y -loglevel error -i in.mp3 -ar 16000 -ac 1
    //           -c:a pcm_s16le out.wav
    const std::vector<std::string> expected = {
        "ffmpeg", "-y", "-loglevel", "error", "-i", "in.mp3",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "out.wav",
    };
    QASR_EXPECT_EQ(args.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        QASR_EXPECT_EQ(args[i], expected[i]);
    }
}

QASR_TEST(BuildFfmpegArgvPreservesShellMetacharsInInput) {
    // The input path contains every dangerous shell metacharacter.
    // It must be preserved verbatim as a single argv element.
    const std::string evil = ";rm -rf / & echo INJECT `whoami` $(date) | cat > /tmp/x #";
    const std::vector<std::string> args =
        qasr::BuildFfmpegArgv(evil, "out.wav");
    // The input path is at argv[5].
    QASR_EXPECT_EQ(args.size(), std::size_t{13});
    QASR_EXPECT_EQ(args[5], evil);
    // Defensive: no arg should be an empty string.
    for (const auto & a : args) {
        QASR_EXPECT(!a.empty());
    }
}

QASR_TEST(BuildFfmpegArgvPreservesShellMetacharsInOutput) {
    const std::string evil = "out;rm -rf / #.wav";
    const std::vector<std::string> args =
        qasr::BuildFfmpegArgv("in.mp3", evil);
    // The output path is the final argv element.
    QASR_EXPECT_EQ(args.back(), evil);
}

QASR_TEST(BuildFfmpegArgvPreservesQuoteChars) {
    // Double-quote and backslash characters must NOT be re-interpreted
    // as shell metacharacters (they are passed verbatim, so they
    // would only matter to ffmpeg, not to a shell).
    const std::string evil_input = "a\"b\\c$d.wav";
    const std::vector<std::string> args =
        qasr::BuildFfmpegArgv(evil_input, "out.wav");
    QASR_EXPECT_EQ(args[5], evil_input);
}

QASR_TEST(BuildFfmpegArgvNoShellJoining) {
    // The argument list should contain no joined "ffmpeg -y" type
    // strings, and no command-line-style quoting.
    const std::vector<std::string> args =
        qasr::BuildFfmpegArgv("in.mp3", "out.wav");
    for (const auto & a : args) {
        // No element should contain a space — each must be a single
        // argv entry.
        QASR_EXPECT(a.find(' ') == std::string::npos);
    }
}

QASR_TEST(BuildFfmpegArgvProgramIsFfmpeg) {
    const std::vector<std::string> args =
        qasr::BuildFfmpegArgv("in.mp3", "out.wav");
    QASR_EXPECT_EQ(args[0], std::string{"ffmpeg"});
}

}  // namespace
