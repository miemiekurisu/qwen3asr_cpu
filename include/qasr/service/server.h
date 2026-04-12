#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "qasr/core/status.h"
#include "qasr/service/realtime.h"

namespace qasr {

enum class TranscriptionResponseFormat {
    kJson = 0,
    kText,
    kVerboseJson,
};

enum class OpenAiRealtimeAction {
    kSessionCreate = 0,
    kInputAudioBufferAppend,
    kInputAudioBufferCommit,
};

struct OpenAiRealtimeRequest {
    OpenAiRealtimeAction action = OpenAiRealtimeAction::kSessionCreate;
    std::string session_id;
    std::string model;
    std::string language;
    std::string input_audio_format = "pcm16le";
    std::string audio;
    bool stream = true;
};

struct ServerConfig {
    std::string model_dir;
    std::string host = "127.0.0.1";
    std::string ui_dir = "ui";
    std::int32_t port = 8080;
    std::int32_t threads = 0;
    std::int32_t verbosity = 0;
    bool decoder_int8 = false;
    bool encoder_int8 = false;
};

Status ParseBooleanText(std::string_view field_name, std::string_view text, bool * value);
Status ParseTranscriptionResponseFormat(
    std::string_view text,
    TranscriptionResponseFormat * format);
Status ValidateTimestampGranularities(bool want_segment_timestamps, bool want_word_timestamps);
std::string ResolveServedModelId(std::string_view model_dir);
bool IsTerminalJobState(std::string_view state) noexcept;
bool ShouldEvictCompletedJob(
    std::string_view state,
    std::int64_t updated_at_seconds,
    std::int64_t now_seconds,
    std::int64_t ttl_seconds) noexcept;
Status ParseOpenAiRealtimeRequest(std::string_view body, OpenAiRealtimeRequest * request);
Status DecodeBase64Pcm16Le(std::string_view encoded, std::vector<float> * samples);
float RealtimeStreamChunkSeconds(const RealtimePolicyConfig & policy) noexcept;
int RealtimeStreamMaxNewTokens(const RealtimePolicyConfig & policy) noexcept;

Status ValidateServerConfig(const ServerConfig & config);
Status ParseServerArguments(int argc, const char * const argv[], ServerConfig * config, bool * show_help);
std::string BuildServerUsage(std::string_view program_name);
int RunServer(const ServerConfig & config);

}  // namespace qasr
