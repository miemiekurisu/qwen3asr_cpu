#include "qasr/audio/audio_convert.h"
#include "qasr/cli/options.h"
#include "qasr/inference/aligner_client.h"
#include "qasr/inference/aligner_types.h"
#include "qasr/runtime/blas.h"
#include "qasr/runtime/model_bridge.h"
#include "qasr/subtitle/subtitle_writer.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

int main(int argc, char ** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    {
        const qasr::Status blas_status = qasr::CheckBlasAvailable();
        if (!blas_status.ok()) {
            std::cerr << "error: " << blas_status.ToString() << "\n";
            return 1;
        }
    }

    qasr::CliOptions options;
    const qasr::Status parse_status = qasr::ParseCliArguments(argc, argv, &options);
    if (!parse_status.ok()) {
        std::cerr << parse_status.ToString() << "\n";
        std::cerr << qasr::BuildCliUsage(argc > 0 ? argv[0] : "qasr_cli");
        return 1;
    }

    if (options.show_help) {
        std::cout << qasr::BuildCliUsage(argv[0]);
        return 0;
    }

    // Determine output format
    qasr::OutputFormat out_fmt = qasr::OutputFormat::kText;
    if (!options.output_format.empty()) {
        const qasr::Status fmt_status = qasr::ParseOutputFormat(options.output_format, &out_fmt);
        if (!fmt_status.ok()) {
            std::cerr << fmt_status.ToString() << "\n";
            return 1;
        }
    }

    // If non-WAV input, convert via ffmpeg first
    std::string effective_audio = options.asr.audio_path;
    const std::string original_audio = effective_audio;
    std::string temp_wav;
    if (!qasr::IsWavFile(effective_audio)) {
        std::cerr << "Converting audio to WAV (ffmpeg) ...\n";
        const fs::path src(effective_audio);
        temp_wav = (fs::temp_directory_path() / (src.stem().string() + "_qasr_cli.wav")).string();
        const qasr::Status conv = qasr::ConvertToWav(effective_audio, temp_wav);
        if (!conv.ok()) {
            std::cerr << conv.ToString() << "\n";
            return 1;
        }
        effective_audio = temp_wav;
        options.asr.audio_path = effective_audio;
    }

    // Subtitle formats need segmented transcription
    const bool need_segments = (out_fmt == qasr::OutputFormat::kSrt ||
                                out_fmt == qasr::OutputFormat::kVtt ||
                                out_fmt == qasr::OutputFormat::kJson);

    // Determine output stream early (needed for incremental write)
    std::ofstream file_out;
    std::ostream * out_stream = &std::cout;
    std::string output_path = options.output_path;

    if (output_path.empty() && need_segments) {
        const fs::path audio_p(original_audio);
        output_path = (audio_p.parent_path() /
                       (audio_p.stem().string() + "." + qasr::OutputFormatExtension(out_fmt))).string();
    }

    if (!output_path.empty()) {
        file_out.open(output_path, std::ios::out | std::ios::trunc);
        if (!file_out) {
            std::cerr << "failed to open output file: " << output_path << "\n";
            if (!temp_wav.empty()) { std::error_code ec; fs::remove(temp_wav, ec); }
            return 1;
        }
        out_stream = &file_out;
    }

    std::cerr << "Loading model from " << options.asr.model_dir << " ...\n";

    qasr::AsrRunResult result;
    bool align_srt_written = false;

    if (options.align && need_segments) {
        // ---- Per-segment streaming alignment path ----
        // Pre-check: language supported by aligner?
        const std::string lang = options.asr.language;
        if (!lang.empty() && !qasr::IsAlignerLanguageSupported(lang)) {
            std::cerr << "warning: language \"" << lang
                      << "\" not supported by ForcedAligner; falling back to segment timestamps\n";
            result = qasr::RunAsrSegmented(options.asr);
        } else {
            // 1. Load audio samples once (needed for per-segment slicing).
            std::vector<float> full_samples;
            std::int64_t audio_dur_ms = 0;
            {
                qasr::Status ls = qasr::LoadAudioFile(effective_audio, &full_samples, &audio_dur_ms);
                if (!ls.ok()) {
                    std::cerr << "audio load failed: " << ls.ToString() << "\n";
                    if (!temp_wav.empty()) { std::error_code ec; fs::remove(temp_wav, ec); }
                    return 1;
                }
            }
            constexpr int kSampleRate = 16000;

            // 2. Load ForcedAligner model.
            qasr::AlignerConfig acfg;
            acfg.model_dir = options.aligner_model_dir;

            qasr::ForcedAligner aligner;
            qasr::Status as = aligner.Load(acfg);
            if (!as.ok()) {
                std::cerr << "aligner load failed: " << as.ToString() << "\n";
                if (!temp_wav.empty()) { std::error_code ec; fs::remove(temp_wav, ec); }
                return 1;
            }

            // 3. Prepare incremental SRT/VTT header.
            const bool write_srt = (out_fmt == qasr::OutputFormat::kSrt);
            const bool write_vtt = (out_fmt == qasr::OutputFormat::kVtt);
            if (write_vtt) {
                qasr::WriteVttHeader(*out_stream);
                out_stream->flush();
            }

            int cue_index = 0;
            qasr::SubtitlePolicy policy{};
            qasr::Status write_err;

            // Collect aligned segments separately — the trampoline inside
            // RunAsrSegmentedStreaming already populates result.segments with
            // raw (unaligned) segments, so we must replace them afterward.
            std::vector<qasr::TimedSegment> aligned_segments;

            // 4. Stream ASR segments; align + write each immediately.
            result = qasr::RunAsrSegmentedStreaming(options.asr,
                [&](int seg_index, const qasr::TimedSegment & segment) {
                    if (!write_err.ok()) return;

                    // Slice audio for this segment.
                    int begin_sample = static_cast<int>(
                        static_cast<std::int64_t>(segment.range.begin_ms) * kSampleRate / 1000);
                    int end_sample = static_cast<int>(
                        static_cast<std::int64_t>(segment.range.end_ms) * kSampleRate / 1000);
                    if (begin_sample < 0) begin_sample = 0;
                    if (end_sample > static_cast<int>(full_samples.size()))
                        end_sample = static_cast<int>(full_samples.size());
                    int n_seg_samples = end_sample - begin_sample;

                    if (n_seg_samples <= 0 || segment.text.empty()) {
                        // Empty segment — still store it for full text
                        return;
                    }

                    // Run forced alignment on this segment's audio + text.
                    qasr::AlignResult align_result;
                    qasr::Status sa = aligner.AlignSamples(
                        full_samples.data() + begin_sample, n_seg_samples,
                        segment.text,
                        lang.empty() ? "chinese" : lang,
                        &align_result);

                    if (!sa.ok()) {
                        std::cerr << "[seg " << seg_index << "] alignment failed: "
                                  << sa.ToString() << ", using segment timestamps\n";
                        // Fallback: use the raw segment timestamps
                        aligned_segments.push_back(segment);
                        if (write_srt || write_vtt) {
                            std::vector<qasr::TimedSegment> one{segment};
                            auto cues = qasr::LayoutSubtitles(one, policy);
                            for (const auto & cue : cues) {
                                ++cue_index;
                                qasr::Status ws = write_srt
                                    ? qasr::WriteSrtCue(cue, cue_index, *out_stream)
                                    : qasr::WriteVttCue(cue, *out_stream);
                                if (!ws.ok()) { write_err = ws; return; }
                                out_stream->flush();
                            }
                        }
                        return;
                    }

                    // Offset word timestamps to absolute time.
                    double offset_sec = segment.range.begin_ms / 1000.0;
                    for (auto & w : align_result.words) {
                        w.start_sec += offset_sec;
                        w.end_sec   += offset_sec;
                    }

                    // Convert to segments and write.
                    auto aligned_segs = qasr::WordsToSegments(align_result.words);
                    std::cerr << "[seg " << seg_index << "] aligned "
                              << align_result.words.size() << " words -> "
                              << aligned_segs.size() << " cues ("
                              << segment.range.begin_ms << "-"
                              << segment.range.end_ms << " ms)\n";

                    // Accumulate for full text / perf stats
                    for (auto & s : aligned_segs) {
                        aligned_segments.push_back(s);
                    }

                    if (write_srt || write_vtt) {
                        auto cues = qasr::LayoutSubtitles(aligned_segs, policy);
                        for (const auto & cue : cues) {
                            ++cue_index;
                            qasr::Status ws = write_srt
                                ? qasr::WriteSrtCue(cue, cue_index, *out_stream)
                                : qasr::WriteVttCue(cue, *out_stream);
                            if (!ws.ok()) { write_err = ws; return; }
                            out_stream->flush();
                        }
                    }
                });

            aligner.Unload();

            // Replace the raw segments (populated by the trampoline) with
            // the precisely aligned segments collected in the callback.
            result.segments = std::move(aligned_segments);

            if (!write_err.ok()) {
                std::cerr << write_err.ToString() << "\n";
                if (!temp_wav.empty()) { std::error_code ec; fs::remove(temp_wav, ec); }
                return 1;
            }

            if (write_srt || write_vtt) {
                align_srt_written = true;
            }

            // Build full text
            std::string full_text;
            for (const auto & seg : result.segments) {
                if (!full_text.empty()) full_text += ' ';
                full_text += seg.text;
            }
            result.text = std::move(full_text);
        }
    } else if (need_segments && out_fmt != qasr::OutputFormat::kJson) {
        // Incremental SRT/VTT: write each cue as it arrives
        if (out_fmt == qasr::OutputFormat::kVtt) {
            qasr::WriteVttHeader(*out_stream);
            out_stream->flush();
        }

        int cue_index = 0;
        qasr::SubtitlePolicy policy{};
        qasr::Status write_err;

        result = qasr::RunAsrSegmentedStreaming(options.asr,
            [&](int /*seg_index*/, const qasr::TimedSegment & segment) {
                if (!write_err.ok()) return;
                std::vector<qasr::TimedSegment> one{segment};
                auto cues = qasr::LayoutSubtitles(one, policy);
                for (const auto & cue : cues) {
                    ++cue_index;
                    qasr::Status s;
                    if (out_fmt == qasr::OutputFormat::kSrt) {
                        s = qasr::WriteSrtCue(cue, cue_index, *out_stream);
                    } else {
                        s = qasr::WriteVttCue(cue, *out_stream);
                    }
                    if (!s.ok()) { write_err = s; return; }
                    out_stream->flush();
                    std::cerr << "[seg " << cue_index << "] "
                              << segment.range.begin_ms << "ms - "
                              << segment.range.end_ms << "ms\n";
                }
            });

        if (!write_err.ok()) {
            std::cerr << write_err.ToString() << "\n";
            // Clean up temp WAV
            if (!temp_wav.empty()) { std::error_code ec; fs::remove(temp_wav, ec); }
            return 1;
        }
    } else if (need_segments) {
        // JSON: must accumulate all segments first (valid JSON structure)
        result = qasr::RunAsrSegmented(options.asr);
    } else {
        // Text mode: stream each segment to console as it finishes
        result = qasr::RunAsrSegmentedStreaming(options.asr,
            [&](int /*seg_index*/, const qasr::TimedSegment & segment) {
                if (!segment.text.empty()) {
                    *out_stream << segment.text << "\n";
                    out_stream->flush();
                }
            });
    }

    // Clean up temp WAV
    if (!temp_wav.empty()) {
        std::error_code ec;
        fs::remove(temp_wav, ec);
    }

    if (!result.status.ok()) {
        std::cerr << result.status.ToString() << "\n";
        return 1;
    }

    // Write output for non-incremental paths
    qasr::Status write_status = qasr::OkStatus();
    if (out_fmt == qasr::OutputFormat::kJson) {
        const double dur_sec = result.audio_ms / 1000.0;
        write_status = qasr::WriteSegmentJson(result.segments, dur_sec, *out_stream);
    } else if (out_fmt == qasr::OutputFormat::kSrt && options.align && !align_srt_written) {
        // Aligned path: batch-write SRT from precise segments (fallback)
        qasr::SubtitlePolicy policy{};
        auto cues = qasr::LayoutSubtitles(result.segments, policy);
        write_status = qasr::WriteSrt(cues, *out_stream);
    } else if (out_fmt == qasr::OutputFormat::kVtt && options.align && !align_srt_written) {
        // Aligned path: batch-write VTT from precise segments (fallback)
        qasr::SubtitlePolicy policy{};
        auto cues = qasr::LayoutSubtitles(result.segments, policy);
        write_status = qasr::WriteVtt(cues, *out_stream);
    } else if (out_fmt == qasr::OutputFormat::kText) {
        // Already streamed incrementally above.
    }
    // SRT/VTT without --align already written incrementally above

    if (!write_status.ok()) {
        std::cerr << write_status.ToString() << "\n";
        return 1;
    }

    if (!output_path.empty()) {
        std::cerr << "output: " << output_path << "\n";
    }

    std::cerr << "inference_ms=" << result.total_ms
              << " audio_ms=" << result.audio_ms
              << " tokens=" << result.text_tokens
              << " encode_ms=" << result.encode_ms
              << " decode_ms=" << result.decode_ms << "\n";
    return 0;
}
