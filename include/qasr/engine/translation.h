#pragma once

#include "qasr/core/status.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace qasr {

/* Translation job for the translation queue */
struct TranslationJob {
    std::uint64_t session_id;
    std::uint64_t segment_id;
    std::string source_text;
    std::string source_lang;  /* zh, en, ja, de */
    std::string target_lang;  /* zh, en, ja, de */
    int priority = 0;
};

/* Translation result */
struct TranslationResult {
    std::uint64_t session_id;
    std::uint64_t segment_id;
    Status status;
    std::string translated_text;
    double latency_ms = 0.0;
};

using TranslationCompleteCallback = std::function<void(const TranslationResult &)>;

/* OpenAI-compatible API client for translation LLM */
class TranslationClient {
public:
    TranslationClient();
    ~TranslationClient();

    /* Configure the remote LLM endpoint */
    void SetEndpoint(const std::string & url);
    void SetModel(const std::string & model);
    void SetApiKey(const std::string & key);
    void SetDisableThinking(bool disable);

    /* Translate a single segment */
    Status Translate(const TranslationJob & job,
                     std::string & out_text,
                     double & out_latency_ms);

    /* Async translate with callback */
    Status TranslateAsync(const TranslationJob & job,
                          TranslationCompleteCallback cb);

    /* Health check */
    Status HealthCheck();

private:
    std::string endpoint_;
    std::string model_;
    std::string api_key_;
    bool disable_thinking_ = true;
};

/* Translation queue with backpressure */
class TranslationQueue {
public:
    TranslationQueue();
    ~TranslationQueue();

    Status Submit(const TranslationJob & job);
    void SetCallback(TranslationCompleteCallback cb);
    void SetWorker(int worker_count);
    void Start();
    void Shutdown();

    int queue_depth() const;
    int max_queue_size() const { return max_queue_size_; }

    /* Access the embedded translation client for configuration */
    TranslationClient & client();

private:
    void WorkerLoop();

    TranslationClient client_;
    int max_queue_size_ = 32;
    std::vector<TranslationJob> queue_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> running_{false};
    std::vector<std::thread> workers_;
    TranslationCompleteCallback on_complete_;
};

}  // namespace qasr
