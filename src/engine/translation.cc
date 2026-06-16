#include "qasr/engine/translation.h"
#include "qasr/base/json.h"
#include <cstdio>
#include <cstring>
#include <sstream>
#include <chrono>

#ifdef QASR_CURL_AVAILABLE
#include <curl/curl.h>
#endif

namespace qasr {

/* libcurl-based HTTP client */
class CurlHttpClient {
public:
    static Status Post(const std::string & url,
                       const std::string & body,
                       const std::string & auth_header,
                       std::string & response,
                       long & http_code,
                       double & latency_ms) {
        CURL * curl = curl_easy_init();
        if (!curl) {
            return Status(StatusCode::kInternal, "curl_easy_init() failed");
        }

        auto t0 = std::chrono::steady_clock::now();

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)body.size());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, +[](char *ptr, size_t size, size_t nmemb, void *userdata) {
            auto *buf = static_cast<std::string *>(userdata);
            buf->append(ptr, size * nmemb);
            return size * nmemb;
        });
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

        /* Headers */
        auto * headers = curl_slist_append(nullptr, "Content-Type: application/json");
        headers = curl_slist_append(headers, "Accept: application/json");
        if (!auth_header.empty()) {
            headers = curl_slist_append(headers, ("Authorization: " + auth_header).c_str());
        }
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            curl_slist_free_all(headers);
            curl_easy_cleanup(curl);
            return Status(StatusCode::kInternal,
                          std::string("curl_easy_perform failed: ") + curl_easy_strerror(res));
        }

        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        auto t1 = std::chrono::steady_clock::now();
        latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        return OkStatus();
    }
};

/* --- TranslationClient --- */

TranslationClient::TranslationClient() : disable_thinking_(true) {}

TranslationClient::~TranslationClient() {}

void TranslationClient::SetEndpoint(const std::string & url) {
    endpoint_ = url;
}

void TranslationClient::SetModel(const std::string & model) {
    model_ = model;
}

void TranslationClient::SetApiKey(const std::string & key) {
    api_key_ = key;
}

void TranslationClient::SetDisableThinking(bool disable) {
    disable_thinking_ = disable;
}

Status TranslationClient::Translate(const TranslationJob & job,
                                     std::string & out_text,
                                     double & out_latency_ms) {
    if (endpoint_.empty()) {
        return Status(StatusCode::kFailedPrecondition,
                      "translation endpoint not configured");
    }

    std::string full_url = endpoint_;
    if (!full_url.empty() && full_url.back() == '/') {
        full_url.pop_back();
    }
    full_url += "/v1/chat/completions";

    Json body;
    body["model"] = model_.empty() ? "qwen3-8b" : model_;
    body["temperature"] = 0.1;
    body["max_tokens"] = 2048;

    if (disable_thinking_) {
        Json kwargs;
        kwargs["enable_thinking"] = false;
        body["extra_body"] = Json::object();
        body["extra_body"]["chat_template_kwargs"] = kwargs;
    }

    Json messages;
    Json system_msg;
    system_msg["role"] = "system";
    system_msg["content"] =
        "You are a professional translator. Translate the following text from "
        + job.source_lang + " to " + job.target_lang +
        ". Output ONLY the translation, nothing else.";
    messages.push_back(system_msg);

    Json user_msg;
    user_msg["role"] = "user";
    user_msg["content"] = job.source_text;
    messages.push_back(user_msg);

    body["messages"] = messages;

    std::string json_body = body.dump();

    std::string response;
    long http_code = 0;

    std::string auth_header;
    if (!api_key_.empty()) {
        auth_header = "Bearer " + api_key_;
    }

    auto status = CurlHttpClient::Post(full_url, json_body, auth_header,
                                        response, http_code, out_latency_ms);
    if (!status.ok()) return status;

    if (http_code != 200) {
        return Status(StatusCode::kInternal,
                      "translation API returned HTTP " + std::to_string(http_code) +
                      ": " + response.substr(0, 200));
    }

    Json resp = Json::parse(response);
    if (resp.is_discarded()) {
        return Status(StatusCode::kInternal, "invalid JSON from translation API");
    }

    auto choices = resp["choices"];
    if (choices.is_array() && choices.size() > 0) {
        out_text = choices[0]["message"]["content"].get<std::string>();
    } else {
        return Status(StatusCode::kInternal, "no translation content in response");
    }

    return OkStatus();
}

Status TranslationClient::TranslateAsync(const TranslationJob & job,
                                          TranslationCompleteCallback cb) {
    std::string text;
    double latency;
    auto status = Translate(job, text, latency);

    TranslationResult result;
    result.session_id = job.session_id;
    result.segment_id = job.segment_id;
    result.status = status;
    result.translated_text = text;
    result.latency_ms = latency;

    if (cb) cb(result);
    return OkStatus();
}

Status TranslationClient::HealthCheck() {
    if (endpoint_.empty()) {
        return Status(StatusCode::kFailedPrecondition,
                      "translation endpoint not configured");
    }
    return OkStatus();
}

/* --- TranslationQueue --- */

TranslationQueue::TranslationQueue() {}

TranslationQueue::~TranslationQueue() {
    Shutdown();
}

Status TranslationQueue::Submit(const TranslationJob & job) {
    if (shutdown_.load()) {
        return Status(StatusCode::kFailedPrecondition, "translation queue shut down");
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (static_cast<int>(queue_.size()) >= max_queue_size_) {
            return Status(StatusCode::kResourceExhausted, "translation queue full");
        }
        queue_.push_back(job);
    }
    cv_.notify_one();
    return OkStatus();
}

void TranslationQueue::SetCallback(TranslationCompleteCallback cb) {
    on_complete_ = std::move(cb);
}

void TranslationQueue::SetWorker(int worker_count) {
    (void)worker_count;
}

void TranslationQueue::Start() {
    if (running_.exchange(true)) return;
    workers_.emplace_back([this]() { WorkerLoop(); });
}

void TranslationQueue::Shutdown() {
    if (!running_.exchange(false)) return;
    shutdown_.store(true);
    cv_.notify_all();
    for (auto & w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

int TranslationQueue::queue_depth() const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(queue_.size());
}

TranslationClient & TranslationQueue::client() {
    return client_;
}

void TranslationQueue::WorkerLoop() {
    while (!shutdown_.load()) {
        TranslationJob job;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait_for(lock, std::chrono::milliseconds(500), [this]() {
                return shutdown_.load() || !queue_.empty();
            });
            if (shutdown_.load()) break;
            if (queue_.empty()) continue;

            job = queue_.front();
            queue_.erase(queue_.begin());
        }

        TranslationResult result;
        result.session_id = job.session_id;
        result.segment_id = job.segment_id;

        std::string text;
        double latency;
        auto status = client_.Translate(job, text, latency);
        result.status = status;
        result.translated_text = text;
        result.latency_ms = latency;

        if (on_complete_) {
            on_complete_(result);
        }
    }
}

}  // namespace qasr
