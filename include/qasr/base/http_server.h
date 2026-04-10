#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace qasr {

/// Multipart form-data part.
struct MultipartFormData {
    std::string name;
    std::string filename;
    std::string content;
    std::string content_type;
};

/// HTTP request.
/// Pre: fields are populated by the server before calling the handler.
/// Post: read-only inside the handler.
/// Thread-safe: not shared across threads.
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
    std::unordered_map<std::string, MultipartFormData> files;
    std::unordered_map<std::string, std::string> path_params;

    bool has_param(const std::string & name) const;
    std::string get_param_value(const std::string & name, std::size_t index = 0) const;
    std::size_t get_param_value_count(const std::string & name) const;

    std::unordered_map<std::string, std::vector<std::string>> params_;
    std::unordered_map<std::string, std::string> headers_;
};

/// HTTP response.
/// Pre: status defaults to 200.  Handler sets content and headers.
/// Post: server sends the built response to the client.
/// Thread-safe: not shared across threads.
struct HttpResponse {
    int status = 200;
    void set_content(const std::string & body, const std::string & content_type);
    void set_header(const std::string & name, const std::string & value);

    std::string body_;
    std::string content_type_;
    std::vector<std::pair<std::string, std::string>> headers_;
};

/// Minimal self-owned HTTP/1.1 server.
/// Pre: call Get/Post to register routes before listen().
/// Post: listen() blocks until the server is stopped.
/// Thread-safe: route registration is not thread-safe; listen/stop are.
class HttpServer {
public:
    using Handler = std::function<void(const HttpRequest &, HttpResponse &)>;

    HttpServer();
    ~HttpServer();

    HttpServer(const HttpServer &) = delete;
    HttpServer & operator=(const HttpServer &) = delete;

    void Get(const std::string & pattern, Handler handler);
    void Post(const std::string & pattern, Handler handler);

    void set_thread_pool_size(std::size_t workers, std::size_t queue_limit);
    void set_keep_alive_max_count(int count);
    void set_keep_alive_timeout(int seconds);
    void set_read_timeout(int seconds, int microseconds);
    void set_write_timeout(int seconds, int microseconds);
    void set_idle_interval(int seconds, int microseconds);
    void set_payload_max_length(std::size_t length);

    bool listen(const std::string & host, int port);
    void stop();

    /// Parse helpers exposed for testing.
    static bool ParseQueryString(
        const std::string & query,
        std::unordered_map<std::string, std::vector<std::string>> & params);

    static bool ParseMultipartBody(
        const std::string & boundary,
        const std::string & body,
        std::unordered_map<std::string, MultipartFormData> & parts);

    static bool MatchRoute(
        const std::string & pattern,
        const std::string & path,
        std::unordered_map<std::string, std::string> & path_params);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qasr
