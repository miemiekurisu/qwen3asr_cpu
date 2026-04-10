#include "qasr/base/http_server.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace qasr {

// ---------------------------------------------------------------------------
// HttpRequest
// ---------------------------------------------------------------------------

bool HttpRequest::has_param(const std::string & name) const {
    return params_.count(name) > 0;
}

std::string HttpRequest::get_param_value(const std::string & name, std::size_t index) const {
    const auto it = params_.find(name);
    if (it == params_.end() || index >= it->second.size()) return {};
    return it->second[index];
}

std::size_t HttpRequest::get_param_value_count(const std::string & name) const {
    const auto it = params_.find(name);
    return it == params_.end() ? 0 : it->second.size();
}

// ---------------------------------------------------------------------------
// HttpResponse
// ---------------------------------------------------------------------------

void HttpResponse::set_content(const std::string & body, const std::string & content_type) {
    body_ = body;
    content_type_ = content_type;
}

void HttpResponse::set_header(const std::string & name, const std::string & value) {
    for (auto & h : headers_) {
        if (h.first == name) { h.second = value; return; }
    }
    headers_.emplace_back(name, value);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

std::string UrlDecode(const std::string & input) {
    std::string output;
    output.reserve(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '%' && i + 2 < input.size()) {
            const auto hex = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            const int hi = hex(input[i + 1]);
            const int lo = hex(input[i + 2]);
            if (hi >= 0 && lo >= 0) {
                output.push_back(static_cast<char>((hi << 4) | lo));
                i += 2;
                continue;
            }
        }
        if (input[i] == '+') {
            output.push_back(' ');
        } else {
            output.push_back(input[i]);
        }
    }
    return output;
}

std::string StatusText(int code) {
    switch (code) {
        case 200: return "OK";
        case 202: return "Accepted";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 409: return "Conflict";
        case 412: return "Precondition Failed";
        case 413: return "Payload Too Large";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        default:  return "Unknown";
    }
}

std::string ToLower(std::string_view sv) {
    std::string result;
    result.reserve(sv.size());
    for (char c : sv) {
        result.push_back((c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c);
    }
    return result;
}

bool ReadLine(int fd, std::string & line, int timeout_ms) {
    line.clear();
    while (true) {
        struct pollfd pfd{};
        pfd.fd = fd;
        pfd.events = POLLIN;
        const int ret = poll(&pfd, 1, timeout_ms);
        if (ret <= 0) return false;
        char ch = 0;
        const ssize_t n = recv(fd, &ch, 1, 0);
        if (n <= 0) return false;
        if (ch == '\n') {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        }
        line.push_back(ch);
        if (line.size() > 65536) return false;
    }
}

bool ReadExact(int fd, std::string & buffer, std::size_t count, int timeout_ms) {
    buffer.resize(count);
    std::size_t total = 0;
    while (total < count) {
        struct pollfd pfd{};
        pfd.fd = fd;
        pfd.events = POLLIN;
        const int ret = poll(&pfd, 1, timeout_ms);
        if (ret <= 0) return false;
        const ssize_t n = recv(fd, &buffer[total], count - total, 0);
        if (n <= 0) return false;
        total += static_cast<std::size_t>(n);
    }
    return true;
}

bool SendAll(int fd, const std::string & data) {
    std::size_t total = 0;
    while (total < data.size()) {
        const ssize_t n = send(fd, data.data() + total, data.size() - total, MSG_NOSIGNAL);
        if (n <= 0) return false;
        total += static_cast<std::size_t>(n);
    }
    return true;
}

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

struct Route {
    std::string method;
    std::string pattern;
    HttpServer::Handler handler;
};

// Extract the boundary string from Content-Type header.
std::string ExtractBoundary(const std::string & content_type) {
    const std::string marker = "boundary=";
    const auto pos = content_type.find(marker);
    if (pos == std::string::npos) return {};
    std::string boundary = content_type.substr(pos + marker.size());
    if (boundary.size() >= 2 && boundary.front() == '"' && boundary.back() == '"') {
        boundary = boundary.substr(1, boundary.size() - 2);
    }
    return boundary;
}

// Extract Content-Disposition fields: name and filename.
void ParseContentDisposition(
    const std::string & header,
    std::string & name,
    std::string & filename) {
    name.clear();
    filename.clear();

    auto extract = [&](const char * field) -> std::string {
        const std::string tag = std::string(field) + "=\"";
        const auto pos = header.find(tag);
        if (pos == std::string::npos) return {};
        const auto start = pos + tag.size();
        const auto end = header.find('"', start);
        if (end == std::string::npos) return {};
        return header.substr(start, end - start);
    };

    name = extract("name");
    filename = extract("filename");
}

class ThreadPool {
public:
    ThreadPool(std::size_t workers, std::size_t queue_limit)
        : queue_limit_(queue_limit), stop_(false) {
        for (std::size_t i = 0; i < workers; ++i) {
            workers_.emplace_back([this]() { WorkerLoop(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto & t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    bool Submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stop_ || tasks_.size() >= queue_limit_) return false;
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
        return true;
    }

private:
    void WorkerLoop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::size_t queue_limit_;
    bool stop_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> tasks_;
    std::vector<std::thread> workers_;
};

}  // namespace

// ---------------------------------------------------------------------------
// HttpServer::Impl
// ---------------------------------------------------------------------------

struct HttpServer::Impl {
    std::vector<Route> routes;
    std::size_t thread_pool_workers = 4;
    std::size_t thread_pool_queue_limit = 64;
    int keep_alive_max_count = 100;
    int keep_alive_timeout_s = 5;
    int read_timeout_ms = 30000;
    int write_timeout_ms = 30000;
    int idle_interval_ms = 1000;
    std::size_t payload_max_length = 64ULL * 1024ULL * 1024ULL;
    std::atomic<bool> running{false};
    int listen_fd = -1;

    bool MatchAndDispatch(const HttpRequest & req_in, HttpResponse & resp) {
        for (const auto & route : routes) {
            if (route.method != req_in.method) continue;
            std::unordered_map<std::string, std::string> path_params;
            if (HttpServer::MatchRoute(route.pattern, req_in.path, path_params)) {
                HttpRequest req_copy = req_in;
                req_copy.path_params = std::move(path_params);
                route.handler(req_copy, resp);
                return true;
            }
        }
        return false;
    }

    bool ParseRequest(int fd, HttpRequest & request) {
        std::string request_line;
        if (!ReadLine(fd, request_line, read_timeout_ms)) return false;

        // Parse: METHOD /path?query HTTP/1.1
        const auto sp1 = request_line.find(' ');
        if (sp1 == std::string::npos) return false;
        const auto sp2 = request_line.find(' ', sp1 + 1);
        if (sp2 == std::string::npos) return false;

        request.method = request_line.substr(0, sp1);
        std::string full_path = request_line.substr(sp1 + 1, sp2 - sp1 - 1);

        // Split path and query
        const auto qpos = full_path.find('?');
        if (qpos != std::string::npos) {
            request.path = full_path.substr(0, qpos);
            HttpServer::ParseQueryString(full_path.substr(qpos + 1), request.params_);
        } else {
            request.path = full_path;
        }

        // Parse headers
        std::string header_line;
        while (ReadLine(fd, header_line, read_timeout_ms)) {
            if (header_line.empty()) break;
            const auto colon = header_line.find(':');
            if (colon == std::string::npos) continue;
            std::string name = ToLower(header_line.substr(0, colon));
            std::string value = header_line.substr(colon + 1);
            while (!value.empty() && value.front() == ' ') value.erase(value.begin());
            request.headers_[name] = value;
        }

        // Read body
        std::size_t content_length = 0;
        const auto cl_it = request.headers_.find("content-length");
        if (cl_it != request.headers_.end()) {
            content_length = static_cast<std::size_t>(std::strtoull(cl_it->second.c_str(), nullptr, 10));
        }
        if (content_length > payload_max_length) return false;
        if (content_length > 0) {
            if (!ReadExact(fd, request.body, content_length, read_timeout_ms)) return false;
        }

        // Parse multipart if applicable
        const auto ct_it = request.headers_.find("content-type");
        if (ct_it != request.headers_.end()) {
            const std::string ct_lower = ToLower(ct_it->second);
            if (ct_lower.find("multipart/form-data") != std::string::npos) {
                const std::string boundary = ExtractBoundary(ct_it->second);
                if (!boundary.empty()) {
                    HttpServer::ParseMultipartBody(boundary, request.body, request.files);
                }
            }
        }

        return true;
    }

    void HandleConnection(int client_fd) {
        int requests_handled = 0;
        while (running.load() && requests_handled < keep_alive_max_count) {
            HttpRequest request;
            if (!ParseRequest(client_fd, request)) break;

            HttpResponse response;
            if (!MatchAndDispatch(request, response)) {
                response.status = 404;
                response.set_content("{\"error\":\"not found\"}", "application/json");
            }

            // Build HTTP response
            std::string raw_response;
            raw_response += "HTTP/1.1 " + std::to_string(response.status) + " " +
                            StatusText(response.status) + "\r\n";
            raw_response += "Content-Length: " + std::to_string(response.body_.size()) + "\r\n";
            if (!response.content_type_.empty()) {
                raw_response += "Content-Type: " + response.content_type_ + "\r\n";
            }
            for (const auto & h : response.headers_) {
                raw_response += h.first + ": " + h.second + "\r\n";
            }

            // Determine keep-alive
            const auto conn_it = request.headers_.find("connection");
            const bool keep_alive = conn_it != request.headers_.end() &&
                                    ToLower(conn_it->second) == "keep-alive";
            if (keep_alive) {
                raw_response += "Connection: keep-alive\r\n";
            } else {
                raw_response += "Connection: close\r\n";
            }
            raw_response += "\r\n";
            raw_response += response.body_;

            if (!SendAll(client_fd, raw_response)) break;
            ++requests_handled;
            if (!keep_alive) break;
        }
        close(client_fd);
    }
};

// ---------------------------------------------------------------------------
// HttpServer public API
// ---------------------------------------------------------------------------

HttpServer::HttpServer() : impl_(std::make_unique<Impl>()) {}
HttpServer::~HttpServer() { stop(); }

void HttpServer::Get(const std::string & pattern, Handler handler) {
    impl_->routes.push_back({"GET", pattern, std::move(handler)});
}

void HttpServer::Post(const std::string & pattern, Handler handler) {
    impl_->routes.push_back({"POST", pattern, std::move(handler)});
}

void HttpServer::set_thread_pool_size(std::size_t workers, std::size_t queue_limit) {
    impl_->thread_pool_workers = workers;
    impl_->thread_pool_queue_limit = queue_limit;
}

void HttpServer::set_keep_alive_max_count(int count) { impl_->keep_alive_max_count = count; }
void HttpServer::set_keep_alive_timeout(int seconds) { impl_->keep_alive_timeout_s = seconds; }

void HttpServer::set_read_timeout(int seconds, int microseconds) {
    impl_->read_timeout_ms = seconds * 1000 + microseconds / 1000;
}

void HttpServer::set_write_timeout(int seconds, int microseconds) {
    impl_->write_timeout_ms = seconds * 1000 + microseconds / 1000;
}

void HttpServer::set_idle_interval(int seconds, int microseconds) {
    impl_->idle_interval_ms = seconds * 1000 + microseconds / 1000;
}

void HttpServer::set_payload_max_length(std::size_t length) {
    impl_->payload_max_length = length;
}

bool HttpServer::listen(const std::string & host, int port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return false;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<std::uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        close(fd);
        return false;
    }

    if (::bind(fd, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        close(fd);
        return false;
    }

    if (::listen(fd, 128) < 0) {
        close(fd);
        return false;
    }

    impl_->listen_fd = fd;
    impl_->running.store(true);

    ThreadPool pool(impl_->thread_pool_workers, impl_->thread_pool_queue_limit);

    while (impl_->running.load()) {
        struct pollfd pfd{};
        pfd.fd = fd;
        pfd.events = POLLIN;
        const int ret = poll(&pfd, 1, impl_->idle_interval_ms);
        if (ret < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (ret == 0) continue;

        struct sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        const int client_fd = accept(fd, reinterpret_cast<struct sockaddr *>(&client_addr), &client_len);
        if (client_fd < 0) continue;

        // Set timeouts on client socket
        struct timeval tv{};
        tv.tv_sec = impl_->read_timeout_ms / 1000;
        tv.tv_usec = (impl_->read_timeout_ms % 1000) * 1000;
        setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        tv.tv_sec = impl_->write_timeout_ms / 1000;
        tv.tv_usec = (impl_->write_timeout_ms % 1000) * 1000;
        setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

        int nodelay = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

        auto * impl = impl_.get();
        if (!pool.Submit([impl, client_fd]() { impl->HandleConnection(client_fd); })) {
            // Queue full — reject with 503
            const std::string reject =
                "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            send(client_fd, reject.data(), reject.size(), MSG_NOSIGNAL);
            close(client_fd);
        }
    }

    close(fd);
    impl_->listen_fd = -1;
    return true;
}

void HttpServer::stop() {
    impl_->running.store(false);
    if (impl_->listen_fd >= 0) {
        shutdown(impl_->listen_fd, SHUT_RDWR);
        close(impl_->listen_fd);
        impl_->listen_fd = -1;
    }
}

// ---------------------------------------------------------------------------
// Parse helpers (public for testing)
// ---------------------------------------------------------------------------

bool HttpServer::ParseQueryString(
    const std::string & query,
    std::unordered_map<std::string, std::vector<std::string>> & params) {
    if (query.empty()) return true;
    std::size_t pos = 0;
    while (pos < query.size()) {
        auto amp = query.find('&', pos);
        if (amp == std::string::npos) amp = query.size();
        const std::string pair = query.substr(pos, amp - pos);
        pos = amp + 1;
        if (pair.empty()) continue;
        const auto eq = pair.find('=');
        if (eq == std::string::npos) {
            params[UrlDecode(pair)].emplace_back();
        } else {
            params[UrlDecode(pair.substr(0, eq))].push_back(UrlDecode(pair.substr(eq + 1)));
        }
    }
    return true;
}

bool HttpServer::ParseMultipartBody(
    const std::string & boundary,
    const std::string & body,
    std::unordered_map<std::string, MultipartFormData> & parts) {
    const std::string delimiter = "--" + boundary;
    const std::string end_delimiter = delimiter + "--";

    std::size_t pos = body.find(delimiter);
    if (pos == std::string::npos) return false;
    pos += delimiter.size();

    while (pos < body.size()) {
        // Skip CRLF after boundary
        if (pos + 1 < body.size() && body[pos] == '\r' && body[pos + 1] == '\n') {
            pos += 2;
        } else if (pos < body.size() && body[pos] == '\n') {
            pos += 1;
        } else {
            break;
        }

        // Parse part headers
        std::string disposition;
        std::string part_content_type;
        while (pos < body.size()) {
            auto eol = body.find('\n', pos);
            if (eol == std::string::npos) return false;
            std::string line = body.substr(pos, eol - pos);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            pos = eol + 1;
            if (line.empty()) break;
            const auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string hname = ToLower(line.substr(0, colon));
            std::string hval = line.substr(colon + 1);
            while (!hval.empty() && hval.front() == ' ') hval.erase(hval.begin());
            if (hname == "content-disposition") disposition = hval;
            else if (hname == "content-type") part_content_type = hval;
        }

        // Find end of part body
        auto next = body.find(delimiter, pos);
        if (next == std::string::npos) break;
        std::size_t content_end = next;
        // Remove trailing CRLF before boundary
        if (content_end >= 2 && body[content_end - 2] == '\r' && body[content_end - 1] == '\n') {
            content_end -= 2;
        } else if (content_end >= 1 && body[content_end - 1] == '\n') {
            content_end -= 1;
        }

        MultipartFormData part;
        ParseContentDisposition(disposition, part.name, part.filename);
        part.content = body.substr(pos, content_end - pos);
        part.content_type = part_content_type;
        if (!part.name.empty()) {
            parts.emplace(part.name, std::move(part));
        }

        pos = next + delimiter.size();
        // Check for end
        if (pos + 1 < body.size() && body[pos] == '-' && body[pos + 1] == '-') break;
    }
    return true;
}

bool HttpServer::MatchRoute(
    const std::string & pattern,
    const std::string & path,
    std::unordered_map<std::string, std::string> & path_params) {
    path_params.clear();

    // Split pattern and path into segments
    auto split = [](const std::string & s) -> std::vector<std::string> {
        std::vector<std::string> parts;
        std::size_t start = 0;
        if (!s.empty() && s[0] == '/') start = 1;
        while (start < s.size()) {
            auto end = s.find('/', start);
            if (end == std::string::npos) end = s.size();
            parts.push_back(s.substr(start, end - start));
            start = end + 1;
        }
        return parts;
    };

    const auto pat_parts = split(pattern);
    const auto path_parts = split(path);
    if (pat_parts.size() != path_parts.size()) return false;

    std::unordered_map<std::string, std::string> captured;
    for (std::size_t i = 0; i < pat_parts.size(); ++i) {
        if (!pat_parts[i].empty() && pat_parts[i][0] == ':') {
            captured[pat_parts[i].substr(1)] = path_parts[i];
        } else if (pat_parts[i] != path_parts[i]) {
            return false;
        }
    }
    path_params = std::move(captured);
    return true;
}

}  // namespace qasr
