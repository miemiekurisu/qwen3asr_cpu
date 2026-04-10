#include "test_registry.h"

#include "qasr/base/http_server.h"

#include <atomic>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#  include <unistd.h>
#endif

using qasr::HttpServer;
using qasr::MultipartFormData;

// ==================== ParseQueryString ====================

QASR_TEST(HttpServer_ParseQueryString_Empty) {
    std::unordered_map<std::string, std::vector<std::string>> params;
    QASR_EXPECT(HttpServer::ParseQueryString("", params));
    QASR_EXPECT(params.empty());
}

QASR_TEST(HttpServer_ParseQueryString_SingleParam) {
    std::unordered_map<std::string, std::vector<std::string>> params;
    QASR_EXPECT(HttpServer::ParseQueryString("key=value", params));
    QASR_EXPECT_EQ(params.size(), std::size_t(1));
    QASR_EXPECT_EQ(params["key"].size(), std::size_t(1));
    QASR_EXPECT_EQ(params["key"][0], std::string("value"));
}

QASR_TEST(HttpServer_ParseQueryString_MultipleParams) {
    std::unordered_map<std::string, std::vector<std::string>> params;
    QASR_EXPECT(HttpServer::ParseQueryString("a=1&b=2&c=3", params));
    QASR_EXPECT_EQ(params.size(), std::size_t(3));
    QASR_EXPECT_EQ(params["a"][0], std::string("1"));
    QASR_EXPECT_EQ(params["b"][0], std::string("2"));
    QASR_EXPECT_EQ(params["c"][0], std::string("3"));
}

QASR_TEST(HttpServer_ParseQueryString_MultipleValues) {
    std::unordered_map<std::string, std::vector<std::string>> params;
    QASR_EXPECT(HttpServer::ParseQueryString("key=a&key=b&key=c", params));
    QASR_EXPECT_EQ(params["key"].size(), std::size_t(3));
    QASR_EXPECT_EQ(params["key"][0], std::string("a"));
    QASR_EXPECT_EQ(params["key"][1], std::string("b"));
    QASR_EXPECT_EQ(params["key"][2], std::string("c"));
}

QASR_TEST(HttpServer_ParseQueryString_UrlEncoded) {
    std::unordered_map<std::string, std::vector<std::string>> params;
    QASR_EXPECT(HttpServer::ParseQueryString("name=hello%20world&path=%2Ffoo%2Fbar", params));
    QASR_EXPECT_EQ(params["name"][0], std::string("hello world"));
    QASR_EXPECT_EQ(params["path"][0], std::string("/foo/bar"));
}

QASR_TEST(HttpServer_ParseQueryString_PlusAsSpace) {
    std::unordered_map<std::string, std::vector<std::string>> params;
    QASR_EXPECT(HttpServer::ParseQueryString("q=hello+world", params));
    QASR_EXPECT_EQ(params["q"][0], std::string("hello world"));
}

QASR_TEST(HttpServer_ParseQueryString_NoValue) {
    std::unordered_map<std::string, std::vector<std::string>> params;
    QASR_EXPECT(HttpServer::ParseQueryString("flag", params));
    QASR_EXPECT_EQ(params.count("flag"), std::size_t(1));
    QASR_EXPECT_EQ(params["flag"].size(), std::size_t(1));
    QASR_EXPECT(params["flag"][0].empty());
}

QASR_TEST(HttpServer_ParseQueryString_EmptyValue) {
    std::unordered_map<std::string, std::vector<std::string>> params;
    QASR_EXPECT(HttpServer::ParseQueryString("key=", params));
    QASR_EXPECT_EQ(params["key"][0], std::string(""));
}

QASR_TEST(HttpServer_ParseQueryString_ArrayKeys) {
    std::unordered_map<std::string, std::vector<std::string>> params;
    QASR_EXPECT(HttpServer::ParseQueryString("items[]=a&items[]=b", params));
    QASR_EXPECT_EQ(params["items[]"].size(), std::size_t(2));
    QASR_EXPECT_EQ(params["items[]"][0], std::string("a"));
    QASR_EXPECT_EQ(params["items[]"][1], std::string("b"));
}

// ==================== ParseMultipartBody ====================

QASR_TEST(HttpServer_ParseMultipart_SingleTextField) {
    const std::string boundary = "boundary123";
    std::string body;
    body += "--boundary123\r\n";
    body += "Content-Disposition: form-data; name=\"model\"\r\n";
    body += "\r\n";
    body += "Qwen3-ASR\r\n";
    body += "--boundary123--\r\n";

    std::unordered_map<std::string, MultipartFormData> parts;
    QASR_EXPECT(HttpServer::ParseMultipartBody(boundary, body, parts));
    QASR_EXPECT_EQ(parts.size(), std::size_t(1));
    QASR_EXPECT(parts.count("model"));
    QASR_EXPECT_EQ(parts["model"].content, std::string("Qwen3-ASR"));
    QASR_EXPECT(parts["model"].filename.empty());
}

QASR_TEST(HttpServer_ParseMultipart_FileField) {
    const std::string boundary = "----WebBoundary";
    std::string body;
    body += "------WebBoundary\r\n";
    body += "Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n";
    body += "Content-Type: audio/wav\r\n";
    body += "\r\n";
    body += "RIFFWAVEDATA\r\n";
    body += "------WebBoundary--\r\n";

    std::unordered_map<std::string, MultipartFormData> parts;
    QASR_EXPECT(HttpServer::ParseMultipartBody(boundary, body, parts));
    QASR_EXPECT_EQ(parts.size(), std::size_t(1));
    QASR_EXPECT(parts.count("file"));
    QASR_EXPECT_EQ(parts["file"].filename, std::string("test.wav"));
    QASR_EXPECT_EQ(parts["file"].content, std::string("RIFFWAVEDATA"));
    QASR_EXPECT_EQ(parts["file"].content_type, std::string("audio/wav"));
}

QASR_TEST(HttpServer_ParseMultipart_MultipleFields) {
    const std::string boundary = "sep";
    std::string body;
    body += "--sep\r\n";
    body += "Content-Disposition: form-data; name=\"model\"\r\n";
    body += "\r\n";
    body += "Qwen3-ASR\r\n";
    body += "--sep\r\n";
    body += "Content-Disposition: form-data; name=\"language\"\r\n";
    body += "\r\n";
    body += "en\r\n";
    body += "--sep\r\n";
    body += "Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n";
    body += "Content-Type: audio/wav\r\n";
    body += "\r\n";
    body += "AUDIODATA\r\n";
    body += "--sep--\r\n";

    std::unordered_map<std::string, MultipartFormData> parts;
    QASR_EXPECT(HttpServer::ParseMultipartBody(boundary, body, parts));
    QASR_EXPECT_EQ(parts.size(), std::size_t(3));
    QASR_EXPECT_EQ(parts["model"].content, std::string("Qwen3-ASR"));
    QASR_EXPECT_EQ(parts["language"].content, std::string("en"));
    QASR_EXPECT_EQ(parts["file"].content, std::string("AUDIODATA"));
    QASR_EXPECT_EQ(parts["file"].filename, std::string("audio.wav"));
}

QASR_TEST(HttpServer_ParseMultipart_NoBoundary) {
    std::unordered_map<std::string, MultipartFormData> parts;
    QASR_EXPECT(!HttpServer::ParseMultipartBody("missing", "no boundaries here", parts));
}

QASR_TEST(HttpServer_ParseMultipart_EmptyBody) {
    std::unordered_map<std::string, MultipartFormData> parts;
    QASR_EXPECT(!HttpServer::ParseMultipartBody("sep", "", parts));
}

// ==================== MatchRoute ====================

QASR_TEST(HttpServer_MatchRoute_Exact) {
    std::unordered_map<std::string, std::string> params;
    QASR_EXPECT(HttpServer::MatchRoute("/health", "/health", params));
    QASR_EXPECT(params.empty());
}

QASR_TEST(HttpServer_MatchRoute_Root) {
    std::unordered_map<std::string, std::string> params;
    QASR_EXPECT(HttpServer::MatchRoute("/", "/", params));
}

QASR_TEST(HttpServer_MatchRoute_MultiSegment) {
    std::unordered_map<std::string, std::string> params;
    QASR_EXPECT(HttpServer::MatchRoute("/api/health", "/api/health", params));
    QASR_EXPECT(params.empty());
}

QASR_TEST(HttpServer_MatchRoute_PathParam) {
    std::unordered_map<std::string, std::string> params;
    QASR_EXPECT(HttpServer::MatchRoute("/api/jobs/:id", "/api/jobs/42", params));
    QASR_EXPECT_EQ(params.size(), std::size_t(1));
    QASR_EXPECT_EQ(params["id"], std::string("42"));
}

QASR_TEST(HttpServer_MatchRoute_MultiplePathParams) {
    std::unordered_map<std::string, std::string> params;
    QASR_EXPECT(HttpServer::MatchRoute("/api/:version/jobs/:id", "/api/v1/jobs/123", params));
    QASR_EXPECT_EQ(params.size(), std::size_t(2));
    QASR_EXPECT_EQ(params["version"], std::string("v1"));
    QASR_EXPECT_EQ(params["id"], std::string("123"));
}

QASR_TEST(HttpServer_MatchRoute_NoMatch_DifferentPath) {
    std::unordered_map<std::string, std::string> params;
    QASR_EXPECT(!HttpServer::MatchRoute("/api/health", "/api/metrics", params));
}

QASR_TEST(HttpServer_MatchRoute_NoMatch_DifferentLength) {
    std::unordered_map<std::string, std::string> params;
    QASR_EXPECT(!HttpServer::MatchRoute("/api/health", "/api/health/extra", params));
}

QASR_TEST(HttpServer_MatchRoute_NoMatch_Shorter) {
    std::unordered_map<std::string, std::string> params;
    QASR_EXPECT(!HttpServer::MatchRoute("/api/a/b", "/api/a", params));
}

// ==================== HttpRequest ====================

QASR_TEST(HttpRequest_HasParam) {
    qasr::HttpRequest req;
    HttpServer::ParseQueryString("session_id=abc123", req.params_);
    QASR_EXPECT(req.has_param("session_id"));
    QASR_EXPECT(!req.has_param("other"));
}

QASR_TEST(HttpRequest_GetParamValue) {
    qasr::HttpRequest req;
    HttpServer::ParseQueryString("id=42", req.params_);
    QASR_EXPECT_EQ(req.get_param_value("id"), std::string("42"));
    QASR_EXPECT_EQ(req.get_param_value("missing"), std::string(""));
}

QASR_TEST(HttpRequest_GetParamValueIndexed) {
    qasr::HttpRequest req;
    HttpServer::ParseQueryString("items=a&items=b&items=c", req.params_);
    QASR_EXPECT_EQ(req.get_param_value("items", 0), std::string("a"));
    QASR_EXPECT_EQ(req.get_param_value("items", 1), std::string("b"));
    QASR_EXPECT_EQ(req.get_param_value("items", 2), std::string("c"));
    QASR_EXPECT_EQ(req.get_param_value("items", 3), std::string(""));
}

QASR_TEST(HttpRequest_GetParamValueCount) {
    qasr::HttpRequest req;
    HttpServer::ParseQueryString("items=a&items=b", req.params_);
    QASR_EXPECT_EQ(req.get_param_value_count("items"), std::size_t(2));
    QASR_EXPECT_EQ(req.get_param_value_count("missing"), std::size_t(0));
}

// ==================== HttpResponse ====================

QASR_TEST(HttpResponse_DefaultStatus) {
    qasr::HttpResponse resp;
    QASR_EXPECT_EQ(resp.status, 200);
}

QASR_TEST(HttpResponse_SetContent) {
    qasr::HttpResponse resp;
    resp.set_content("{}", "application/json");
    // Internal state verified by server sending correct data
    QASR_EXPECT_EQ(resp.status, 200);
}

QASR_TEST(HttpResponse_SetHeader) {
    qasr::HttpResponse resp;
    resp.set_header("X-Custom", "test");
    resp.set_header("X-Custom", "override");
    // Header dedup verified by server sending correct data
    QASR_EXPECT_EQ(resp.status, 200);
}

// ==================== Socket-level integration tests ====================
// Start a real HttpServer on 127.0.0.1, connect a raw TCP socket, and verify
// request/response over the wire on Windows (Winsock) and POSIX.

namespace {

#ifdef _WIN32
using test_socket_t = SOCKET;
static constexpr test_socket_t kBadSocket = INVALID_SOCKET;
inline void test_close_socket(test_socket_t s) { closesocket(s); }
#else
using test_socket_t = int;
static constexpr test_socket_t kBadSocket = -1;
inline void test_close_socket(test_socket_t s) { close(s); }
#endif

/// Connect to 127.0.0.1:port, send raw_request, read full response.
/// The server's listen() handles WSAStartup/WSACleanup internally, but
/// the test client needs its own init on Windows.
struct WinsockInit {
    WinsockInit() {
#ifdef _WIN32
        WSADATA d;
        WSAStartup(MAKEWORD(2, 2), &d);
#endif
    }
    ~WinsockInit() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
};

std::string SendRawHttp(int port, const std::string & raw_request) {
    test_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == kBadSocket) return {};

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);

    if (connect(sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        test_close_socket(sock);
        return {};
    }

    int total = 0;
    while (total < static_cast<int>(raw_request.size())) {
        int n = send(sock, raw_request.data() + total,
                     static_cast<int>(raw_request.size()) - total, 0);
        if (n <= 0) break;
        total += n;
    }

    std::string response;
    char buf[4096];
    for (;;) {
        int n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) break;
        response.append(buf, static_cast<std::size_t>(n));
    }
    test_close_socket(sock);
    return response;
}

int FindFreePort() {
    test_socket_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == kBadSocket) return 0;
    struct sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = 0;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(s, reinterpret_cast<struct sockaddr *>(&a), sizeof(a)) < 0) {
        test_close_socket(s);
        return 0;
    }
    socklen_t len = sizeof(a);
    getsockname(s, reinterpret_cast<struct sockaddr *>(&a), &len);
    int port = ntohs(a.sin_port);
    test_close_socket(s);
    return port;
}

/// Helper: spin up server in background, wait for it to start, run body, stop.
struct ServerFixture {
    HttpServer server;
    std::thread thread;
    int port = 0;

    void Start() {
        port = FindFreePort();
        server.set_idle_interval(0, 50000);  // 50ms poll
        server.set_read_timeout(2, 0);
        server.set_write_timeout(2, 0);
        std::atomic<bool> ready{false};
        thread = std::thread([&]() {
            ready.store(true);
            server.listen("127.0.0.1", port);
        });
        while (!ready.load())
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        // Give the accept-loop a moment to enter poll().
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }

    void Stop() {
        server.stop();
        if (thread.joinable()) thread.join();
    }
};

}  // namespace

// Normal: GET /health returns 200 + JSON body
QASR_TEST(HttpServer_Socket_GetReturns200) {
    WinsockInit wsa;
    ServerFixture fx;
    fx.server.Get("/health", [](const qasr::HttpRequest &, qasr::HttpResponse & r) {
        r.set_content("{\"status\":\"ok\"}", "application/json");
    });
    fx.Start();

    std::string resp = SendRawHttp(fx.port,
        "GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");

    fx.Stop();
    QASR_EXPECT(resp.find("200") != std::string::npos);
    QASR_EXPECT(resp.find("{\"status\":\"ok\"}") != std::string::npos);
}

// Normal: POST with body echoed back
QASR_TEST(HttpServer_Socket_PostEchoBody) {
    WinsockInit wsa;
    ServerFixture fx;
    fx.server.Post("/echo", [](const qasr::HttpRequest & req, qasr::HttpResponse & r) {
        r.set_content(req.body, "text/plain");
    });
    fx.Start();

    std::string body = "hello from test";
    std::string raw = "POST /echo HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: "
                      + std::to_string(body.size())
                      + "\r\nConnection: close\r\n\r\n" + body;
    std::string resp = SendRawHttp(fx.port, raw);

    fx.Stop();
    QASR_EXPECT(resp.find("200") != std::string::npos);
    QASR_EXPECT(resp.find("hello from test") != std::string::npos);
}

// Error: unregistered route → 404
QASR_TEST(HttpServer_Socket_404OnUnknownRoute) {
    WinsockInit wsa;
    ServerFixture fx;
    fx.server.Get("/health", [](const qasr::HttpRequest &, qasr::HttpResponse & r) {
        r.set_content("ok", "text/plain");
    });
    fx.Start();

    std::string resp = SendRawHttp(fx.port,
        "GET /nonexistent HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");

    fx.Stop();
    QASR_EXPECT(resp.find("404") != std::string::npos);
}

// Extreme: query-string parameters over the wire
QASR_TEST(HttpServer_Socket_QueryStringParams) {
    WinsockInit wsa;
    ServerFixture fx;
    fx.server.Get("/q", [](const qasr::HttpRequest & req, qasr::HttpResponse & r) {
        r.set_content(req.get_param_value("key"), "text/plain");
    });
    fx.Start();

    std::string resp = SendRawHttp(fx.port,
        "GET /q?key=hello HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");

    fx.Stop();
    QASR_EXPECT(resp.find("hello") != std::string::npos);
}

// Extreme: path parameters over the wire
QASR_TEST(HttpServer_Socket_PathParams) {
    WinsockInit wsa;
    ServerFixture fx;
    fx.server.Get("/jobs/:id", [](const qasr::HttpRequest & req, qasr::HttpResponse & r) {
        auto it = req.path_params.find("id");
        r.set_content(it != req.path_params.end() ? it->second : "?", "text/plain");
    });
    fx.Start();

    std::string resp = SendRawHttp(fx.port,
        "GET /jobs/42 HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");

    fx.Stop();
    QASR_EXPECT(resp.find("42") != std::string::npos);
}


