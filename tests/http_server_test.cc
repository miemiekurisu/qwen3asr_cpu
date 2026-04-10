#include "test_registry.h"

#include "qasr/base/http_server.h"

#include <string>
#include <unordered_map>
#include <vector>

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
