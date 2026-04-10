#include "tests/test_registry.h"

#include "qasr/core/status.h"

QASR_TEST(StatusDefaultsToOk) {
    const qasr::Status status;
    QASR_EXPECT(status.ok());
    QASR_EXPECT_EQ(status.code(), qasr::StatusCode::kOk);
    QASR_EXPECT_EQ(status.ToString(), std::string("OK"));
}

QASR_TEST(StatusCarriesMessage) {
    const qasr::Status status(qasr::StatusCode::kInvalidArgument, "bad arg");
    QASR_EXPECT(!status.ok());
    QASR_EXPECT_EQ(status.message(), std::string("bad arg"));
    QASR_EXPECT_EQ(status.ToString(), std::string("INVALID_ARGUMENT: bad arg"));
}

QASR_TEST(StatusCodeNamesCoverAllValues) {
    QASR_EXPECT_EQ(qasr::StatusCodeName(qasr::StatusCode::kOk), std::string_view("OK"));
    QASR_EXPECT_EQ(qasr::StatusCodeName(qasr::StatusCode::kInvalidArgument), std::string_view("INVALID_ARGUMENT"));
    QASR_EXPECT_EQ(qasr::StatusCodeName(qasr::StatusCode::kOutOfRange), std::string_view("OUT_OF_RANGE"));
    QASR_EXPECT_EQ(qasr::StatusCodeName(qasr::StatusCode::kFailedPrecondition), std::string_view("FAILED_PRECONDITION"));
    QASR_EXPECT_EQ(qasr::StatusCodeName(qasr::StatusCode::kNotFound), std::string_view("NOT_FOUND"));
    QASR_EXPECT_EQ(qasr::StatusCodeName(qasr::StatusCode::kInternal), std::string_view("INTERNAL"));
    QASR_EXPECT_EQ(qasr::StatusCodeName(qasr::StatusCode::kUnimplemented), std::string_view("UNIMPLEMENTED"));
}

QASR_TEST(OkStatusFactoryReturnsOk) {
    const qasr::Status status = qasr::OkStatus();
    QASR_EXPECT(status.ok());
    QASR_EXPECT_EQ(status.ToString(), std::string("OK"));
}
