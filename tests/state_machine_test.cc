#include "tests/test_registry.h"
#include "qasr/core/state_machine.h"

// --- Normal ---

QASR_TEST(SessionStateNameReturnsCorrectStrings) {
    QASR_EXPECT_EQ(qasr::SessionStateName(qasr::SessionState::kCreated), "Created");
    QASR_EXPECT_EQ(qasr::SessionStateName(qasr::SessionState::kWarmed), "Warmed");
    QASR_EXPECT_EQ(qasr::SessionStateName(qasr::SessionState::kRunning), "Running");
    QASR_EXPECT_EQ(qasr::SessionStateName(qasr::SessionState::kFlushing), "Flushing");
    QASR_EXPECT_EQ(qasr::SessionStateName(qasr::SessionState::kClosed), "Closed");
}

QASR_TEST(RequestStateNameReturnsCorrectStrings) {
    QASR_EXPECT_EQ(qasr::RequestStateName(qasr::RequestState::kAccepted), "Accepted");
    QASR_EXPECT_EQ(qasr::RequestStateName(qasr::RequestState::kQueued), "Queued");
    QASR_EXPECT_EQ(qasr::RequestStateName(qasr::RequestState::kRunning), "Running");
    QASR_EXPECT_EQ(qasr::RequestStateName(qasr::RequestState::kStreaming), "Streaming");
    QASR_EXPECT_EQ(qasr::RequestStateName(qasr::RequestState::kSucceeded), "Succeeded");
    QASR_EXPECT_EQ(qasr::RequestStateName(qasr::RequestState::kFailed), "Failed");
    QASR_EXPECT_EQ(qasr::RequestStateName(qasr::RequestState::kCancelled), "Cancelled");
}

QASR_TEST(RealtimeTextLaneNameReturnsCorrectStrings) {
    QASR_EXPECT_EQ(qasr::RealtimeTextLaneName(qasr::RealtimeTextLane::kUnseen), "Unseen");
    QASR_EXPECT_EQ(qasr::RealtimeTextLaneName(qasr::RealtimeTextLane::kPartial), "Partial");
    QASR_EXPECT_EQ(qasr::RealtimeTextLaneName(qasr::RealtimeTextLane::kStable), "Stable");
    QASR_EXPECT_EQ(qasr::RealtimeTextLaneName(qasr::RealtimeTextLane::kFinal), "Final");
}

QASR_TEST(StreamChunkStateNameReturnsCorrectStrings) {
    QASR_EXPECT_EQ(qasr::StreamChunkStateName(qasr::StreamChunkState::kIngested), "Ingested");
    QASR_EXPECT_EQ(qasr::StreamChunkStateName(qasr::StreamChunkState::kEncoded), "Encoded");
    QASR_EXPECT_EQ(qasr::StreamChunkStateName(qasr::StreamChunkState::kPrefilled), "Prefilled");
    QASR_EXPECT_EQ(qasr::StreamChunkStateName(qasr::StreamChunkState::kDecoded), "Decoded");
    QASR_EXPECT_EQ(qasr::StreamChunkStateName(qasr::StreamChunkState::kCommitted), "Committed");
}

// --- Normal transitions ---

QASR_TEST(SessionTransitionForwardIsValid) {
    QASR_EXPECT(qasr::ValidateSessionTransition(qasr::SessionState::kCreated, qasr::SessionState::kWarmed).ok());
    QASR_EXPECT(qasr::ValidateSessionTransition(qasr::SessionState::kWarmed, qasr::SessionState::kRunning).ok());
    QASR_EXPECT(qasr::ValidateSessionTransition(qasr::SessionState::kRunning, qasr::SessionState::kFlushing).ok());
    QASR_EXPECT(qasr::ValidateSessionTransition(qasr::SessionState::kFlushing, qasr::SessionState::kClosed).ok());
}

QASR_TEST(SessionForceCloseFromAnyState) {
    QASR_EXPECT(qasr::ValidateSessionTransition(qasr::SessionState::kCreated, qasr::SessionState::kClosed).ok());
    QASR_EXPECT(qasr::ValidateSessionTransition(qasr::SessionState::kRunning, qasr::SessionState::kClosed).ok());
}

QASR_TEST(RequestTransitionForwardIsValid) {
    QASR_EXPECT(qasr::ValidateRequestTransition(qasr::RequestState::kAccepted, qasr::RequestState::kQueued).ok());
    QASR_EXPECT(qasr::ValidateRequestTransition(qasr::RequestState::kQueued, qasr::RequestState::kRunning).ok());
    QASR_EXPECT(qasr::ValidateRequestTransition(qasr::RequestState::kRunning, qasr::RequestState::kStreaming).ok());
    QASR_EXPECT(qasr::ValidateRequestTransition(qasr::RequestState::kStreaming, qasr::RequestState::kSucceeded).ok());
}

QASR_TEST(RequestSkipStreamingForOffline) {
    QASR_EXPECT(qasr::ValidateRequestTransition(qasr::RequestState::kRunning, qasr::RequestState::kSucceeded).ok());
}

// --- Error: invalid transitions ---

QASR_TEST(SessionBackwardTransitionRejected) {
    QASR_EXPECT(!qasr::ValidateSessionTransition(qasr::SessionState::kRunning, qasr::SessionState::kCreated).ok());
}

QASR_TEST(RequestFromTerminalRejected) {
    QASR_EXPECT(!qasr::ValidateRequestTransition(qasr::RequestState::kSucceeded, qasr::RequestState::kRunning).ok());
    QASR_EXPECT(!qasr::ValidateRequestTransition(qasr::RequestState::kFailed, qasr::RequestState::kRunning).ok());
    QASR_EXPECT(!qasr::ValidateRequestTransition(qasr::RequestState::kCancelled, qasr::RequestState::kRunning).ok());
}

// --- Extreme: cancel/fail from any non-terminal ---

QASR_TEST(RequestCancelFromAnyNonTerminal) {
    QASR_EXPECT(qasr::ValidateRequestTransition(qasr::RequestState::kAccepted, qasr::RequestState::kCancelled).ok());
    QASR_EXPECT(qasr::ValidateRequestTransition(qasr::RequestState::kQueued, qasr::RequestState::kCancelled).ok());
    QASR_EXPECT(qasr::ValidateRequestTransition(qasr::RequestState::kRunning, qasr::RequestState::kCancelled).ok());
    QASR_EXPECT(qasr::ValidateRequestTransition(qasr::RequestState::kStreaming, qasr::RequestState::kCancelled).ok());
}

QASR_TEST(IsTerminalCorrect) {
    QASR_EXPECT(!qasr::IsTerminalRequestState(qasr::RequestState::kAccepted));
    QASR_EXPECT(!qasr::IsTerminalRequestState(qasr::RequestState::kRunning));
    QASR_EXPECT(qasr::IsTerminalRequestState(qasr::RequestState::kSucceeded));
    QASR_EXPECT(qasr::IsTerminalRequestState(qasr::RequestState::kFailed));
    QASR_EXPECT(qasr::IsTerminalRequestState(qasr::RequestState::kCancelled));
    QASR_EXPECT(!qasr::IsTerminalSessionState(qasr::SessionState::kRunning));
    QASR_EXPECT(qasr::IsTerminalSessionState(qasr::SessionState::kClosed));
}
