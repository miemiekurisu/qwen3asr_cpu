#pragma once

#include <string_view>

#include "qasr/core/status.h"

namespace qasr {

// Session lifecycle: Created → Warmed → Running → Flushing → Closed
enum class SessionState {
    kCreated = 0,
    kWarmed,
    kRunning,
    kFlushing,
    kClosed,
};

// Request lifecycle: Accepted → Queued → Running → Streaming → Succeeded/Failed/Cancelled
enum class RequestState {
    kAccepted = 0,
    kQueued,
    kRunning,
    kStreaming,
    kSucceeded,
    kFailed,
    kCancelled,
};

// Realtime text lifecycle: Unseen → Partial → Stable → Final
enum class RealtimeTextLane {
    kUnseen = 0,
    kPartial,
    kStable,
    kFinal,
};

// Stream chunk lifecycle: Ingested → Encoded → Prefilled → Decoded → Committed
enum class StreamChunkState {
    kIngested = 0,
    kEncoded,
    kPrefilled,
    kDecoded,
    kCommitted,
};

/// Pre: none.
/// Post: returns human-readable name.
/// Thread-safe: yes (pure function, no shared state).
std::string_view SessionStateName(SessionState state) noexcept;

/// Pre: none.
/// Post: returns human-readable name.
/// Thread-safe: yes.
std::string_view RequestStateName(RequestState state) noexcept;

/// Pre: none.
/// Post: returns human-readable name.
/// Thread-safe: yes.
std::string_view RealtimeTextLaneName(RealtimeTextLane lane) noexcept;

/// Pre: none.
/// Post: returns human-readable name.
/// Thread-safe: yes.
std::string_view StreamChunkStateName(StreamChunkState state) noexcept;

/// Pre: current != target, valid transition.
/// Post: returns Ok if transition is legal, error otherwise.
/// Thread-safe: yes.
Status ValidateSessionTransition(SessionState current, SessionState target);

/// Pre: current != target, valid transition.
/// Post: returns Ok if transition is legal, error otherwise.
/// Thread-safe: yes.
Status ValidateRequestTransition(RequestState current, RequestState target);

bool IsTerminalRequestState(RequestState state) noexcept;
bool IsTerminalSessionState(SessionState state) noexcept;

}  // namespace qasr
