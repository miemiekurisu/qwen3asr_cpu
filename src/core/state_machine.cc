#include "qasr/core/state_machine.h"

namespace qasr {

std::string_view SessionStateName(SessionState state) noexcept {
    switch (state) {
        case SessionState::kCreated:  return "Created";
        case SessionState::kWarmed:   return "Warmed";
        case SessionState::kRunning:  return "Running";
        case SessionState::kFlushing: return "Flushing";
        case SessionState::kClosed:   return "Closed";
    }
    return "Unknown";
}

std::string_view RequestStateName(RequestState state) noexcept {
    switch (state) {
        case RequestState::kAccepted:  return "Accepted";
        case RequestState::kQueued:    return "Queued";
        case RequestState::kRunning:   return "Running";
        case RequestState::kStreaming: return "Streaming";
        case RequestState::kSucceeded: return "Succeeded";
        case RequestState::kFailed:    return "Failed";
        case RequestState::kCancelled: return "Cancelled";
    }
    return "Unknown";
}

std::string_view RealtimeTextLaneName(RealtimeTextLane lane) noexcept {
    switch (lane) {
        case RealtimeTextLane::kUnseen:  return "Unseen";
        case RealtimeTextLane::kPartial: return "Partial";
        case RealtimeTextLane::kStable:  return "Stable";
        case RealtimeTextLane::kFinal:   return "Final";
    }
    return "Unknown";
}

std::string_view StreamChunkStateName(StreamChunkState state) noexcept {
    switch (state) {
        case StreamChunkState::kIngested:  return "Ingested";
        case StreamChunkState::kEncoded:   return "Encoded";
        case StreamChunkState::kPrefilled: return "Prefilled";
        case StreamChunkState::kDecoded:   return "Decoded";
        case StreamChunkState::kCommitted: return "Committed";
    }
    return "Unknown";
}

Status ValidateSessionTransition(SessionState current, SessionState target) {
    // Created -> Warmed -> Running -> Flushing -> Closed
    // Also allow: any -> Closed (force close)
    if (target == SessionState::kClosed) {
        return OkStatus();
    }
    const int cur = static_cast<int>(current);
    const int tgt = static_cast<int>(target);
    if (tgt == cur + 1) {
        return OkStatus();
    }
    return Status(StatusCode::kFailedPrecondition,
                  std::string("invalid session transition: ") +
                  std::string(SessionStateName(current)) + " -> " +
                  std::string(SessionStateName(target)));
}

Status ValidateRequestTransition(RequestState current, RequestState target) {
    if (IsTerminalRequestState(current)) {
        return Status(StatusCode::kFailedPrecondition,
                      std::string("request already terminal: ") +
                      std::string(RequestStateName(current)));
    }
    // Cancelled and Failed can be reached from any non-terminal state
    if (target == RequestState::kCancelled || target == RequestState::kFailed) {
        return OkStatus();
    }
    // Normal forward: Accepted->Queued->Running->Streaming->Succeeded
    const int cur = static_cast<int>(current);
    const int tgt = static_cast<int>(target);
    if (tgt == cur + 1) {
        return OkStatus();
    }
    // Running -> Succeeded (skip Streaming for offline tasks)
    if (current == RequestState::kRunning && target == RequestState::kSucceeded) {
        return OkStatus();
    }
    return Status(StatusCode::kFailedPrecondition,
                  std::string("invalid request transition: ") +
                  std::string(RequestStateName(current)) + " -> " +
                  std::string(RequestStateName(target)));
}

bool IsTerminalRequestState(RequestState state) noexcept {
    return state == RequestState::kSucceeded ||
           state == RequestState::kFailed ||
           state == RequestState::kCancelled;
}

bool IsTerminalSessionState(SessionState state) noexcept {
    return state == SessionState::kClosed;
}

}  // namespace qasr
