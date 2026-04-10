#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "qasr/core/state_machine.h"
#include "qasr/core/status.h"

namespace qasr {

struct SessionInfo {
    std::string session_id;
    SessionState state = SessionState::kCreated;
    std::int64_t created_at_ms = 0;
    std::int64_t last_active_ms = 0;
};

/// Session lifecycle manager.
/// Pre: none.
/// Post: manages creation, lookup, closing, and expiry of sessions.
/// Thread-safe: yes (internally synchronized).
class SessionManager {
public:
    explicit SessionManager(std::int32_t max_sessions = 64,
                            std::int64_t session_ttl_ms = 300000);

    Status CreateSession(std::string * session_id);
    Status CloseSession(const std::string & session_id);
    Status LookupSession(const std::string & session_id, SessionInfo * info) const;
    Status TransitionSession(const std::string & session_id, SessionState new_state);

    std::int32_t SweepExpiredSessions();
    std::int32_t active_count() const;
    std::int32_t max_sessions() const noexcept { return max_sessions_; }

private:
    mutable std::mutex mu_;
    std::unordered_map<std::string, SessionInfo> sessions_;
    std::int32_t max_sessions_;
    std::int64_t session_ttl_ms_;
    std::uint64_t next_id_ = 1;
};

}  // namespace qasr
