#include "qasr/runtime/session_manager.h"

#include <chrono>
#include <sstream>

namespace qasr {

namespace {

std::int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

SessionManager::SessionManager(std::int32_t max_sessions, std::int64_t session_ttl_ms)
    : max_sessions_(max_sessions), session_ttl_ms_(session_ttl_ms) {}

Status SessionManager::CreateSession(std::string * session_id) {
    if (!session_id) {
        return Status(StatusCode::kInvalidArgument, "session_id must not be null");
    }
    std::lock_guard<std::mutex> lock(mu_);

    if (static_cast<std::int32_t>(sessions_.size()) >= max_sessions_) {
        return Status(StatusCode::kFailedPrecondition,
                      "max sessions reached: " + std::to_string(max_sessions_));
    }

    std::ostringstream id_builder;
    id_builder << "sess-" << next_id_++;
    *session_id = id_builder.str();

    SessionInfo info;
    info.session_id = *session_id;
    info.state = SessionState::kCreated;
    info.created_at_ms = NowMs();
    info.last_active_ms = info.created_at_ms;

    sessions_[*session_id] = std::move(info);
    return OkStatus();
}

Status SessionManager::CloseSession(const std::string & session_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return Status(StatusCode::kNotFound, "session not found: " + session_id);
    }
    it->second.state = SessionState::kClosed;
    it->second.last_active_ms = NowMs();
    return OkStatus();
}

Status SessionManager::LookupSession(const std::string & session_id,
                                      SessionInfo * info) const {
    if (!info) {
        return Status(StatusCode::kInvalidArgument, "info must not be null");
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return Status(StatusCode::kNotFound, "session not found: " + session_id);
    }
    *info = it->second;
    return OkStatus();
}

Status SessionManager::TransitionSession(const std::string & session_id,
                                          SessionState new_state) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = sessions_.find(session_id);
    if (it == sessions_.end()) {
        return Status(StatusCode::kNotFound, "session not found: " + session_id);
    }

    Status s = ValidateSessionTransition(it->second.state, new_state);
    if (!s.ok()) return s;

    it->second.state = new_state;
    it->second.last_active_ms = NowMs();
    return OkStatus();
}

std::int32_t SessionManager::SweepExpiredSessions() {
    std::lock_guard<std::mutex> lock(mu_);
    const std::int64_t now = NowMs();
    std::int32_t swept = 0;

    for (auto it = sessions_.begin(); it != sessions_.end();) {
        if (it->second.state != SessionState::kClosed &&
            now - it->second.last_active_ms > session_ttl_ms_) {
            it->second.state = SessionState::kClosed;
            ++swept;
        }
        // Remove closed sessions
        if (it->second.state == SessionState::kClosed &&
            now - it->second.last_active_ms > session_ttl_ms_) {
            it = sessions_.erase(it);
        } else {
            ++it;
        }
    }
    return swept;
}

std::int32_t SessionManager::active_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::int32_t count = 0;
    for (const auto & [id, info] : sessions_) {
        if (info.state != SessionState::kClosed) {
            ++count;
        }
    }
    return count;
}

}  // namespace qasr
