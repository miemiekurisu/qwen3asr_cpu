#include "tests/test_registry.h"
#include "qasr/runtime/session_manager.h"

// --- Normal ---

QASR_TEST(SessionManagerCreateSession) {
    qasr::SessionManager mgr(10, 300000);
    std::string id;
    qasr::Status s = mgr.CreateSession(&id);
    QASR_EXPECT(s.ok());
    QASR_EXPECT(!id.empty());
    QASR_EXPECT_EQ(mgr.active_count(), int32_t(1));
}

QASR_TEST(SessionManagerCreateMultipleSessions) {
    qasr::SessionManager mgr(10, 300000);
    std::string id1, id2;
    QASR_EXPECT(mgr.CreateSession(&id1).ok());
    QASR_EXPECT(mgr.CreateSession(&id2).ok());
    QASR_EXPECT(id1 != id2);
    QASR_EXPECT_EQ(mgr.active_count(), int32_t(2));
}

QASR_TEST(SessionManagerLookup) {
    qasr::SessionManager mgr(10, 300000);
    std::string id;
    mgr.CreateSession(&id);
    qasr::SessionInfo info;
    qasr::Status s = mgr.LookupSession(id, &info);
    QASR_EXPECT(s.ok());
    QASR_EXPECT_EQ(info.session_id, id);
    QASR_EXPECT(info.state == qasr::SessionState::kCreated);
}

QASR_TEST(SessionManagerCloseSession) {
    qasr::SessionManager mgr(10, 300000);
    std::string id;
    mgr.CreateSession(&id);
    qasr::Status s = mgr.CloseSession(id);
    QASR_EXPECT(s.ok());

    qasr::SessionInfo info;
    mgr.LookupSession(id, &info);
    QASR_EXPECT(info.state == qasr::SessionState::kClosed);
}

QASR_TEST(SessionManagerTransition) {
    qasr::SessionManager mgr(10, 300000);
    std::string id;
    mgr.CreateSession(&id);
    QASR_EXPECT(mgr.TransitionSession(id, qasr::SessionState::kWarmed).ok());
    QASR_EXPECT(mgr.TransitionSession(id, qasr::SessionState::kRunning).ok());

    qasr::SessionInfo info;
    mgr.LookupSession(id, &info);
    QASR_EXPECT(info.state == qasr::SessionState::kRunning);
}

// --- Error ---

QASR_TEST(SessionManagerLookupNonexistent) {
    qasr::SessionManager mgr(10, 300000);
    qasr::SessionInfo info;
    qasr::Status s = mgr.LookupSession("nonexistent", &info);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(SessionManagerCloseNonexistent) {
    qasr::SessionManager mgr(10, 300000);
    qasr::Status s = mgr.CloseSession("nonexistent");
    QASR_EXPECT(!s.ok());
}

QASR_TEST(SessionManagerTransitionNonexistent) {
    qasr::SessionManager mgr(10, 300000);
    qasr::Status s = mgr.TransitionSession("nonexistent", qasr::SessionState::kRunning);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(SessionManagerInvalidTransition) {
    qasr::SessionManager mgr(10, 300000);
    std::string id;
    mgr.CreateSession(&id);
    // Skip from Created directly to Running (should fail)
    qasr::Status s = mgr.TransitionSession(id, qasr::SessionState::kRunning);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(SessionManagerCreateNullOutput) {
    qasr::SessionManager mgr(10, 300000);
    qasr::Status s = mgr.CreateSession(nullptr);
    QASR_EXPECT(!s.ok());
}

QASR_TEST(SessionManagerLookupNullOutput) {
    qasr::SessionManager mgr(10, 300000);
    std::string id;
    mgr.CreateSession(&id);
    qasr::Status s = mgr.LookupSession(id, nullptr);
    QASR_EXPECT(!s.ok());
}

// --- Extreme: max sessions ---

QASR_TEST(SessionManagerMaxSessions) {
    qasr::SessionManager mgr(2, 300000);
    std::string id1, id2, id3;
    QASR_EXPECT(mgr.CreateSession(&id1).ok());
    QASR_EXPECT(mgr.CreateSession(&id2).ok());
    qasr::Status s = mgr.CreateSession(&id3);
    QASR_EXPECT(!s.ok());  // Should reject
}

QASR_TEST(SessionManagerActiveCountAfterClose) {
    qasr::SessionManager mgr(10, 300000);
    std::string id;
    mgr.CreateSession(&id);
    QASR_EXPECT_EQ(mgr.active_count(), int32_t(1));
    mgr.CloseSession(id);
    QASR_EXPECT_EQ(mgr.active_count(), int32_t(0));
}

QASR_TEST(SessionManagerForceCloseFromCreated) {
    qasr::SessionManager mgr(10, 300000);
    std::string id;
    mgr.CreateSession(&id);
    qasr::Status s = mgr.TransitionSession(id, qasr::SessionState::kClosed);
    QASR_EXPECT(s.ok());
}
