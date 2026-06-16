#include "qasr/scheduler/scheduler.h"
#include "qasr/engine/asr_engine.h"
#include <chrono>
#include <cstdio>
#include <memory>

namespace qasr {

Status SessionFairQueue::Push(const SegmentJob & job) {
    std::lock_guard<std::mutex> lock(mu_);
    queue_.push(job);
    session_count_[job.session_id]++;
    return OkStatus();
}

bool SessionFairQueue::Pop(SegmentJob & out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (queue_.empty()) return false;
    out = queue_.front();
    queue_.pop();
    session_count_[out.session_id]--;
    if (session_count_[out.session_id] == 0) {
        session_count_.erase(out.session_id);
    }
    return true;
}

bool SessionFairQueue::SessionHasPending(std::uint64_t session_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return session_count_.count(session_id) > 0;
}

GpuScheduler::GpuScheduler() = default;

GpuScheduler::~GpuScheduler() {
    Shutdown();
}

void GpuScheduler::SetEngine(AsrEngine * engine) {
    engine_ = engine;
}

Status GpuScheduler::Submit(const SegmentJob & job) {
    if (shutdown_.load()) {
        return Status(StatusCode::kFailedPrecondition, "scheduler shut down");
    }

    /* Backpressure: reject if session already has a pending job in queue.
     * This ensures per-session ordering: at most one queued job per session,
     * plus at most one inflight.  When inflight completes, the next queued
     * job is picked up by the worker. */
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (realtime_queue_.SessionHasPending(job.session_id) ||
            batch_queue_.SessionHasPending(job.session_id)) {
            return Status(StatusCode::kResourceExhausted,
                          "session already has pending job in queue");
        }
    }

    if (job.realtime) {
        if (realtime_queue_.Full(max_pending_per_session_ * max_sessions_)) {
            return Status(StatusCode::kResourceExhausted, "realtime queue full");
        }
        realtime_queue_.Push(job);
    } else {
        if (batch_queue_.Full(max_batch_queue_)) {
            return Status(StatusCode::kResourceExhausted, "batch queue full");
        }
        batch_queue_.Push(job);
    }
    cv_.notify_one();
    return OkStatus();
}

SegmentResult GpuScheduler::SubmitAndAwait(const SegmentJob & job) {
    if (shutdown_.load()) {
        SegmentResult res;
        res.session_id = job.session_id;
        res.segment_id = job.segment_id;
        res.status = Status(StatusCode::kFailedPrecondition, "scheduler shut down");
        return res;
    }

    /* Per-job synchronization: wait for this specific segment to complete.
     * Uses shared_ptr so the state survives even if SubmitAndAwait returns
     * before ExecuteJob calls the callback — the worker thread owns a ref. */
    struct WaitState {
        std::mutex mu;
        std::condition_variable cv;
        SegmentResult result;
        bool done = false;
    };
    auto state = std::make_shared<WaitState>();
    state->result.session_id = job.session_id;
    state->result.segment_id = job.segment_id;
    state->result.status = Status(StatusCode::kInternal,
                                  "timed out waiting for scheduler result");

    /* Create a mutable copy with the per-job waiter callback.
     * The shared_ptr ensures the state is alive when the worker calls back,
     * even if this function has already returned. */
    SegmentJob mutable_job = job;
    mutable_job.on_job_complete = [state](const SegmentResult & res) {
        std::lock_guard<std::mutex> lock(state->mu);
        state->result = res;
        state->done = true;
        state->cv.notify_one();
    };

    Status st = Submit(mutable_job);
    if (!st.ok()) {
        SegmentResult res;
        res.session_id = job.session_id;
        res.segment_id = job.segment_id;
        res.status = st;
        return res;
    }

    /* Wait until the done flag is set (max 60 s — covers long segments). */
    std::unique_lock<std::mutex> lock(state->mu);
    state->cv.wait_for(lock, std::chrono::seconds(60),
                       [&state]() { return state->done; });

    return state->result;
}

void GpuScheduler::SetWorker(int worker_count) {
    (void)worker_count;
}

void GpuScheduler::SetCallback(JobCompleteCallback cb) {
    on_complete_ = std::move(cb);
}

void GpuScheduler::Start() {
    if (running_.exchange(true)) return;
    workers_.emplace_back([this]() { WorkerLoop(); });
}

void GpuScheduler::Shutdown() {
    if (!running_.exchange(false)) return;
    shutdown_.store(true);
    cv_.notify_all();
    for (auto & w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

void GpuScheduler::WorkerLoop() {
    while (!shutdown_.load()) {
        SegmentJob job;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait_for(lock, std::chrono::milliseconds(500), [this]() {
                return shutdown_.load() || !realtime_queue_.empty() || !batch_queue_.empty();
            });
            if (shutdown_.load()) break;

            /* Realtime queue has priority over batch queue (§7.3). */
            if (!realtime_queue_.empty()) {
                realtime_queue_.Pop(job);
            } else if (!batch_queue_.empty()) {
                batch_queue_.Pop(job);
            } else {
                continue;
            }

            /* Mark session as inflight. */
            inflight_sessions_.insert(job.session_id);
        }

        /* Execute outside lock (§20.1). */
        ExecuteJob(job);

        {
            std::lock_guard<std::mutex> lock(mu_);
            inflight_sessions_.erase(job.session_id);
        }

        /* Notify — there may be queued jobs for this session. */
        cv_.notify_one();
    }
}

void GpuScheduler::ExecuteJob(SegmentJob & job) {
    SegmentResult res;
    res.session_id = job.session_id;
    res.segment_id = job.segment_id;

    /* Extract per-job callback once — it's consumed after this call. */
    JobCompleteCallback job_cb = std::move(job.on_job_complete);

    auto fire = [this, &res, &job_cb]() {
        if (job_cb) job_cb(res);
        if (on_complete_) on_complete_(res);
    };

    /* Wrap entire execution in try-catch to guarantee the callback fires
     * even if TranscribeSegment throws (e.g. CUDA driver error).
     * IMPORTANT: fire() is called AFTER the try-catch, never inside, to
     * avoid re-throwing in the catch handler (which would propagate
     * uncaught through WorkerLoop → std::terminate). */
    bool transcribe_ok = false;
    try {
        if (!engine_ || job.samples.empty()) {
            res.status = Status(StatusCode::kFailedPrecondition,
                                "no engine or empty samples");
            transcribe_ok = true;
        } else {
            /* Use pre-existing engine session if provided, otherwise create
             * a transient one.  Pre-existing sessions are useful for realtime
             * workers that maintain persistent GPU state (KV cache, etc.). */
            const bool use_existing = job.engine_session_id != 0;
            std::uint64_t sid = 0;

            if (!use_existing) {
                SessionOptions opts;
                opts.language = job.language;
                opts.prompt = job.prompt;
                Status st = engine_->CreateSession(opts, sid);
                if (!st.ok()) {
                    res.status = st;
                    transcribe_ok = true;
                }
            } else {
                sid = job.engine_session_id;
            }

            if (!transcribe_ok) {
                /* Transcribe via engine. */
                AsrSegmentResult seg = engine_->TranscribeSegment(
                    sid, job.samples, job.sample_rate);

                res.status = seg.status;
                res.text = seg.text;
                res.encode_ms = seg.encode_ms;
                res.decode_ms = seg.decode_ms;
                res.total_ms = seg.total_ms;
                res.tokens = seg.text_tokens;

                if (!use_existing) {
                    engine_->CloseSession(sid);
                }
                transcribe_ok = true;
            }
        }
    } catch (const std::exception & e) {
        res.status = Status(StatusCode::kInternal,
                            std::string("scheduler ExecuteJob exception: ") + e.what());
        fprintf(stderr, "scheduler: ExecuteJob exception: %s\n", e.what());
    } catch (...) {
        res.status = Status(StatusCode::kInternal,
                            "scheduler ExecuteJob unknown exception");
    }

    /* fire() is called exactly once, outside the try-catch, so
     * any exception it throws propagates to WorkerLoop's outer scope. */
    fire();
}

}  // namespace qasr
