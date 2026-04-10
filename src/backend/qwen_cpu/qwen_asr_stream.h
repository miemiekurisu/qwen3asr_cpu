/*
 * Internal helpers for streaming text reconciliation.
 *
 * The public streaming API is append-only.  Once a token has been emitted,
 * later decoder revisions must not replay the same committed span.
 */

#ifndef QWEN_ASR_STREAM_H
#define QWEN_ASR_STREAM_H

#include <string.h>

static inline int qwen_stream_skip_recent_duplicate_prefix(
    const int *emitted_tokens,
    int n_emitted_tokens,
    const int *candidate_tokens,
    int emit_start,
    int candidate_len,
    int min_match_tokens,
    int max_match_tokens,
    int lookback_tokens) {
    if (!emitted_tokens || !candidate_tokens) return emit_start;
    if (n_emitted_tokens <= 0 || emit_start < 0 || candidate_len <= emit_start) return emit_start;
    if (min_match_tokens <= 0 || max_match_tokens < min_match_tokens) return emit_start;

    int search_start = 0;
    if (lookback_tokens > 0 && n_emitted_tokens > lookback_tokens) {
        search_start = n_emitted_tokens - lookback_tokens;
    }

    int start = emit_start;
    while (candidate_len - start >= min_match_tokens) {
        int best = 0;
        int max_match = candidate_len - start;
        if (max_match > max_match_tokens) max_match = max_match_tokens;
        if (max_match > n_emitted_tokens) max_match = n_emitted_tokens;

        for (int k = max_match; k >= min_match_tokens; k--) {
            for (int pos = search_start; pos + k <= n_emitted_tokens; pos++) {
                if (memcmp(emitted_tokens + pos,
                           candidate_tokens + start,
                           (size_t)k * sizeof(int)) == 0) {
                    best = k;
                    break;
                }
            }
            if (best > 0) break;
        }

        if (best <= 0) break;
        start += best;
    }

    return start;
}

#endif
