/*
 * qwen_asr_utf8.c - UTF-8 string utilities for streaming decoder output.
 *
 * The Qwen3 BPE tokenizer decodes to raw UTF-8 byte strings, but a
 * single user-visible Chinese character can be split across 1-3 BPE
 * tokens.  When the streaming decoder cuts mid-character (VAD
 * early-stop, max_new_tokens cap, or recovery reset) the resulting
 * string has a partial UTF-8 tail that renders as garbled bytes in
 * the UI.  This file provides qwen_utf8_truncate() which trims back
 * to the last complete character boundary so the tail is never
 * invalid.
 *
 * The implementation is intentionally allocation-free and only
 * scans backward from the end of the string, so callers can run it
 * on every chunk emission with no measurable cost.
 */
#include "qwen_asr.h"

#include <stddef.h>
#include <stdint.h>

/* How many bytes a UTF-8 leading byte in the range [0xC0, 0xFD]
 * promises for the current code point.  ASCII (0x00-0x7F) is 1 byte.
 * 0x80-0xBF is a continuation byte and never starts a code point.
 * 0xFE/0xFF are invalid leading bytes. */
static size_t qwen_utf8_lead_len(unsigned char lead) {
    if (lead < 0x80) return 1;             /* ASCII */
    if (lead < 0xC0) return 0;             /* continuation byte, not a lead */
    if (lead < 0xE0) return 2;             /* 110xxxxx -> 2 bytes total */
    if (lead < 0xF0) return 3;             /* 1110xxxx -> 3 bytes total */
    if (lead < 0xF8) return 4;             /* 11110xxx -> 4 bytes total (rare) */
    return 0;                              /* invalid leading byte */
}

size_t qwen_utf8_truncate(char *s, size_t len) {
    if (s == NULL || len == 0) return 0;
    /* Find the start of the trailing UTF-8 code point.  We scan at
     * most 4 bytes back, which is the maximum possible lead length
     * for valid UTF-8.  Anything beyond that means the previous
     * character was already malformed — but in that case we still
     * leave it alone (we only trim the tail). */
    size_t scan = (len >= 4) ? 4 : len;
    size_t start = len;
    int found = 0;
    for (size_t i = 1; i <= scan; ++i) {
        unsigned char b = (unsigned char)s[len - i];
        size_t lead_len = qwen_utf8_lead_len(b);
        if (lead_len == 0) continue;       /* continuation byte, keep scanning */
        if (i == lead_len) {
            /* Tail ends on a complete code point: no truncation. */
            return len;
        }
        if (i < lead_len) {
            /* Tail ends mid-code-point: trim back to the start of
             * this (incomplete) code point. */
            start = len - i;
            found = 1;
            break;
        }
        /* i > lead_len: the leading byte at (len - i) claims
         * `lead_len` bytes total, ending at (len - i + lead_len - 1).
         * The trailing window extends beyond that, so the bytes
         * from (len - i + lead_len) to (len - 1) are orphan
         * continuation bytes (malformed input) — trim from
         * (len - i + lead_len). */
        start = len - i + lead_len;
        found = 1;
        break;
    }
    if (!found) {
        /* No leading byte found in the last 4 bytes: the entire
         * trailing window is continuation bytes with no matching
         * lead (malformed input).  Walk back further in 4-byte
         * strides to find the real lead. */
        size_t walk = len;
        while (walk > 0) {
            unsigned char b = (unsigned char)s[walk - 1];
            size_t lead_len = qwen_utf8_lead_len(b);
            if (lead_len != 0) {
                /* Found a lead.  If it claims the bytes from walk
                 * to the end form a complete code point, keep it. */
                if (walk + lead_len - 1 == len) {
                    return len;
                }
                /* Lead is mid-string; trim starts at walk. */
                break;
            }
            walk--;
        }
        start = walk;
    }
    if (start >= len) return len;          /* already aligned */
    s[start] = '\0';
    return start;
}
