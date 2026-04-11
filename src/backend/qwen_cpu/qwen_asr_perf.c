#include "qwen_asr_perf.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

#define QWEN_ENC_QKV_PACK_MIN_SEQ_DEFAULT 8
#define QWEN_ENC_QKV_SHAPE_AUTO_FALLBACK_SEQ_DEFAULT 96
#define QWEN_ENC_QKV_SHAPE_AUTO_FALLBACK_DMODEL_DEFAULT 1024

typedef struct {
    qwen_enc_qkv_policy_t policy;
    int pack_min_seq;
    int shape_auto_fallback_seq;
    int shape_auto_fallback_d_model;
    int initialized;
} qwen_enc_qkv_policy_config_t;

static qwen_enc_qkv_policy_config_t g_qkv_policy_config = {
    QWEN_ENC_QKV_POLICY_BEST,
    QWEN_ENC_QKV_PACK_MIN_SEQ_DEFAULT,
    QWEN_ENC_QKV_SHAPE_AUTO_FALLBACK_SEQ_DEFAULT,
    QWEN_ENC_QKV_SHAPE_AUTO_FALLBACK_DMODEL_DEFAULT,
    0,
};

static int ascii_iequals(const char *lhs, const char *rhs) {
    while (*lhs != '\0' && *rhs != '\0') {
        if (tolower((unsigned char)*lhs) != tolower((unsigned char)*rhs)) {
            return 0;
        }
        ++lhs;
        ++rhs;
    }
    return *lhs == '\0' && *rhs == '\0';
}

static int parse_positive_env(const char *name, int fallback) {
    const char *text = getenv(name);
    char *end = NULL;
    long value = 0;

    if (!text || *text == '\0') {
        return fallback;
    }

    value = strtol(text, &end, 10);
    if (!end || *end != '\0' || value <= 0 || value > (1L << 20)) {
        return fallback;
    }
    return (int)value;
}

static qwen_enc_qkv_policy_t parse_policy_env(const char *text) {
    if (!text || *text == '\0') {
        return QWEN_ENC_QKV_POLICY_BEST;
    }
    if (ascii_iequals(text, "best") || ascii_iequals(text, "default")) {
        return QWEN_ENC_QKV_POLICY_BEST;
    }
    if (ascii_iequals(text, "separate") || ascii_iequals(text, "force_separate")) {
        return QWEN_ENC_QKV_POLICY_FORCE_SEPARATE;
    }
    if (ascii_iequals(text, "packed") || ascii_iequals(text, "force_packed")) {
        return QWEN_ENC_QKV_POLICY_FORCE_PACKED;
    }
    if (ascii_iequals(text, "shape_auto") || ascii_iequals(text, "auto")) {
        return QWEN_ENC_QKV_POLICY_SHAPE_AUTO;
    }
    return QWEN_ENC_QKV_POLICY_BEST;
}

static void ensure_qkv_policy_config(void) {
    if (g_qkv_policy_config.initialized) {
        return;
    }

    g_qkv_policy_config.policy = parse_policy_env(getenv("QWEN_ENC_QKV_POLICY"));
    g_qkv_policy_config.pack_min_seq = parse_positive_env(
        "QWEN_ENC_QKV_PACK_MIN_SEQ", QWEN_ENC_QKV_PACK_MIN_SEQ_DEFAULT);
    g_qkv_policy_config.shape_auto_fallback_seq = parse_positive_env(
        "QWEN_ENC_QKV_SHAPE_AUTO_FALLBACK_SEQ",
        QWEN_ENC_QKV_SHAPE_AUTO_FALLBACK_SEQ_DEFAULT);
    g_qkv_policy_config.shape_auto_fallback_d_model = parse_positive_env(
        "QWEN_ENC_QKV_SHAPE_AUTO_FALLBACK_DMODEL",
        QWEN_ENC_QKV_SHAPE_AUTO_FALLBACK_DMODEL_DEFAULT);
    g_qkv_policy_config.initialized = 1;
}

qwen_enc_qkv_policy_t qwen_get_encoder_qkv_policy(void) {
    ensure_qkv_policy_config();
    return g_qkv_policy_config.policy;
}

qwen_enc_qkv_impl_t qwen_select_encoder_qkv_impl(qwen_enc_qkv_policy_t policy,
                                                 int seq_len,
                                                 int d_model,
                                                 int has_packed_weights) {
    ensure_qkv_policy_config();

    if (!has_packed_weights || seq_len < g_qkv_policy_config.pack_min_seq) {
        return QWEN_ENC_QKV_IMPL_SEPARATE;
    }

    switch (policy) {
    case QWEN_ENC_QKV_POLICY_FORCE_SEPARATE:
        return QWEN_ENC_QKV_IMPL_SEPARATE;
    case QWEN_ENC_QKV_POLICY_FORCE_PACKED:
        return QWEN_ENC_QKV_IMPL_PACKED;
    case QWEN_ENC_QKV_POLICY_SHAPE_AUTO:
        if (seq_len >= g_qkv_policy_config.shape_auto_fallback_seq &&
            d_model >= g_qkv_policy_config.shape_auto_fallback_d_model) {
            return QWEN_ENC_QKV_IMPL_SEPARATE;
        }
        return QWEN_ENC_QKV_IMPL_PACKED;
    case QWEN_ENC_QKV_POLICY_BEST:
    default:
        return QWEN_ENC_QKV_IMPL_PACKED;
    }
}

const char *qwen_encoder_qkv_policy_name(qwen_enc_qkv_policy_t policy) {
    switch (policy) {
    case QWEN_ENC_QKV_POLICY_FORCE_SEPARATE:
        return "force_separate";
    case QWEN_ENC_QKV_POLICY_FORCE_PACKED:
        return "force_packed";
    case QWEN_ENC_QKV_POLICY_SHAPE_AUTO:
        return "shape_auto";
    case QWEN_ENC_QKV_POLICY_BEST:
    default:
        return "best";
    }
}

const char *qwen_encoder_qkv_impl_name(qwen_enc_qkv_impl_t impl) {
    switch (impl) {
    case QWEN_ENC_QKV_IMPL_PACKED:
        return "packed";
    case QWEN_ENC_QKV_IMPL_SEPARATE:
    default:
        return "separate";
    }
}

int qwen_x86_cpu_supports_avx2_fma(void) {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int cpu_info[4] = {0, 0, 0, 0};
    int osxsave = 0;
    int avx = 0;
    int fma = 0;
    int avx2 = 0;
    unsigned __int64 xcr0 = 0;

    __cpuid(cpu_info, 0);
    if (cpu_info[0] < 7) {
        return 0;
    }

    __cpuidex(cpu_info, 1, 0);
    osxsave = (cpu_info[2] & (1 << 27)) != 0;
    avx = (cpu_info[2] & (1 << 28)) != 0;
    fma = (cpu_info[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma) {
        return 0;
    }

    xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return 0;
    }

    __cpuidex(cpu_info, 7, 0);
    avx2 = (cpu_info[1] & (1 << 5)) != 0;
    return avx2 ? 1 : 0;
#elif (defined(__GNUC__) || defined(__clang__)) &&
      (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}