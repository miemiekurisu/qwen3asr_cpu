/*
 * qwen_asr_kernels.c - Math kernels for Qwen3-ASR inference
 * Adapted from voxtral-realtime project.
 */

#include "qwen_asr_kernels.h"
#include "qwen_asr_kernels_impl.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#if (defined(__AVX512F__) || defined(__AVX2__)) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#else
#include <unistd.h>
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================
 * Thread Pool
 * ======================================================================== */

#define QWEN_MAX_THREADS 16

typedef void (*parallel_fn_t)(int tid, int n_threads, void *arg);

static struct {
    pthread_t threads[QWEN_MAX_THREADS - 1];
    int tids[QWEN_MAX_THREADS - 1];
    int n_threads;
    int shutdown;

    parallel_fn_t fn;
    void *arg;
    int generation;

    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    int n_done;
} tp = {
    .n_threads = 1,
    .shutdown = 0,
    .generation = 0,
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond_work = PTHREAD_COND_INITIALIZER,
    .cond_done = PTHREAD_COND_INITIALIZER,
};

static void *worker_loop(void *arg) {
    int tid = *(int *)arg;
    int my_gen = 0;

    for (;;) {
        pthread_mutex_lock(&tp.mutex);
        while (tp.generation == my_gen && !tp.shutdown)
            pthread_cond_wait(&tp.cond_work, &tp.mutex);
        if (tp.shutdown) {
            pthread_mutex_unlock(&tp.mutex);
            return NULL;
        }
        my_gen = tp.generation;
        parallel_fn_t fn = tp.fn;
        void *a = tp.arg;
        int nt = tp.n_threads;
        pthread_mutex_unlock(&tp.mutex);

        fn(tid, nt, a);

        pthread_mutex_lock(&tp.mutex);
        if (++tp.n_done >= tp.n_threads - 1)
            pthread_cond_signal(&tp.cond_done);
        pthread_mutex_unlock(&tp.mutex);
    }
}

void qwen_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > QWEN_MAX_THREADS) n = QWEN_MAX_THREADS;

    /* Shutdown existing workers */
    if (tp.n_threads > 1) {
        pthread_mutex_lock(&tp.mutex);
        tp.shutdown = 1;
        pthread_cond_broadcast(&tp.cond_work);
        pthread_mutex_unlock(&tp.mutex);
        for (int i = 0; i < tp.n_threads - 1; i++)
            pthread_join(tp.threads[i], NULL);
        tp.shutdown = 0;
        tp.generation = 0;
    }

    tp.n_threads = n;
    if (n <= 1) return;

    for (int i = 0; i < n - 1; i++) {
        tp.tids[i] = i + 1;
        pthread_create(&tp.threads[i], NULL, worker_loop, &tp.tids[i]);
    }

    if (qwen_verbose >= 2)
        fprintf(stderr, "Thread pool: %d threads\n", n);
}

int qwen_get_num_cpus(void) {
#ifdef __APPLE__
    int n = 0;
    size_t len = sizeof(n);
    sysctlbyname("hw.ncpu", &n, &len, NULL, 0);
    return n > 0 ? n : 1;
#else
    int n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? n : 1;
#endif
}

/* Dispatch work to all threads; main thread is tid=0 */
static void parallel_for(parallel_fn_t fn, void *arg) {
    if (tp.n_threads <= 1) {
        fn(0, 1, arg);
        return;
    }

    pthread_mutex_lock(&tp.mutex);
    tp.fn = fn;
    tp.arg = arg;
    tp.n_done = 0;
    tp.generation++;
    pthread_cond_broadcast(&tp.cond_work);
    pthread_mutex_unlock(&tp.mutex);

    fn(0, tp.n_threads, arg);

    pthread_mutex_lock(&tp.mutex);
    while (tp.n_done < tp.n_threads - 1)
        pthread_cond_wait(&tp.cond_done, &tp.mutex);
    pthread_mutex_unlock(&tp.mutex);
}

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void qwen_add_inplace(float *a, const float *b, int n) {
    /*
     * OPT: SIMD vectorized residual addition.
     * Rationale: Called on every residual connection in both encoder and decoder.
     *   For dim=2048, the scalar loop has 2048 iterations of load-add-store.
     *   NEON processes 16 floats per iteration (4x unroll), AVX processes 32.
     * Method: Direct SIMD load/add/store with 4x unrolling.
     * Effect: ~4x throughput improvement over scalar for large vectors.
     */
#ifdef __ARM_NEON
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(a + i,      vaddq_f32(vld1q_f32(a + i),      vld1q_f32(b + i)));
        vst1q_f32(a + i + 4,  vaddq_f32(vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4)));
        vst1q_f32(a + i + 8,  vaddq_f32(vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8)));
        vst1q_f32(a + i + 12, vaddq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12)));
    }
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(a + i, vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i)));
    }
    for (; i < n; i++) a[i] += b[i];
#elif defined(__AVX2__)
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm256_storeu_ps(a+i,    _mm256_add_ps(_mm256_loadu_ps(a+i),    _mm256_loadu_ps(b+i)));
        _mm256_storeu_ps(a+i+8,  _mm256_add_ps(_mm256_loadu_ps(a+i+8),  _mm256_loadu_ps(b+i+8)));
        _mm256_storeu_ps(a+i+16, _mm256_add_ps(_mm256_loadu_ps(a+i+16), _mm256_loadu_ps(b+i+16)));
        _mm256_storeu_ps(a+i+24, _mm256_add_ps(_mm256_loadu_ps(a+i+24), _mm256_loadu_ps(b+i+24)));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(a+i, _mm256_add_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i)));
    }
    for (; i < n; i++) a[i] += b[i];
#else
    for (int i = 0; i < n; i++) a[i] += b[i];
#endif
}

void qwen_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) a[i] *= b[i];
}

void qwen_scale(float *x, float s, int n) {
    for (int i = 0; i < n; i++) x[i] *= s;
}

void qwen_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

void qwen_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K, 1.0f, A, K, B, K, 0.0f, C, N);
#else
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void qwen_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }
#endif
}

void qwen_linear_nobias(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim) {
    qwen_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

/* Convert bf16 buffer to f32 buffer */
static void bf16_to_f32_buf(float *dst, const uint16_t *src, size_t n) {
    /*
     * OPT: SIMD vectorized BF16→F32 conversion.
     * Rationale: Called for every weight matrix during prefill (seq>1) to convert
     *   decoder BF16 weights to F32 for BLAS sgemm. For gate_up_fused (25M values),
     *   scalar conversion costs ~50ms per layer.
     * Method: NEON vshll_n_u16 shifts 8 bf16 values at once; AVX _mm256_slli_epi32
     *   for x86. Both avoid per-element type punning overhead.
     * Effect: ~4-8x throughput over scalar, saving ~40ms on prefill.
     */
#ifdef __ARM_NEON
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint16x8_t bf0 = vld1q_u16(src + i);
        uint16x8_t bf1 = vld1q_u16(src + i + 8);
        vst1q_f32(dst + i,      vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf0), 16)));
        vst1q_f32(dst + i + 4,  vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf0), 16)));
        vst1q_f32(dst + i + 8,  vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf1), 16)));
        vst1q_f32(dst + i + 12, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf1), 16)));
    }
    for (; i + 8 <= n; i += 8) {
        uint16x8_t bf = vld1q_u16(src + i);
        vst1q_f32(dst + i,     vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf), 16)));
        vst1q_f32(dst + i + 4, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf), 16)));
    }
    {
        uint32_t *d = (uint32_t *)(void *)dst;
        for (; i < n; i++) d[i] = ((uint32_t)src[i]) << 16;
    }
#elif defined(__AVX2__)
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i raw = _mm256_loadu_si256((const __m256i *)(src + i));
        __m256i lo = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(raw)), 16);
        __m256i hi = _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(raw, 1)), 16);
        _mm256_storeu_ps(dst + i,     _mm256_castsi256_ps(lo));
        _mm256_storeu_ps(dst + i + 8, _mm256_castsi256_ps(hi));
    }
    {
        uint32_t *d = (uint32_t *)(void *)dst;
        for (; i < n; i++) d[i] = ((uint32_t)src[i]) << 16;
    }
#else
    uint32_t *d = (uint32_t *)(void *)dst;
    for (size_t i = 0; i < n; i++)
        d[i] = ((uint32_t)src[i]) << 16;
#endif
}

/* Threaded BF16→F32 conversion for large weight matrices */
typedef struct {
    float *dst;
    const uint16_t *src;
    size_t n;
} bf16_cvt_task_t;

static void bf16_cvt_worker(int tid, int n_threads, void *arg) {
    bf16_cvt_task_t *t = (bf16_cvt_task_t *)arg;
    size_t chunk = (t->n + (size_t)n_threads - 1) / (size_t)n_threads;
    size_t start = (size_t)tid * chunk;
    size_t end = start + chunk;
    if (end > t->n) end = t->n;
    if (start >= end) return;
    bf16_to_f32_buf(t->dst + start, t->src + start, end - start);
}

static void bf16_to_f32_buf_threaded(float *dst, const uint16_t *src, size_t n) {
    if (tp.n_threads <= 1 || n < 65536) {
        bf16_to_f32_buf(dst, src, n);
        return;
    }
    bf16_cvt_task_t task = { .dst = dst, .src = src, .n = n };
    parallel_for(bf16_cvt_worker, &task);
}

/* Reusable scratch buffer for bf16->f32 conversion */
static float *bf16_scratch = NULL;
static size_t bf16_scratch_cap = 0;

static float *bf16_get_scratch(size_t n) {
    if (n > bf16_scratch_cap) {
        free(bf16_scratch);
        bf16_scratch = (float *)malloc(n * sizeof(float));
        bf16_scratch_cap = bf16_scratch ? n : 0;
    }
    return bf16_scratch;
}

typedef struct {
    const uint16_t *src;
    size_t n;
    float *dst_f32;
} bf16_cache_entry_t;

static bf16_cache_entry_t *bf16_cache = NULL;
static int bf16_cache_len = 0;
static int bf16_cache_cap = 0;
static size_t bf16_cache_bytes = 0;
static size_t bf16_cache_limit_bytes = 0;
static int bf16_cache_limit_init = 0;

static void bf16_cache_init_limit(void) {
    if (bf16_cache_limit_init) return;
    bf16_cache_limit_init = 1;

    /* Default OFF. Override with QWEN_BF16_CACHE_MB=<n> to enable. */
    unsigned long long mb = 0;
    const char *env = getenv("QWEN_BF16_CACHE_MB");
    if (env && env[0] != '\0') {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) mb = v;
    }
    bf16_cache_limit_bytes = (size_t)(mb * 1024ULL * 1024ULL);

    if (qwen_verbose >= 2) {
        fprintf(stderr, "BF16 cache: limit=%llu MB\n", mb);
    }
}

static const float *bf16_get_cached_f32(const uint16_t *src, size_t n) {
    bf16_cache_init_limit();

    for (int i = 0; i < bf16_cache_len; i++) {
        if (bf16_cache[i].src == src && bf16_cache[i].n == n) {
            return bf16_cache[i].dst_f32;
        }
    }

    if (bf16_cache_limit_bytes == 0) return NULL;

    size_t bytes = n * sizeof(float);
    if (bytes > bf16_cache_limit_bytes) return NULL;
    if (bf16_cache_bytes + bytes > bf16_cache_limit_bytes) return NULL;

    float *dst = (float *)malloc(bytes);
    if (!dst) return NULL;
    bf16_to_f32_buf(dst, src, n);

    if (bf16_cache_len == bf16_cache_cap) {
        int new_cap = bf16_cache_cap > 0 ? bf16_cache_cap * 2 : 256;
        bf16_cache_entry_t *tmp = (bf16_cache_entry_t *)realloc(
            bf16_cache, (size_t)new_cap * sizeof(bf16_cache_entry_t));
        if (!tmp) {
            free(dst);
            return NULL;
        }
        bf16_cache = tmp;
        bf16_cache_cap = new_cap;
    }

    bf16_cache[bf16_cache_len].src = src;
    bf16_cache[bf16_cache_len].n = n;
    bf16_cache[bf16_cache_len].dst_f32 = dst;
    bf16_cache_len++;
    bf16_cache_bytes += bytes;
    return dst;
}

static const float *bf16_get_f32_view(const uint16_t *src, size_t n) {
    const float *cached = bf16_get_cached_f32(src, n);
    if (cached) return cached;

    float *scratch = bf16_get_scratch(n);
    if (!scratch) return NULL;
    bf16_to_f32_buf_threaded(scratch, src, n);
    return scratch;
}

/*
 * Fused BF16 matvec: y[out_dim] = W_bf16[out_dim, in_dim] @ x[in_dim] + bias
 * Processes 2 output rows at a time to amortize x vector loads.
 */
static void bf16_matvec_fused(float *y, const float *x, const uint16_t *W_bf16,
                               const float *bias, int in_dim, int out_dim) {
    qwen_bf16_matvec_fused_impl(y, x, W_bf16, bias, in_dim, out_dim);
}

/* Threaded matvec: split output rows across threads */
typedef struct {
    float *y;
    const float *x;
    const uint16_t *W_bf16;
    const float *bias;
    int in_dim;
    int out_dim;
} matvec_task_t;

static void matvec_worker(int tid, int n_threads, void *arg) {
    matvec_task_t *t = (matvec_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) return;

    bf16_matvec_fused(t->y + start, t->x,
                      t->W_bf16 + (size_t)start * t->in_dim,
                      t->bias ? t->bias + start : NULL,
                      t->in_dim, end - start);
}

static void bf16_matvec_threaded(float *y, const float *x, const uint16_t *W_bf16,
                                  const float *bias, int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(y, x, W_bf16, bias, in_dim, out_dim);
        return;
    }
    matvec_task_t task = { y, x, W_bf16, bias, in_dim, out_dim };
    parallel_for(matvec_worker, &task);
}

typedef struct {
    float *q;
    float *k;
    float *v;
    const float *x;
    const uint16_t *Wq_bf16;
    const uint16_t *Wk_bf16;
    const uint16_t *Wv_bf16;
    int in_dim;
    int q_dim;
    int kv_dim;
    int total_dim;
} qkv_matvec_task_t;

static void qkv_matvec_worker(int tid, int n_threads, void *arg) {
    qkv_matvec_task_t *t = (qkv_matvec_task_t *)arg;
    int chunk = (t->total_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->total_dim) end = t->total_dim;
    if (start >= end) return;

    int q_end = t->q_dim;
    int k_end = q_end + t->kv_dim;
    int v_end = k_end + t->kv_dim;

    if (start < q_end) {
        int s = start;
        int e = end < q_end ? end : q_end;
        if (s < e) {
            bf16_matvec_fused(t->q + s, t->x,
                              t->Wq_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > q_end && start < k_end) {
        int s = start > q_end ? start - q_end : 0;
        int e_abs = end < k_end ? end : k_end;
        int e = e_abs - q_end;
        if (s < e) {
            bf16_matvec_fused(t->k + s, t->x,
                              t->Wk_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }

    if (end > k_end && start < v_end) {
        int s = start > k_end ? start - k_end : 0;
        int e_abs = end < v_end ? end : v_end;
        int e = e_abs - k_end;
        if (s < e) {
            bf16_matvec_fused(t->v + s, t->x,
                              t->Wv_bf16 + (size_t)s * t->in_dim,
                              NULL, t->in_dim, e - s);
        }
    }
}

void qwen_linear_nobias_bf16_qkv(float *q, float *k, float *v, const float *x,
                                 const uint16_t *Wq_bf16,
                                 const uint16_t *Wk_bf16,
                                 const uint16_t *Wv_bf16,
                                 int in_dim, int q_dim, int kv_dim) {
    if (tp.n_threads <= 1) {
        bf16_matvec_fused(q, x, Wq_bf16, NULL, in_dim, q_dim);
        bf16_matvec_fused(k, x, Wk_bf16, NULL, in_dim, kv_dim);
        bf16_matvec_fused(v, x, Wv_bf16, NULL, in_dim, kv_dim);
        return;
    }

    qkv_matvec_task_t task = {
        .q = q,
        .k = k,
        .v = v,
        .x = x,
        .Wq_bf16 = Wq_bf16,
        .Wk_bf16 = Wk_bf16,
        .Wv_bf16 = Wv_bf16,
        .in_dim = in_dim,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .total_dim = q_dim + 2 * kv_dim,
    };
    parallel_for(qkv_matvec_worker, &task);
}

void qwen_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_threaded(y, x, W_bf16, NULL, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    qwen_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
}

/* Fused QKV prefill: converts Wq/Wk/Wv into one contiguous F32 block,
 * does a single BLAS sgemm, then splits the output into q/k/v.
 * Saves 2 BLAS call overheads per layer and reduces input matrix re-reads. */

typedef struct {
    float *W_fused;
    const uint16_t *Wq_bf16;
    const uint16_t *Wk_bf16;
    const uint16_t *Wv_bf16;
    size_t wq_n;
    size_t wk_n;
    size_t w_total;
} qkv_cvt_task_t;

static void qkv_cvt_worker(int tid, int n_threads, void *arg) {
    qkv_cvt_task_t *t = (qkv_cvt_task_t *)arg;
    size_t chunk = (t->w_total + (size_t)n_threads - 1) / (size_t)n_threads;
    size_t start = (size_t)tid * chunk;
    size_t end = start + chunk;
    if (end > t->w_total) end = t->w_total;
    if (start >= end) return;

    /* Map logical offset in fused buffer to correct source segment */
    size_t pos = start;
    while (pos < end) {
        const uint16_t *src;
        size_t seg_start, seg_end;
        if (pos < t->wq_n) {
            src = t->Wq_bf16;
            seg_start = 0;
            seg_end = t->wq_n;
        } else if (pos < t->wq_n + t->wk_n) {
            src = t->Wk_bf16 - t->wq_n;  /* offset so src[pos] is correct */
            seg_start = t->wq_n;
            seg_end = t->wq_n + t->wk_n;
        } else {
            src = t->Wv_bf16 - (t->wq_n + t->wk_n);
            seg_start = t->wq_n + t->wk_n;
            seg_end = t->w_total;
        }
        size_t run_end = end < seg_end ? end : seg_end;
        size_t run_len = run_end - pos;
        bf16_to_f32_buf(t->W_fused + pos, src + pos, run_len);
        pos = run_end;
    }
}

void qwen_linear_nobias_bf16_qkv_prefill(
    float *q, float *k, float *v, const float *x,
    const uint16_t *Wq_bf16, const uint16_t *Wk_bf16, const uint16_t *Wv_bf16,
    int seq_len, int in_dim, int q_dim, int kv_dim)
{
    /* seq=1 should use the decode-optimized fused matvec path instead */
    if (seq_len == 1) {
        qwen_linear_nobias_bf16_qkv(q, k, v, x, Wq_bf16, Wk_bf16, Wv_bf16,
                                    in_dim, q_dim, kv_dim);
        return;
    }

    int total_out = q_dim + 2 * kv_dim;
    size_t wq_n = (size_t)q_dim * in_dim;
    size_t wk_n = (size_t)kv_dim * in_dim;
    size_t w_total = wq_n + 2 * wk_n;

    /* Reuse scratch for fused weight [total_out, in_dim] in F32 */
    float *W_fused = bf16_get_scratch(w_total);
    if (!W_fused) return;

    /* Convert all three BF16 weight matrices in one threaded dispatch */
    if (tp.n_threads > 1) {
        qkv_cvt_task_t cvt = {
            .W_fused = W_fused,
            .Wq_bf16 = Wq_bf16,
            .Wk_bf16 = Wk_bf16,
            .Wv_bf16 = Wv_bf16,
            .wq_n = wq_n,
            .wk_n = wk_n,
            .w_total = w_total,
        };
        parallel_for(qkv_cvt_worker, &cvt);
    } else {
        bf16_to_f32_buf(W_fused, Wq_bf16, wq_n);
        bf16_to_f32_buf(W_fused + wq_n, Wk_bf16, wk_n);
        bf16_to_f32_buf(W_fused + wq_n + wk_n, Wv_bf16, wk_n);
    }

    /* Reusable scratch for fused output [seq_len, total_out] */
    static float *qkv_out = NULL;
    static size_t qkv_out_cap = 0;
    size_t out_n = (size_t)seq_len * total_out;
    if (out_n > qkv_out_cap) {
        free(qkv_out);
        qkv_out = (float *)malloc(out_n * sizeof(float));
        qkv_out_cap = qkv_out ? out_n : 0;
    }
    if (!qkv_out) return;

#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, total_out, in_dim,
                1.0f, x, in_dim, W_fused, in_dim,
                0.0f, qkv_out, total_out);
#else
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = qkv_out + s * total_out;
        for (int o = 0; o < total_out; o++) {
            const float *w_row = W_fused + (size_t)o * in_dim;
            float sum = 0.0f;
            for (int i = 0; i < in_dim; i++)
                sum += x_row[i] * w_row[i];
            y_row[o] = sum;
        }
    }
#endif

    /* Split fused output into separate q, k, v buffers */
    for (int s = 0; s < seq_len; s++) {
        const float *row = qkv_out + (size_t)s * total_out;
        memcpy(q + (size_t)s * q_dim,  row,                    (size_t)q_dim * sizeof(float));
        memcpy(k + (size_t)s * kv_dim, row + q_dim,            (size_t)kv_dim * sizeof(float));
        memcpy(v + (size_t)s * kv_dim, row + q_dim + kv_dim,   (size_t)kv_dim * sizeof(float));
    }
}

void qwen_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                      const float *b, int seq_len, int in_dim, int out_dim) {
    if (seq_len == 1) {
        bf16_matvec_threaded(y, x, W_bf16, b, in_dim, out_dim);
        return;
    }
    size_t n = (size_t)out_dim * in_dim;
    const float *W_f32 = bf16_get_f32_view(W_bf16, n);
    if (!W_f32) return;
    qwen_linear(y, x, W_f32, b, seq_len, in_dim, out_dim);
}

/* Find argmax over a range of output rows [start, end).
 * Uses 2-row processing to amortize x vector loads (same as bf16_matvec_fused). */
static void argmax_bf16_range(const float *x, const uint16_t *W_bf16,
                               int in_dim, int start, int end,
                               int *best_out, float *best_val_out) {
    qwen_argmax_bf16_range_impl(x, W_bf16, in_dim, start, end, best_out, best_val_out);
}

typedef struct {
    const float *x;
    const uint16_t *W_bf16;
    int in_dim;
    int out_dim;
    int best_idx[QWEN_MAX_THREADS];
    float best_val[QWEN_MAX_THREADS];
} argmax_task_t;

static void argmax_worker(int tid, int n_threads, void *arg) {
    argmax_task_t *t = (argmax_task_t *)arg;
    int chunk = (t->out_dim + n_threads - 1) / n_threads;
    int start = tid * chunk;
    int end = start + chunk;
    if (end > t->out_dim) end = t->out_dim;
    if (start >= end) {
        t->best_val[tid] = -1e30f;
        t->best_idx[tid] = 0;
        return;
    }
    argmax_bf16_range(t->x, t->W_bf16, t->in_dim, start, end,
                      &t->best_idx[tid], &t->best_val[tid]);
}

int qwen_argmax_matvec_bf16(const float *x, const uint16_t *W_bf16,
                             int in_dim, int out_dim) {
    if (tp.n_threads <= 1) {
        int best;
        float best_val;
        argmax_bf16_range(x, W_bf16, in_dim, 0, out_dim, &best, &best_val);
        return best;
    }

    argmax_task_t task;
    task.x = x;
    task.W_bf16 = W_bf16;
    task.in_dim = in_dim;
    task.out_dim = out_dim;
    parallel_for(argmax_worker, &task);

    int best = task.best_idx[0];
    float best_val = task.best_val[0];
    for (int i = 1; i < tp.n_threads; i++) {
        if (task.best_val[i] > best_val) {
            best_val = task.best_val[i];
            best = task.best_idx[i];
        }
    }
    return best;
}

void qwen_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                         int M, int K, int N) {
    if (M == 1) {
        bf16_matvec_threaded(C, A, B_bf16, NULL, K, N);
    } else {
        size_t n = (size_t)N * K;
        const float *B_f32 = bf16_get_f32_view(B_bf16, n);
        if (!B_f32) return;
        qwen_matmul_t(C, A, B_f32, M, K, N);
    }
}

/* ========================================================================
 * 2D Convolution (im2col + BLAS sgemm)
 * ======================================================================== */

/*
 * im2col: Unroll input patches into a column matrix for GEMM-based conv2d.
 * Input: [C_in, H_in, W_in]
 * Output columns: [C_in * kH * kW, H_out * W_out]
 */
static void im2col(const float *in, float *cols,
                   int c_in, int h_in, int w_in,
                   int kh, int kw, int stride, int padding,
                   int h_out, int w_out) {
    int col_len = h_out * w_out;
    for (int ic = 0; ic < c_in; ic++) {
        for (int ki = 0; ki < kh; ki++) {
            for (int kj = 0; kj < kw; kj++) {
                int col_row = (ic * kh + ki) * kw + kj;
                float *col_ptr = cols + (size_t)col_row * col_len;
                for (int oh = 0; oh < h_out; oh++) {
                    int ih = oh * stride - padding + ki;
                    for (int ow = 0; ow < w_out; ow++) {
                        int iw = ow * stride - padding + kj;
                        if (ih >= 0 && ih < h_in && iw >= 0 && iw < w_in) {
                            col_ptr[oh * w_out + ow] = in[ic * h_in * w_in + ih * w_in + iw];
                        } else {
                            col_ptr[oh * w_out + ow] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

void qwen_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int c_in, int c_out, int h_in, int w_in,
                 int kh, int kw, int stride, int padding) {
    int h_out = (h_in + 2 * padding - kh) / stride + 1;
    int w_out = (w_in + 2 * padding - kw) / stride + 1;
    int patch_size = c_in * kh * kw;
    int spatial_out = h_out * w_out;

    /* im2col: input -> column matrix [patch_size, spatial_out] */
    float *cols = (float *)malloc((size_t)patch_size * spatial_out * sizeof(float));
    im2col(in, cols, c_in, h_in, w_in, kh, kw, stride, padding, h_out, w_out);

    /* GEMM: weight[c_out, patch_size] @ cols[patch_size, spatial_out] = out[c_out, spatial_out] */
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                c_out, spatial_out, patch_size,
                1.0f, weight, patch_size, cols, spatial_out,
                0.0f, out, spatial_out);
#else
    for (int oc = 0; oc < c_out; oc++) {
        for (int s = 0; s < spatial_out; s++) {
            float sum = 0.0f;
            for (int p = 0; p < patch_size; p++) {
                sum += weight[oc * patch_size + p] * cols[p * spatial_out + s];
            }
            out[oc * spatial_out + s] = sum;
        }
    }
#endif

    free(cols);

    /* Add bias */
    if (bias) {
        for (int oc = 0; oc < c_out; oc++) {
            float b = bias[oc];
            float *row = out + oc * spatial_out;
            for (int s = 0; s < spatial_out; s++) {
                row[s] += b;
            }
        }
    }
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void qwen_layer_norm(float *out, const float *x, const float *weight, const float *bias,
                     int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
#if defined(__AVX512F__)
        __m512 sumv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            sumv = _mm512_add_ps(sumv, _mm512_loadu_ps(x_row + i));
        }
        float mean = _mm512_reduce_add_ps(sumv);
        for (; i < hidden; i++) mean += x_row[i];
#elif defined(__AVX2__)
        __m256 sumv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            sumv = _mm256_add_ps(sumv, _mm256_loadu_ps(x_row + i));
        }
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sumv), _mm256_extractf128_ps(sumv, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float mean = _mm_cvtss_f32(sum128);
        for (; i < hidden; i++) mean += x_row[i];
#else
        float mean = 0.0f;
        for (int i = 0; i < hidden; i++) mean += x_row[i];
#endif
        mean /= hidden;

        /* Compute variance */
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv = _mm512_set1_ps(mean);
        __m512 accv = _mm512_setzero_ps();
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 v = _mm512_sub_ps(_mm512_loadu_ps(x_row + j), meanv);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float var = _mm512_reduce_add_ps(accv);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv = _mm256_set1_ps(mean);
        __m256 accv = _mm256_setzero_ps();
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x_row + j), meanv);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float var = _mm_cvtss_f32(acc128);
        for (; j < hidden; j++) {
            float d = x_row[j] - mean;
            var += d * d;
        }
#else
        float var = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float d = x_row[i] - mean;
            var += d * d;
        }
#endif
        var /= hidden;

        float inv_std = 1.0f / sqrtf(var + eps);
#if defined(__AVX512F__) && defined(__FMA__)
        __m512 meanv2 = _mm512_set1_ps(mean);
        __m512 invv = _mm512_set1_ps(inv_std);
        int k = 0;
        for (; k + 16 <= hidden; k += 16) {
            __m512 vx = _mm512_sub_ps(_mm512_loadu_ps(x_row + k), meanv2);
            __m512 vw = _mm512_loadu_ps(weight + k);
            __m512 vb = _mm512_loadu_ps(bias + k);
            __m512 v = _mm512_mul_ps(_mm512_mul_ps(vx, invv), vw);
            v = _mm512_add_ps(v, vb);
            _mm512_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 meanv2 = _mm256_set1_ps(mean);
        __m256 invv = _mm256_set1_ps(inv_std);
        int k = 0;
        for (; k + 8 <= hidden; k += 8) {
            __m256 vx = _mm256_sub_ps(_mm256_loadu_ps(x_row + k), meanv2);
            __m256 vw = _mm256_loadu_ps(weight + k);
            __m256 vb = _mm256_loadu_ps(bias + k);
            __m256 v = _mm256_mul_ps(_mm256_mul_ps(vx, invv), vw);
            v = _mm256_add_ps(v, vb);
            _mm256_storeu_ps(out_row + k, v);
        }
        for (; k < hidden; k++) {
            out_row[k] = (x_row[k] - mean) * inv_std * weight[k] + bias[k];
        }
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
        }
#endif
    }
}

void qwen_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps) {
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

#if defined(__AVX512F__) && defined(__FMA__)
        __m512 accv = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= hidden; i += 16) {
            __m512 v = _mm512_loadu_ps(x_row + i);
            accv = _mm512_fmadd_ps(v, v, accv);
        }
        float sum_sq = _mm512_reduce_add_ps(accv);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#elif defined(__AVX2__) && defined(__FMA__)
        __m256 accv = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= hidden; i += 8) {
            __m256 v = _mm256_loadu_ps(x_row + i);
            accv = _mm256_fmadd_ps(v, v, accv);
        }
        __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
        acc128 = _mm_hadd_ps(acc128, acc128);
        acc128 = _mm_hadd_ps(acc128, acc128);
        float sum_sq = _mm_cvtss_f32(acc128);
        for (; i < hidden; i++) sum_sq += x_row[i] * x_row[i];
#else
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
#endif
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden + eps);

#if defined(__AVX512F__)
        __m512 scale = _mm512_set1_ps(rms_inv);
        int j = 0;
        for (; j + 16 <= hidden; j += 16) {
            __m512 vx = _mm512_loadu_ps(x_row + j);
            __m512 vw = _mm512_loadu_ps(weight + j);
            _mm512_storeu_ps(out_row + j, _mm512_mul_ps(_mm512_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
        __m256 scale = _mm256_set1_ps(rms_inv);
        int j = 0;
        for (; j + 8 <= hidden; j += 8) {
            __m256 vx = _mm256_loadu_ps(x_row + j);
            __m256 vw = _mm256_loadu_ps(weight + j);
            _mm256_storeu_ps(out_row + j, _mm256_mul_ps(_mm256_mul_ps(vx, vw), scale));
        }
        for (; j < hidden; j++) out_row[j] = x_row[j] * rms_inv * weight[j];
#else
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
#endif
    }
}

void qwen_rms_norm_per_head(float *x, const float *weight,
                             int seq_len, int n_heads, int head_dim, float eps) {
    /* x is [seq, n_heads * head_dim] - normalize each [head_dim] segment */
    int hidden = n_heads * head_dim;
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            __m512 accv = _mm512_setzero_ps();
            int d = 0;
            for (; d + 16 <= head_dim; d += 16) {
                __m512 v = _mm512_loadu_ps(vec + d);
                accv = _mm512_fmadd_ps(v, v, accv);
            }
            float sum_sq = _mm512_reduce_add_ps(accv);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#elif defined(__AVX2__) && defined(__FMA__)
            __m256 accv = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 v = _mm256_loadu_ps(vec + d);
                accv = _mm256_fmadd_ps(v, v, accv);
            }
            __m128 acc128 = _mm_add_ps(_mm256_castps256_ps128(accv), _mm256_extractf128_ps(accv, 1));
            acc128 = _mm_hadd_ps(acc128, acc128);
            acc128 = _mm_hadd_ps(acc128, acc128);
            float sum_sq = _mm_cvtss_f32(acc128);
            for (; d < head_dim; d++) sum_sq += vec[d] * vec[d];
#else
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += vec[d] * vec[d];
            }
#endif
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

#if defined(__AVX512F__)
            __m512 scale = _mm512_set1_ps(rms_inv);
            int j = 0;
            for (; j + 16 <= head_dim; j += 16) {
                __m512 v = _mm512_loadu_ps(vec + j);
                __m512 w = _mm512_loadu_ps(weight + j);
                _mm512_storeu_ps(vec + j, _mm512_mul_ps(_mm512_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#elif defined(__AVX2__)
            __m256 scale = _mm256_set1_ps(rms_inv);
            int j = 0;
            for (; j + 8 <= head_dim; j += 8) {
                __m256 v = _mm256_loadu_ps(vec + j);
                __m256 w = _mm256_loadu_ps(weight + j);
                _mm256_storeu_ps(vec + j, _mm256_mul_ps(_mm256_mul_ps(v, w), scale));
            }
            for (; j < head_dim; j++) vec[j] = vec[j] * rms_inv * weight[j];
#else
            for (int d = 0; d < head_dim; d++) {
                vec[d] = vec[d] * rms_inv * weight[d];
            }
#endif
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void qwen_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + expf(-val));
    }
}

void qwen_gelu(float *x, int n) {
    /*
     * OPT: Vectorized GELU using Accelerate vvtanhf.
     * Rationale: GELU uses scalar tanhf per element, which is very expensive.
     *   The encoder calls GELU per layer (24 layers × FFN + conv stem).
     *   For total_tokens × ffn_dim = 38 × 4096 = 155K elements per layer,
     *   scalar tanhf costs ~2ms per call.
     * Method: Process in blocks of 4096 using Accelerate's vectorized vvtanhf.
     *   Compute inner = sqrt(2/pi)*(x + 0.044715*x³), then batch tanh.
     * Effect: ~4-8x throughput improvement on GELU, saving ~10-20ms on encoder.
     */
#if defined(__APPLE__) && defined(USE_BLAS)
    float buf[4096];
    int i = 0;
    while (i < n) {
        int block = n - i;
        if (block > 4096) block = 4096;
        for (int j = 0; j < block; j++) {
            float v = x[i + j];
            buf[j] = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        }
        vvtanhf(buf, buf, &block);
        for (int j = 0; j < block; j++) {
            x[i + j] = 0.5f * x[i + j] * (1.0f + buf[j]);
        }
        i += block;
    }
#else
    for (int i = 0; i < n; i++) {
        float val = x[i];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    }
#endif
}

typedef struct {
    float *out;
    const float *gate_up;
    int seq_len;
    int intermediate;
} swiglu_task_t;

static void swiglu_worker(int tid, int n_threads, void *arg) {
    swiglu_task_t *t = (swiglu_task_t *)arg;
    int chunk = (t->seq_len + n_threads - 1) / n_threads;
    int s0 = tid * chunk;
    int s1 = s0 + chunk;
    if (s1 > t->seq_len) s1 = t->seq_len;
    if (s0 >= s1) return;

    int inter = t->intermediate;
    int alias_inplace = (t->out == t->gate_up);
    for (int s = s0; s < s1; s++) {
        const float *gu = t->gate_up + (size_t)s * 2 * inter;
        float *o = t->out + (size_t)s * inter;
        if (!alias_inplace) {
#if defined(__APPLE__) && defined(USE_BLAS)
            /* Fast path for prefill: vectorized exp(-g) using Accelerate/vForce. */
            for (int j = 0; j < inter; j++) o[j] = -gu[2 * j];
            int n = inter;
            vvexpf(o, o, &n);
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                o[j] = (g / (1.0f + o[j])) * u;
            }
#else
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        } else {
            /*
             * OPT: Vectorized SiLU via Accelerate vvexpf for decode (seq=1) in-place path.
             * Rationale: SiLU uses scalar expf() per element. With intermediate=6144,
             *   that's 6144 expf calls per layer × 28 layers = ~3.4ms per token.
             * Method: Use alloca scratch buffer to extract gate values, batch-process with
             *   vvexpf (Apple SIMD-optimized), then combine. The gate_up buffer is
             *   interleaved [g0,u0,g1,u1,...] so we can't use vvexpf in-place.
             * Effect: ~14x faster SiLU → saves ~3ms per decode token (~45ms for 15 tokens).
             */
#if defined(__APPLE__) && defined(USE_BLAS)
            float neg_gate[8192]; /* stack scratch, sufficient for intermediate<=8192 */
            int n = inter;
            if (n <= 8192) {
                for (int j = 0; j < inter; j++) neg_gate[j] = -gu[2 * j];
                vvexpf(neg_gate, neg_gate, &n);
                for (int j = 0; j < inter; j++) {
                    float g = gu[2 * j];
                    float u = gu[2 * j + 1];
                    o[j] = (g / (1.0f + neg_gate[j])) * u;
                }
            } else {
                for (int j = 0; j < inter; j++) {
                    float g = gu[2 * j];
                    float u = gu[2 * j + 1];
                    g = g / (1.0f + expf(-g));
                    o[j] = g * u;
                }
            }
#else
            /* In-place mode: sequential scalar SiLU (alias-safe). */
            for (int j = 0; j < inter; j++) {
                float g = gu[2 * j];
                float u = gu[2 * j + 1];
                g = g / (1.0f + expf(-g)); /* SiLU */
                o[j] = g * u;
            }
#endif
        }
    }
}

void qwen_swiglu_multiply(float *out, const float *gate_up, int seq_len, int intermediate) {
    swiglu_task_t task = {
        .out = out,
        .gate_up = gate_up,
        .seq_len = seq_len,
        .intermediate = intermediate
    };

    if (tp.n_threads > 1 && seq_len >= 2 && intermediate >= 256) {
        parallel_for(swiglu_worker, &task);
    } else {
        swiglu_worker(0, 1, &task);
    }
}

void qwen_softmax(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = expf(row[c] - max_val);
            sum += row[c];
        }
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

static inline float qwen_dot_f32(const float *a, const float *b, int n) {
    return qwen_dot_f32_impl(a, b, n);
}

/* dst = dst * scale */
static inline void qwen_vec_scale_inplace(float *dst, float scale, int n) {
    qwen_vec_scale_inplace_impl(dst, scale, n);
}

/* dst += alpha * src */
static inline void qwen_vec_axpy_inplace(float *dst, const float *src, float alpha, int n) {
    qwen_vec_axpy_inplace_impl(dst, src, alpha, n);
}

/* dst = dst * correction + src */
static inline void qwen_vec_scale_add(float *dst, const float *src, float correction, int n) {
    qwen_vec_scale_add_impl(dst, src, correction, n);
}

/*
 * OPT: Threaded bidirectional attention for encoder.
 * Rationale: Encoder attention was single-threaded, iterating over all heads
 *   sequentially. For long audio (e.g. 60s→~500 tokens), this becomes the
 *   encoder bottleneck since each head does O(window²) work independently.
 * Method: Split heads across thread pool, identical to causal_attention threading.
 * Effect: Near-linear speedup with thread count for encoder attention.
 */
static void qwen_bidirectional_attention_heads(const float *Q, const float *K,
                                                const float *V, float *out,
                                                int n_heads, int head_dim, float scale,
                                                const int *window_starts, int n_windows,
                                                int head_start, int head_end) {
    int hidden = n_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        for (int w = 0; w < n_windows; w++) {
            int ws = window_starts[w];
            int we = window_starts[w + 1];

            for (int i = ws; i < we; i++) {
                const float *q_row = Q + i * hidden + h * head_dim;
                float *o_row = out + i * hidden + h * head_dim;

                /* Online softmax */
                float max_score = -1e30f;
                float sum_exp = 0.0f;
                for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

                for (int j = ws; j < we; j++) {
                    const float *k_row = K + j * hidden + h * head_dim;
                    const float *v_row = V + j * hidden + h * head_dim;

                    float score = qwen_dot_f32(q_row, k_row, head_dim) * scale;

                    if (score > max_score) {
                        float correction = expf(max_score - score);
                        sum_exp = sum_exp * correction + 1.0f;
                        qwen_vec_scale_add(o_row, v_row, correction, head_dim);
                        max_score = score;
                    } else {
                        float wt = expf(score - max_score);
                        sum_exp += wt;
                        qwen_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                    }
                }

                if (sum_exp > 0.0f) {
                    float inv_sum = 1.0f / sum_exp;
                    qwen_vec_scale_inplace(o_row, inv_sum, head_dim);
                }
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int n_heads;
    int head_dim;
    float scale;
    const int *window_starts;
    int n_windows;
} bidir_attn_task_t;

static void bidir_attn_worker(int tid, int n_threads, void *arg) {
    bidir_attn_task_t *t = (bidir_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    qwen_bidirectional_attention_heads(t->Q, t->K, t->V, t->out,
                                       t->n_heads, t->head_dim, t->scale,
                                       t->window_starts, t->n_windows, h0, h1);
}

void qwen_bidirectional_attention(float *out, const float *Q, const float *K,
                                   const float *V, int seq __attribute__((unused)),
                                   int n_heads, int head_dim, float scale,
                                   const int *window_starts, int n_windows) {
    if (tp.n_threads > 1 && n_heads >= 2) {
        bidir_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .n_heads = n_heads, .head_dim = head_dim, .scale = scale,
            .window_starts = window_starts, .n_windows = n_windows
        };
        parallel_for(bidir_attn_worker, &task);
        return;
    }

    qwen_bidirectional_attention_heads(Q, K, V, out,
                                       n_heads, head_dim, scale,
                                       window_starts, n_windows, 0, n_heads);
}

static void qwen_causal_attention_heads(float *out, const float *Q, const float *K,
                                        const float *V, int seq_q, int seq_k,
                                        int n_heads, int n_kv_heads, int head_dim,
                                        float scale, int q_offset,
                                        int head_start, int head_end) {
    int heads_per_kv = n_heads / n_kv_heads;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    for (int h = head_start; h < head_end; h++) {
        int kv_h = h / heads_per_kv;

        for (int i = 0; i < seq_q; i++) {
            const float *q_row = Q + i * q_hidden + h * head_dim;
            float *o_row = out + i * q_hidden + h * head_dim;
            int global_pos = q_offset + i;
            int k_end = global_pos + 1;
            if (k_end > seq_k) k_end = seq_k;

            float max_score = -1e30f;
            float sum_exp = 0.0f;
            for (int d = 0; d < head_dim; d++) o_row[d] = 0.0f;

            for (int j = 0; j < k_end; j++) {
                const float *k_row = K + j * kv_hidden + kv_h * head_dim;
                const float *v_row = V + j * kv_hidden + kv_h * head_dim;

                float score = qwen_dot_f32(q_row, k_row, head_dim) * scale;

                if (score > max_score) {
                    float correction = expf(max_score - score);
                    sum_exp = sum_exp * correction + 1.0f;
                    qwen_vec_scale_add(o_row, v_row, correction, head_dim);
                    max_score = score;
                } else {
                    float wt = expf(score - max_score);
                    sum_exp += wt;
                    qwen_vec_axpy_inplace(o_row, v_row, wt, head_dim);
                }
            }

            if (sum_exp > 0.0f) {
                float inv_sum = 1.0f / sum_exp;
                qwen_vec_scale_inplace(o_row, inv_sum, head_dim);
            }
        }
    }
}

typedef struct {
    float *out;
    const float *Q;
    const float *K;
    const float *V;
    int seq_q, seq_k;
    int n_heads, n_kv_heads;
    int head_dim;
    float scale;
    int q_offset;
} causal_attn_task_t;

static void causal_attn_worker(int tid, int n_threads, void *arg) {
    causal_attn_task_t *t = (causal_attn_task_t *)arg;
    int chunk = (t->n_heads + n_threads - 1) / n_threads;
    int h0 = tid * chunk;
    int h1 = h0 + chunk;
    if (h1 > t->n_heads) h1 = t->n_heads;
    if (h0 >= h1) return;

    qwen_causal_attention_heads(t->out, t->Q, t->K, t->V,
                                t->seq_q, t->seq_k, t->n_heads, t->n_kv_heads,
                                t->head_dim, t->scale, t->q_offset, h0, h1);
}

void qwen_causal_attention(float *out, const float *Q, const float *K, const float *V,
                            int seq_q, int seq_k, int n_heads, int n_kv_heads,
                            int head_dim, float scale, int q_offset) {
    if (tp.n_threads > 1 && n_heads >= 2 && (seq_q >= 2 || seq_k >= 128)) {
        causal_attn_task_t task = {
            .out = out, .Q = Q, .K = K, .V = V,
            .seq_q = seq_q, .seq_k = seq_k,
            .n_heads = n_heads, .n_kv_heads = n_kv_heads,
            .head_dim = head_dim, .scale = scale, .q_offset = q_offset
        };
        parallel_for(causal_attn_worker, &task);
        return;
    }

    qwen_causal_attention_heads(out, Q, K, V,
                                seq_q, seq_k, n_heads, n_kv_heads,
                                head_dim, scale, q_offset, 0, n_heads);
}

/* ========================================================================
 * Position Embeddings
 * ======================================================================== */

void qwen_sinusoidal_pe(float *pe, int n_pos, int d_model) {
    int half = d_model / 2;
    float log_timescale = logf(10000.0f) / (float)(half - 1);

    for (int p = 0; p < n_pos; p++) {
        float *row = pe + p * d_model;
        for (int d = 0; d < half; d++) {
            float inv_timescale = expf(-(float)d * log_timescale);
            float angle = (float)p * inv_timescale;
            row[d] = sinf(angle);          /* first half: sin */
            row[half + d] = cosf(angle);   /* second half: cos */
        }
    }
}

void qwen_compute_rope_neox(float *cos_out, float *sin_out, const int *positions,
                              int seq, int head_dim, float theta) {
    int half = head_dim / 2;

    for (int s = 0; s < seq; s++) {
        float pos = (float)positions[s];
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)head_dim);
            float angle = pos * freq;
            float c = cosf(angle);
            float sn = sinf(angle);
            /* Duplicate for full head_dim */
            cos_out[s * head_dim + d] = c;
            cos_out[s * head_dim + half + d] = c;
            sin_out[s * head_dim + d] = sn;
            sin_out[s * head_dim + half + d] = sn;
        }
    }
}

void qwen_apply_rope_neox(float *x, const float *cos_vals, const float *sin_vals,
                            int seq, int n_heads, int head_dim) {
    /*
     * NeoX split-half style:
     *   x1 = x[..., :half], x2 = x[..., half:]
     *   rotated = cat(-x2, x1)
     *   result = x * cos + rotated * sin
     */
    int half = head_dim / 2;
    int hidden = n_heads * head_dim;

    for (int s = 0; s < seq; s++) {
        const float *c = cos_vals + s * head_dim;
        const float *sn = sin_vals + s * head_dim;

        for (int h = 0; h < n_heads; h++) {
            float *vec = x + s * hidden + h * head_dim;

#if defined(__AVX512F__) && defined(__FMA__)
            int d = 0;
            for (; d + 16 <= half; d += 16) {
                __m512 x1 = _mm512_loadu_ps(vec + d);
                __m512 x2 = _mm512_loadu_ps(vec + half + d);
                /* RoPE cache duplicates cos/sin across halves. */
                __m512 cc = _mm512_loadu_ps(c + d);
                __m512 ss = _mm512_loadu_ps(sn + d);
                __m512 new1 = _mm512_fmsub_ps(x1, cc, _mm512_mul_ps(x2, ss));
                __m512 new2 = _mm512_fmadd_ps(x2, cc, _mm512_mul_ps(x1, ss));
                _mm512_storeu_ps(vec + d, new1);
                _mm512_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#elif defined(__AVX2__) && defined(__FMA__)
            int d = 0;
            for (; d + 8 <= half; d += 8) {
                __m256 x1 = _mm256_loadu_ps(vec + d);
                __m256 x2 = _mm256_loadu_ps(vec + half + d);
                __m256 cc = _mm256_loadu_ps(c + d);
                __m256 ss = _mm256_loadu_ps(sn + d);
                __m256 new1 = _mm256_fmsub_ps(x1, cc, _mm256_mul_ps(x2, ss));
                __m256 new2 = _mm256_fmadd_ps(x2, cc, _mm256_mul_ps(x1, ss));
                _mm256_storeu_ps(vec + d, new1);
                _mm256_storeu_ps(vec + half + d, new2);
            }
            for (; d < half; d++) {
                float x1 = vec[d];
                float x2 = vec[half + d];
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#else
            for (int d = 0; d < half; d++) {
                float x1 = vec[d];           /* first half */
                float x2 = vec[half + d];    /* second half */
                vec[d]        = x1 * c[d]        + (-x2) * sn[d];
                vec[half + d] = x2 * c[half + d] + x1 * sn[half + d];
            }
#endif
        }
    }
}
