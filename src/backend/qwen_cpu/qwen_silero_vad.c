/* Silero VAD v5 ONNX-runtime wrapper implementation.  See header for
 * semantics.  All ONNX runtime calls are guarded by QWEN_HAS_ONNXRUNTIME
 * (set in CMake) so the build is portable: when ONNX runtime isn't
 * available the VAD is a no-op that always reports "speech" (prob=1.0),
 * letting callers keep their legacy timeout-based detection. */
#include "qwen_silero_vad.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef QWEN_HAS_ONNXRUNTIME
#include <onnxruntime_c_api.h>
#endif

#define QWEN_SILERO_VAD_CHUNK     512     /* 32 ms @ 16 kHz */
#define QWEN_SILERO_VAD_STATE_DIM 128
#define QWEN_SILERO_VAD_HIDDEN    2       /* LSTM has 2 layers of state */
#define QWEN_SILERO_VAD_CONTEXT   64      /* @ 16 kHz, prepended to each chunk */

struct qwen_silero_vad_t {
    int active;            /* 1 = VAD is real, 0 = stub (always reports speech) */
    int frame_count;       /* frames processed since reset, for stability averaging */
    float prob_ring[8];    /* ring buffer of last 8 frame probabilities */
    int   prob_ring_pos;

#ifdef QWEN_HAS_ONNXRUNTIME
    OrtApi *api;
    OrtEnv  *env;
    OrtSession *session;
    OrtSessionOptions *opts;
    OrtMemoryInfo *mem;
    /* Pre-allocated input tensors (reused across calls). */
    OrtValue *input_audio;     /* [1, CHUNK + CONTEXT] = [1, 576] */
    OrtValue *input_state;     /* [2, 1, 128] */
    OrtValue *input_sr;        /* [] int64 */
    float    *audio_buf;       /* backing storage for input_audio */
    float    *state_buf;       /* backing storage for input_state */
    float    *context_buf;     /* [CONTEXT] = [64] (kept across calls) */
    int64_t   sr_buf;          /* 16000 */
    char      input_names[3][32];
    char      output_names[2][32];
#endif
};

/* Resolve the Silero VAD model path.  Order:
 *  1) explicit argument (absolute or relative)
 *  2) env QWEN_SILERO_VAD_MODEL
 *  3) ${QWEN_SILERO_VAD_DIR}/silero_vad.onnx
 *  4) ${QWEN_MODEL_DIR}/silero_vad.onnx  (next to Qwen weights)
 *  Returns NULL if not found. */
static const char *qwen_silero_vad_resolve_path(const char *explicit_path) {
    static char resolved[1024];
    if (explicit_path && explicit_path[0]) {
        FILE *f = fopen(explicit_path, "rb");
        if (f) { fclose(f); return explicit_path; }
    }
    const char *candidates[] = {
        getenv("QWEN_SILERO_VAD_MODEL"),
        NULL,
    };
    for (int i = 0; i < (int)(sizeof(candidates)/sizeof(candidates[0])); ++i) {
        const char *c = candidates[i];
        if (!c || !c[0]) continue;
        FILE *f = fopen(c, "rb");
        if (f) { fclose(f); snprintf(resolved, sizeof(resolved), "%s", c); return resolved; }
    }
    const char *dirs[] = { getenv("QWEN_SILERO_VAD_DIR"), getenv("QWEN_MODEL_DIR") };
    for (int i = 0; i < 2; ++i) {
        const char *d = dirs[i];
        if (!d || !d[0]) continue;
        snprintf(resolved, sizeof(resolved), "%s/silero_vad.onnx", d);
        FILE *f = fopen(resolved, "rb");
        if (f) { fclose(f); return resolved; }
    }
    return NULL;
}

qwen_silero_vad_t *qwen_silero_vad_create(const char *model_path) {
    const char *path = qwen_silero_vad_resolve_path(model_path);
    if (!path) {
        fprintf(stderr, "silero_vad: model not found (set QWEN_SILERO_VAD_MODEL or pass a path); "
                        "falling back to legacy silence detection\n");
    }

    qwen_silero_vad_t *v = (qwen_silero_vad_t *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->active = 0;
    for (int i = 0; i < 8; ++i) v->prob_ring[i] = 1.0f;

#ifndef QWEN_HAS_ONNXRUNTIME
    (void)path;
    return v;
#else
    if (!path) return v;

    v->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!v->api) { fprintf(stderr, "silero_vad: OrtGetApiBase failed\n"); free(v); return NULL; }

    if (v->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "silero_vad", &v->env) != NULL) {
        fprintf(stderr, "silero_vad: CreateEnv failed\n"); free(v); return NULL;
    }
    if (v->api->CreateSessionOptions(&v->opts) != NULL) {
        fprintf(stderr, "silero_vad: CreateSessionOptions failed\n");
        v->api->ReleaseEnv(v->env); free(v); return NULL;
    }
    if (v->api->SetIntraOpNumThreads(v->opts, 1) != NULL) {
        fprintf(stderr, "silero_vad: SetIntraOpNumThreads failed (non-fatal)\n");
    }
    OrtStatus *st = v->api->CreateSession(v->env, path, v->opts, &v->session);
    if (st != NULL) {
        fprintf(stderr, "silero_vad: CreateSession(%s) failed: %s\n", path, v->api->GetErrorCode(st) ? "" : "");
        const char *msg = v->api->GetErrorMessage ? v->api->GetErrorMessage(st) : NULL;
        if (msg) fprintf(stderr, "  %s\n", msg);
        v->api->ReleaseSessionOptions(v->opts);
        v->api->ReleaseEnv(v->env);
        free(v);
        return NULL;
    }
    v->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &v->mem);

    /* Allocate input tensor backing storage. */
    v->audio_buf = (float *)calloc(QWEN_SILERO_VAD_CHUNK + QWEN_SILERO_VAD_CONTEXT, sizeof(float));
    v->state_buf = (float *)calloc(QWEN_SILERO_VAD_HIDDEN * QWEN_SILERO_VAD_STATE_DIM, sizeof(float));
    v->context_buf = (float *)calloc(QWEN_SILERO_VAD_CONTEXT, sizeof(float));
    v->sr_buf = 16000;
    if (!v->audio_buf || !v->state_buf || !v->context_buf) {
        fprintf(stderr, "silero_vad: out of memory allocating buffers\n");
        qwen_silero_vad_destroy(v);
        return NULL;
    }

    /* Input shape: [1, CONTEXT + CHUNK] = [1, 576].  The model needs
     * the previous 64 samples of context concatenated with the new
     * 512-sample chunk; otherwise its output is essentially random
     * (~0.003 for any signal).  See silero-vad utils_vad.py for the
     * reference implementation. */
    int64_t audio_shape[2] = {1, QWEN_SILERO_VAD_CONTEXT + QWEN_SILERO_VAD_CHUNK};
    int64_t state_shape[3] = {QWEN_SILERO_VAD_HIDDEN, 1, QWEN_SILERO_VAD_STATE_DIM};
    int64_t sr_shape[1] = {};
    if (v->api->CreateTensorWithDataAsOrtValue(v->mem, v->audio_buf,
            sizeof(float) * (QWEN_SILERO_VAD_CONTEXT + QWEN_SILERO_VAD_CHUNK),
            audio_shape, 2,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v->input_audio) != NULL) {
        fprintf(stderr, "silero_vad: failed to create input tensor\n");
        qwen_silero_vad_destroy(v);
        return NULL;
    }
    if (v->api->CreateTensorWithDataAsOrtValue(v->mem, v->state_buf,
            sizeof(float) * QWEN_SILERO_VAD_HIDDEN * QWEN_SILERO_VAD_STATE_DIM,
            state_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v->input_state) != NULL) {
        fprintf(stderr, "silero_vad: failed to create state tensor\n");
        qwen_silero_vad_destroy(v);
        return NULL;
    }
    if (v->api->CreateTensorWithDataAsOrtValue(v->mem, &v->sr_buf,
            sizeof(int64_t), sr_shape, 0,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &v->input_sr) != NULL) {
        fprintf(stderr, "silero_vad: failed to create sr tensor\n");
        qwen_silero_vad_destroy(v);
        return NULL;
    }

    snprintf(v->input_names[0],  sizeof(v->input_names[0]),  "input");
    snprintf(v->input_names[1],  sizeof(v->input_names[1]),  "state");
    snprintf(v->input_names[2],  sizeof(v->input_names[2]),  "sr");
    snprintf(v->output_names[0], sizeof(v->output_names[0]), "output");
    snprintf(v->output_names[1], sizeof(v->output_names[1]), "stateN");

    v->active = 1;
    fprintf(stderr, "silero_vad: loaded model from %s\n", path);
    return v;
#endif /* QWEN_HAS_ONNXRUNTIME */
}

void qwen_silero_vad_destroy(qwen_silero_vad_t *v) {
    if (!v) return;
#ifdef QWEN_HAS_ONNXRUNTIME
    if (v->api) {
        if (v->input_audio) v->api->ReleaseValue(v->input_audio);
        if (v->input_state) v->api->ReleaseValue(v->input_state);
        if (v->input_sr)    v->api->ReleaseValue(v->input_sr);
        if (v->session)     v->api->ReleaseSession(v->session);
        if (v->opts)        v->api->ReleaseSessionOptions(v->opts);
        if (v->mem)         v->api->ReleaseMemoryInfo(v->mem);
        if (v->env)         v->api->ReleaseEnv(v->env);
    }
    free(v->audio_buf);
    free(v->state_buf);
    free(v->context_buf);
#endif
    free(v);
}

int qwen_silero_vad_reset(qwen_silero_vad_t *v) {
    if (!v) return -1;
#ifdef QWEN_HAS_ONNXRUNTIME
    if (v->active) {
        memset(v->state_buf, 0, sizeof(float) * QWEN_SILERO_VAD_HIDDEN * QWEN_SILERO_VAD_STATE_DIM);
        memset(v->context_buf, 0, sizeof(float) * QWEN_SILERO_VAD_CONTEXT);
    }
#endif
    v->frame_count = 0;
    v->prob_ring_pos = 0;
    for (int i = 0; i < 8; ++i) v->prob_ring[i] = 1.0f;
    return 0;
}

int qwen_silero_vad_is_active(const qwen_silero_vad_t *v) {
    return v && v->active;
}

#ifdef QWEN_HAS_ONNXRUNTIME
/* Run one 512-sample frame; returns prob or -1.0 on error. */
static float qwen_silero_vad_run_frame(qwen_silero_vad_t *v, const float *frame) {
    /* Prepend the cross-call context (last 64 samples from the
     * previous frame) to the new chunk.  This is required: without
     * it, the model output is essentially noise (~0.003 for any
     * signal).  Reference: silero-vad utils_vad.py OnnxWrapper. */
    memcpy(v->audio_buf, v->context_buf, sizeof(float) * QWEN_SILERO_VAD_CONTEXT);
    memcpy(v->audio_buf + QWEN_SILERO_VAD_CONTEXT, frame,
           sizeof(float) * QWEN_SILERO_VAD_CHUNK);

    const OrtValue *inputs[3] = { v->input_audio, v->input_state, v->input_sr };
    const char *in_names[3]    = { v->input_names[0], v->input_names[1], v->input_names[2] };
    const char *out_names[2]   = { v->output_names[0], v->output_names[1] };
    OrtValue *outputs[2]       = { NULL, NULL };

    OrtStatus *st = v->api->Run(v->session, NULL, in_names, inputs, 3,
                                out_names, 2, outputs);
    if (st != NULL) {
        fprintf(stderr, "silero_vad: Run failed: %s\n", v->api->GetErrorMessage(st));
        v->api->ReleaseStatus(st);
        return -1.0f;
    }

    float *out_prob = NULL;
    float *out_state = NULL;
    v->api->GetTensorMutableData(outputs[0], (void **)&out_prob);
    v->api->GetTensorMutableData(outputs[1], (void **)&out_state);
    float prob = out_prob ? out_prob[0] : -1.0f;

    /* Copy new state into the input state buffer for the next frame. */
    if (out_state) {
        memcpy(v->state_buf, out_state,
               sizeof(float) * QWEN_SILERO_VAD_HIDDEN * QWEN_SILERO_VAD_STATE_DIM);
    }
    /* Update context: keep the last 64 samples of THIS frame's input
     * for the next call. */
    memcpy(v->context_buf, v->audio_buf + QWEN_SILERO_VAD_CHUNK,
           sizeof(float) * QWEN_SILERO_VAD_CONTEXT);
    v->api->ReleaseValue(outputs[0]);
    v->api->ReleaseValue(outputs[1]);
    return prob;
}
#endif /* QWEN_HAS_ONNXRUNTIME */

int qwen_silero_vad_process(qwen_silero_vad_t *v,
                            const float *samples,
                            int n_samples,
                            float *prob_out) {
    if (!v || !prob_out) return -1;
    if (n_samples <= 0) { *prob_out = 1.0f; return 0; }
#ifndef QWEN_HAS_ONNXRUNTIME
    (void)samples;
    *prob_out = 1.0f;
    return 0;
#else
    if (!v->active) { *prob_out = 1.0f; return 0; }
    if (n_samples % QWEN_SILERO_VAD_CHUNK != 0) {
        fprintf(stderr, "silero_vad: n_samples=%d not a multiple of %d; padding\n",
                n_samples, QWEN_SILERO_VAD_CHUNK);
    }
    int frames = n_samples / QWEN_SILERO_VAD_CHUNK;
    float last_prob = 1.0f;
    for (int f = 0; f < frames; ++f) {
        const float *frame = samples + (size_t)f * QWEN_SILERO_VAD_CHUNK;
        float p = qwen_silero_vad_run_frame(v, frame);
        if (p < 0.0f) {
            *prob_out = 1.0f;
            return -1;
        }
        last_prob = p;
        v->prob_ring[v->prob_ring_pos] = p;
        v->prob_ring_pos = (v->prob_ring_pos + 1) & 7;
        v->frame_count++;
    }
    /* Return the mean of the last 4 frames for stability against
     * momentary blips. */
    int n = v->frame_count < 4 ? v->frame_count : 4;
    if (n == 0) n = 1;
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        int idx = (v->prob_ring_pos - 1 - i + 8) & 7;
        sum += v->prob_ring[idx];
    }
    *prob_out = sum / (float)n;
    return 0;
#endif
}
