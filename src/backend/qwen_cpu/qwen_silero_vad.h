/* Minimal Silero VAD v5 ONNX-runtime wrapper for Qwen3-ASR streaming.
 *
 * Silero VAD is a small (~2.3 MB) ONNX model that emits a speech
 * probability per ~32 ms chunk of 16 kHz audio.  This wrapper exposes:
 *
 *   qwen_silero_vad_t *qwen_silero_vad_create(const char *model_path);
 *   void               qwen_silero_vad_destroy(qwen_silero_vad_t *v);
 *   int                qwen_silero_vad_reset(qwen_silero_vad_t *v);
 *   int                qwen_silero_vad_process(qwen_silero_vad_t *v,
 *                                             const float *samples,
 *                                             int n_samples,
 *                                             float *prob_out);
 *
 * Internal model: Silero VAD v5 (ONNX, fp32).  Input layout per the
 * model card:
 *   - input  : [1, N]   float32, N in {512 (16k), 1024 (8k), 256 (16k v5-lt)}
 *   - state  : [2, 1, 128] float32, LSTM hidden state
 *   - sr     : [] int64 scalar, sample rate
 * Output:
 *   - output : [1, 1]  float32, speech probability in [0, 1]
 *   - stateN : [2, 1, 128] float32, new LSTM state
 *
 * The wrapper maintains the LSTM state across calls.  qwen_silero_vad_process
 * accepts any multiple-of-512 audio at 16 kHz; it slices into 512-sample
 * chunks internally and returns the *last* frame's probability.
 *
 * Failure to find ONNX runtime symbols is not a hard error at build time:
 * the build system can leave the VAD as a stub (returns prob=1.0 always,
 * meaning "always speech") so existing call sites still work.  The runtime
 * decide() helper logs and falls back gracefully.
 */
#ifndef QWEN_SILERO_VAD_H
#define QWEN_SILERO_VAD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 32 ms @ 16 kHz.  Silero VAD v5 accepts only this size (or 256 with
 * the v5-lt variant).  Exposed for tests and call sites that need to
 * align audio buffers. */
#define QWEN_SILERO_VAD_CHUNK 512

typedef struct qwen_silero_vad_t qwen_silero_vad_t;

/* Returns NULL on failure.  Use the getenv("QWEN_SILERO_VAD_MODEL") path
 * or pass an absolute path.  Thread-safety: instances are *not* thread-safe
 * (one per live session is enough; the streaming loop is single-threaded
 * per session). */
qwen_silero_vad_t *qwen_silero_vad_create(const char *model_path);

/* Free everything.  Safe on NULL. */
void qwen_silero_vad_destroy(qwen_silero_vad_t *v);

/* Reset LSTM state (call between sessions).  Returns 0 on success. */
int qwen_silero_vad_reset(qwen_silero_vad_t *v);

/* Push audio.  `samples` is mono float in [-1, 1], 16 kHz.
 *  `n_samples` must be a multiple of 512 (a full chunk is 32 ms).
 *  On success: returns 0, writes the last chunk's speech probability
 *  (in [0, 1]) to *prob_out.  The probability is the mean of the last
 *  few frames for stability.  On failure returns -1 and *prob_out is
 *  undefined.
 *
 *  If the ONNX runtime is not available at build time this returns 0
 *  and sets *prob_out = 1.0 (always speech; caller falls back to its
 *  legacy timeout-based detection). */
int qwen_silero_vad_process(qwen_silero_vad_t *v,
                            const float *samples,
                            int n_samples,
                            float *prob_out);

/* True iff the wrapper was compiled with ONNX runtime support AND
 * the model was loaded successfully.  Cheap inline check. */
int qwen_silero_vad_is_active(const qwen_silero_vad_t *v);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* QWEN_SILERO_VAD_H */
