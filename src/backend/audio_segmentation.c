#include "audio_segmentation.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_float_asc(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa < fb) ? -1 : (fa > fb) ? 1 : 0;
}

float *qasr_compact_silence(const float *samples, int n_samples, int *out_samples) {
    if (!samples || n_samples <= 0 || !out_samples) return NULL;

    const int win = 160;
    const float base_thresh = 0.002f;
    const float max_thresh = 0.025f;
    const float smooth_alpha = 0.2f;
    const int min_voice_windows = 5;
    const int pad_voice_windows = 3;
    const int pass_windows = 60;

    int n_win = (n_samples + win - 1) / win;
    float *rms_vals = (float *)malloc((size_t)n_win * sizeof(float));
    float *sorted = (float *)malloc((size_t)n_win * sizeof(float));
    float *smooth_vals = (float *)malloc((size_t)n_win * sizeof(float));
    unsigned char *is_voice = (unsigned char *)malloc((size_t)n_win);
    if (!rms_vals || !sorted || !smooth_vals || !is_voice) {
        free(rms_vals);
        free(sorted);
        free(smooth_vals);
        free(is_voice);
        return NULL;
    }

    for (int w = 0; w < n_win; w++) {
        int start = w * win;
        int end = start + win;
        if (end > n_samples) end = n_samples;
        int len = end - start;
        float energy = 0.0f;
        for (int i = 0; i < len; i++) {
            float v = samples[start + i];
            energy += v * v;
        }
        rms_vals[w] = sqrtf(energy / (float)(len > 0 ? len : 1));
    }

    float smooth = rms_vals[0];
    for (int w = 0; w < n_win; w++) {
        smooth = (1.0f - smooth_alpha) * smooth + smooth_alpha * rms_vals[w];
        smooth_vals[w] = smooth;
    }

    memcpy(sorted, smooth_vals, (size_t)n_win * sizeof(float));
    qsort(sorted, (size_t)n_win, sizeof(float), cmp_float_asc);

    int p25 = (int)((n_win - 1) * 0.25f);
    float noise_floor = sorted[p25];
    float thresh = noise_floor * 1.8f;
    if (thresh < base_thresh) thresh = base_thresh;
    if (thresh > max_thresh) thresh = max_thresh;
    free(sorted);

    for (int w = 0; w < n_win; w++) {
        is_voice[w] = (smooth_vals[w] > thresh) ? 1 : 0;
    }
    free(smooth_vals);

    for (int i = 0; i < n_win; ) {
        if (!is_voice[i]) { i++; continue; }
        int j = i + 1;
        while (j < n_win && is_voice[j]) j++;
        if (j - i < min_voice_windows) {
            memset(is_voice + i, 0, (size_t)(j - i));
        }
        i = j;
    }

    unsigned char *padded = (unsigned char *)calloc((size_t)n_win, 1);
    if (!padded) {
        free(is_voice);
        free(rms_vals);
        return NULL;
    }
    for (int w = 0; w < n_win; w++) {
        if (!is_voice[w]) continue;
        int a = w - pad_voice_windows;
        int b = w + pad_voice_windows;
        if (a < 0) a = 0;
        if (b >= n_win) b = n_win - 1;
        for (int k = a; k <= b; k++) padded[k] = 1;
    }
    free(is_voice);

    int out_size = 0;
    {
        int sc = 0;
        for (int w = 0; w < n_win; w++) {
            int start = w * win;
            int end_s = start + win;
            if (end_s > n_samples) end_s = n_samples;
            int len = end_s - start;
            if (padded[w]) { out_size += len; sc = 0; }
            else { sc++; if (sc <= pass_windows) out_size += len; }
        }
    }
    if (out_size == 0) {
        out_size = n_samples;
        int min_keep = QASR_AUDIO_SAMPLE_RATE / 2;
        if (out_size > min_keep) out_size = min_keep;
    }

    float *out = (float *)malloc((size_t)out_size * sizeof(float));
    if (!out) {
        free(rms_vals);
        free(padded);
        return NULL;
    }

    int out_n = 0;
    int silence_count = 0;
    for (int w = 0; w < n_win; w++) {
        int start = w * win;
        int end = start + win;
        if (end > n_samples) end = n_samples;
        int len = end - start;

        if (padded[w]) {
            memcpy(out + out_n, samples + start, (size_t)len * sizeof(float));
            out_n += len;
            silence_count = 0;
        } else {
            silence_count++;
            if (silence_count <= pass_windows) {
                memcpy(out + out_n, samples + start, (size_t)len * sizeof(float));
                out_n += len;
            }
        }
    }
    free(padded);
    free(rms_vals);

    if (out_n == 0) {
        int keep = n_samples;
        int min_keep = QASR_AUDIO_SAMPLE_RATE / 2;
        if (keep > min_keep) keep = min_keep;
        memcpy(out, samples, (size_t)keep * sizeof(float));
        out_n = keep;
    }

    *out_samples = out_n;
    return out;
}

int qasr_find_split_point(const float *samples, int n_samples,
                           int target_sample, float search_sec) {
    int search_half = (int)(search_sec * QASR_AUDIO_SAMPLE_RATE);
    int lo = target_sample - search_half;
    int hi = target_sample + search_half;
    if (lo < 0) lo = 0;
    if (hi > n_samples) hi = n_samples;

    int win_samples = (QASR_ENERGY_WINDOW_MS * QASR_AUDIO_SAMPLE_RATE) / 1000;
    float best_energy = 1e30f;
    int best_center = target_sample;

    for (int pos = lo; pos + win_samples <= hi; pos += win_samples / 2) {
        float energy = 0;
        int end = pos + win_samples;
        if (end > n_samples) end = n_samples;
        for (int j = pos; j < end; j++) {
            energy += samples[j] * samples[j];
        }
        energy /= (end - pos);
        if (energy < best_energy) {
            best_energy = energy;
            best_center = pos + (end - pos) / 2;
        }
    }
    return best_center;
}
