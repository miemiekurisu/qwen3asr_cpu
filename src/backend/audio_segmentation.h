#ifndef QASR_AUDIO_SEGMENTATION_H
#define QASR_AUDIO_SEGMENTATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define QASR_AUDIO_SAMPLE_RATE 16000
#define QASR_ENERGY_WINDOW_MS  100

float *qasr_compact_silence(const float *samples, int n_samples, int *out_samples);
int qasr_find_split_point(const float *samples, int n_samples,
                           int target_sample, float search_sec);

#ifdef __cplusplus
}
#endif

#endif /* QASR_AUDIO_SEGMENTATION_H */
