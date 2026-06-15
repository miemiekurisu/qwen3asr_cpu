/* cuda_decode_params.h — Device-side dynamic parameters for CUDA Graph DecodeStep
 *
 * This struct is shared between cuda_graph.cu, rope.cu, attention.cu,
 * embedding.cu, and cuda_backend.cc.
 */
#ifndef QASR_CUDA_DECODE_PARAMS_H
#define QASR_CUDA_DECODE_PARAMS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Device-side dynamic parameters — all graph kernels read from here */
struct CudaDecodeParams {
    int prev_token;   /* previous token ID for embedding lookup */
    int seq_pos;      /* current sequence position (0-based) */
};

#ifdef __cplusplus
}
#endif

#endif /* QASR_CUDA_DECODE_PARAMS_H */
