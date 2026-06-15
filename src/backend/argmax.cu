/* argmax.cu — GPU argmax kernel
 *
 * Computes argmax over logits[vocab_size], returns index in *out_idx.
 * Optional *out_val for the max value.
 */

#include <cuda_runtime.h>

/* Simple argmax over a float array — one block, all threads */
__global__ void argmax_kernel(const float *__restrict__ logits,
                               int vocab_size,
                               float * out_val,
                               int * out_idx) {
    /* shared[0:nthreads] = max values (float)
     * shared[nthreads:2*nthreads] = max indices (int) */
    extern __shared__ char shared_raw[];
    float * shared_val = (float *)shared_raw;
    int * shared_idx = (int *)(shared_raw + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    int local_max_idx = -1;
    float local_max_val = -INFINITY;

    /* Phase 1: each thread finds local max */
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > local_max_val) {
            local_max_val = v;
            local_max_idx = i;
        }
    }
    shared_val[tid] = local_max_val;
    shared_idx[tid] = local_max_idx;
    __syncthreads();

    /* Phase 2: tree reduction */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *out_val = shared_val[0];
        *out_idx = shared_idx[0];
    }
}

extern "C" {

void launch_argmax(cudaStream_t stream,
                    const float * logits,
                    int vocab_size,
                    float * out_val,
                    int * out_idx) {
    int nthreads = 512;
    /* Need nthreads floats + nthreads ints = nthreads * (sizeof(float) + sizeof(int)) */
    size_t shmem = nthreads * (sizeof(float) + sizeof(int));
    argmax_kernel<<<1, nthreads, shmem, stream>>>(
        logits, vocab_size, out_val, out_idx);
}

}  // extern "C"
