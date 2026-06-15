/* reshape.cu — Reshape conv2D output for linear projection
 *
 * Input:  [c, h, w]  (conv2D output)
 * Output: [w, c*h]   (for linear projection)
 * Formula: out[t * c * h + ch * h + f] = in[ch * h * w + f * w + t]
 */

#include <cuda_runtime.h>

__global__ void reshape_kernel(const float * __restrict__ in,
                                 float * __restrict__ out,
                                 int c, int h, int w) {
    int total_out = w * c * h;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_out) return;

    /* out[idx] where idx = t * c * h + ch * h + f */
    int t = idx / (c * h);
    int remainder = idx % (c * h);
    int ch = remainder / h;
    int f = remainder % h;
    out[idx] = in[ch * h * w + f * w + t];
}

extern "C" {

/* Launch reshape: in[c,h,w] → out[w, c*h] */
void launch_reshape(cudaStream_t stream,
                     float * out,
                     const float * in,
                     int c, int h, int w) {
    int total = w * c * h;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    reshape_kernel<<<grid_size, block_size, 0, stream>>>(in, out, c, h, w);
}

}  // extern "C"
