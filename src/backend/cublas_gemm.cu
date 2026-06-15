/* cublas_gemm.cu — cuBLAS GEMM wrapper kernels
 *
 * Wrappers around cuBLAS for GEMM operations needed by decoder layers.
 * Supports bf16 weights converted to fp32 for computation.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Manual bf16 to fp32 conversion (software, no hardware intrinsic)
 * bf16 layout: S(1) E(8) M(7)
 * fp32 layout: S(1) E(8) M(23)
 * Shift exponent and mantissa, handle special values */
union Bf16ToFp32 {
    uint32_t u;
    float f;
};

__device__ __forceinline__ float bf16_to_fp32_sw(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp_mant = (uint32_t)(h & 0x7FFF) << 16;

    if (exp_mant == 0) {
        /* Zero or -zero */
        Bf16ToFp32 tmp;
        tmp.u = sign;
        return tmp.f;
    }

    uint32_t exp = (exp_mant >> 16) & 0xFF;
    if (exp == 0xFF) {
        /* Inf or NaN — preserve as-is (mantissa already shifted) */
        Bf16ToFp32 tmp;
        tmp.u = sign | exp_mant;
        return tmp.f;
    }

    /* Normal number: bf16 bias (127) == fp32 bias (127), so no rebias needed */
    Bf16ToFp32 tmp;
    tmp.u = sign | exp_mant;
    return tmp.f;
}

/* bf16 to fp32 conversion kernel */
__global__ void bf16_to_fp32_kernel(float * __restrict__ out,
                                      const uint16_t * __restrict__ in,
                                      int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    out[idx] = bf16_to_fp32_sw(in[idx]);
}

/* fp32 to bf16 conversion kernel */
__global__ void fp32_to_bf16_kernel(uint16_t * out,
                                     const float * in,
                                     int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    out[idx] = __float2bfloat16(in[idx]);
}

extern "C" {

/* Launch bf16->fp32 conversion */
void launch_bf16_to_fp32(cudaStream_t stream,
                          float * out,
                          const uint16_t * in,
                          int total_elements) {
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    bf16_to_fp32_kernel<<<grid_size, block_size, 0, stream>>>(
        out, in, total_elements);
}

/* Launch fp32->bf16 conversion */
void launch_fp32_to_bf16(cudaStream_t stream,
                           uint16_t * out,
                           const float * in,
                           int total_elements) {
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    fp32_to_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
        out, in, total_elements);
}

/* Matrix transpose kernel: out[col, row] = in[row, col]
 * in is [rows, cols] row-major, out is [cols, rows] row-major */
__global__ void transpose_kernel(const float * __restrict__ in,
                                    float * __restrict__ out,
                                    int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rows || col >= cols) return;
    out[col * rows + row] = in[row * cols + col];
}

/* Transpose matrix on GPU: in [rows, cols] -> out [cols, rows] */
void launch_transpose(cudaStream_t stream,
                       float * out,
                       const float * in,
                       int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    transpose_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols);
}

/* Combined bf16->fp32 conversion + transpose kernel:
 * W is [out_dim, in_dim] row-major bf16
 * W_T is [in_dim, out_dim] row-major fp32
 * W_T[j * out_dim + i] = bf16_to_fp32_sw(W[i * in_dim + j])
 *
 * Each thread handles one (i, j) pair:
 *   i in [0, out_dim)  — row index in W (output dimension)
 *   j in [0, in_dim)   — col index in W (input dimension)
 *   W_T[j * out_dim + i] = convert(W[i * in_dim + j]) */
__global__ void bf16_transpose_kernel(const uint16_t * __restrict__ W,
                                        float * __restrict__ W_T,
                                        int out_dim, int in_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  /* [0, out_dim) */
    int j = blockIdx.x * blockDim.x + threadIdx.x;  /* [0, in_dim) */
    if (i >= out_dim || j >= in_dim) return;

    /* W[i][j] = W[i * in_dim + j] in row-major */
    uint16_t w_val = W[i * in_dim + j];
    /* W_T[j][i] = W_T[j * out_dim + i] in row-major */
    W_T[j * out_dim + i] = bf16_to_fp32_sw(w_val);
}

/* Launch bf16->fp32 conversion + transpose:
 * W is [out_dim, in_dim] bf16, W_T is [in_dim, out_dim] fp32 */
void launch_bf16_transpose(cudaStream_t stream,
                            float * W_T,
                            const uint16_t * W,
                            int out_dim, int in_dim) {
    dim3 block(16, 16);
    dim3 grid((in_dim + block.x - 1) / block.x,
              (out_dim + block.y - 1) / block.y);
    bf16_transpose_kernel<<<grid, block, 0, stream>>>(W, W_T, out_dim, in_dim);
}

}  // extern "C"
