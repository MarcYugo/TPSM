#include <algorithm>
#include <cstdio>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                                                 \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);               \
         i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256;
inline int GET_BLOCKS(const int N, const int num_threads) {
    return (N + num_threads - 1) / num_threads;
}

#define opmath_t at::opmath_type<scalar_t>

template <typename scalar_t>
__global__ void im2col_kernel(const int64_t n,
                                const scalar_t* img, scalar_t* col, 
                                const int64_t h, const int64_t w,
                                const int64_t kernel_h, const int64_t kernel_w,
                                const int64_t pad_h, const int64_t pad_w,
                                const int64_t stride_h, const int64_t stride_w,
                                const int64_t dilation_h, const int64_t dilation_w,
                                const int64_t col_h, const int64_t col_w
){
    CUDA_KERNEL_LOOP(index, n){
        int64_t w_out = index % col_w;
        int64_t h_out = (index / col_w) % col_h;
        int64_t c_in = (index / col_w) / col_h;
        int64_t c_out = c_in * kernel_h * kernel_w;
        int64_t h_in = h_out * stride_h - pad_h;
        int64_t w_in = w_out * stride_w - pad_w;

        scalar_t* col_idx = col + (c_out * col_h + h_out) * col_w + w_out;
        const scalar_t* img_idx = img + (c_in * h + h_in) * w + w_in;

        for(int64_t i=0;i<kernel_h;i++){
            int64_t h_k = h_in + i * dilation_h;
            for(int64_t j=0;j<kernel_w;j++){
                int64_t w_k = w_in + j * dilation_w;
                if(h_k >= 0 && h_k < h && w_k >= 0 && w_k < w)
                    *col_idx = img_idx[i*dilation_h*w + j*dilation_w];
                col += col_h * col_w;
            }
        }
    }
}

template <typename scalar_t>
void im2col(
    cudaStream_t stream,
    const scalar_t* img, scalar_t* col,
    const int64_t c, const int64_t h, const int64_t w,
    const int64_t col_h, const int64_t col_w,
    const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w
){
    int64_t num_kernels = c * h * w;
    im2col_kernel<scalar_t>
    <<<GET_BLOCKS(num_kernels, 1024), 1024, 0, stream>>>(
        num_kernels,
        img, col,
        h, w,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        col_h, col_w );
}

template <typename scalar_t>
__global__ void col2im_kernel(const int64_t n,
                                const scalar_t* col, scalar_t* img, 
                                const int64_t h, const int64_t w,
                                const int64_t kernel_h, const int64_t kernel_w,
                                const int64_t pad_h, const int64_t pad_w,
                                const int64_t stride_h, const int64_t stride_w,
                                const int64_t dilation_h, const int64_t dilation_w,
                                const int64_t col_h, const int64_t col_w
){
    CUDA_KERNEL_LOOP(index, n){
        const int64_t w_im = index % w + pad_w;
        const int64_t h_im = (index / w) % h + pad_h;
        const int64_t c_im = index / (w * h);
        int64_t dilat_k_h = (kernel_h - 1)* dilation_h + 1;
        int64_t dilat_k_w = (kernel_w - 1)* dilation_w + 1;
        const int64_t w_col_s = (w_im < dilat_k_w) ? 0 : (w_im - dilat_k_w) / stride_w + 1;
        const int64_t w_col_e = ::min(w_im/stride_w + 1, col_w);
        const int64_t h_col_s = (h_im < stride_h) ? 0 : (h_im - dilat_k_h) / stride_h + 1;
        const int64_t h_col_e = ::min(h_im / stride_h + 1, col_h);

        scalar_t val = static_cast<scalar_t>(0);

        for(int64_t h_col = h_col_s; h_col < h_col_e; h_col += 1){
            int64_t h_k = (h_im - h_col * stride_h);
            for(int64_t w_col = w_col_s; w_col < w_col_e; w_col += 1){
                int64_t w_k = (w_im - w_col * stride_w);
                if(h_k % dilation_h == 0 && w_k % dilation_w == 0){
                    h_k /= dilation_h;
                    w_k /= dilation_w;
                    int64_t data_col_index = ((((c_im * kernel_h + h_k)* kernel_w + w_k)* col_h + h_col)* col_w + w_col);
                    val += col[data_col_index];
                }
            }
        }
        img[index] = val;
    }
}

template <typename scalar_t>
void col2im(
    cudaStream_t stream,
    const scalar_t* col, scalar_t* img,
    const int64_t c, const int64_t h, const int64_t w,
    const int64_t col_h, const int64_t col_w,
    const int64_t kernel_h, const int64_t kernel_w,
    const int64_t pad_h, const int64_t pad_w,
    const int64_t stride_h, const int64_t stride_w,
    const int64_t dilation_h, const int64_t dilation_w
){
    int64_t num_kernels = c * h * w;
    col2im_kernel<scalar_t>
    <<<GET_BLOCKS(num_kernels, 1024), 1024, 0, stream>>>(
        num_kernels,
        col, img,
        h, w,
        kernel_h, kernel_w,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        col_h, col_w );
}