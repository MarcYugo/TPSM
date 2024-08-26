#include "im2col_col2im.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

at::Tensor conv2d_im2col(const at::Tensor &input,
                        const int kernel_h, const int kernel_w,
                        const int stride_h, const int stride_w,
                        const int pad_h, const int pad_w,
                        const int dilation_h, const int dilation_w
){
    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    const int b = input.size(0);
    const int c = input.size(1);
    const int h = input.size(2);
    const int w = input.size(3);
    const int col_h = (h - (dilation_h*(kernel_h - 1) + 1) + 2*pad_h)/stride_h + 1;
    const int col_w = (w - (dilation_w*(kernel_w - 1) + 1) + 2*pad_w)/stride_w + 1;
    const int col_c = c * kernel_h * kernel_w;
    auto output = at::zeros({b, col_c, col_h, col_w}, input.options());
    for(int n = 0; n < b; n++){
        auto slice_input = input.select(0, n);
        auto slice_output = output.select(0, n);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.type(), "conv2d_im2col_selfdefine_forward", ([&]{
                im2col(
                    at::cuda::getCurrentCUDAStream(),
                    slice_input.data<scalar_t>(), slice_output.data<scalar_t>(),
                    c, h, w,
                    col_h, col_w,
                    kernel_h, kernel_w,
                    pad_h, pad_w,
                    stride_h, stride_w,
                    dilation_h, dilation_w );
            }));
    }
    return output;
}

at::Tensor conv2d_col2im(
    const at::Tensor &input, const at::Tensor &grad_output,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w
){
    AT_ASSERTM(input.is_contiguous(), "input tensor has to be contiguous");
    AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
    AT_ASSERTM(grad_output.is_contiguous(), "gradient of output tensor has to be contiguous");
    AT_ASSERTM(grad_output.type().is_cuda(), "gradient of output tensor must be a CUDA tensor");
    
    const int b = input.size(0);
    const int c = input.size(1);
    const int h = input.size(2);
    const int w = input.size(3);
    const int col_h = grad_output.size(2);
    const int col_w = grad_output.size(3);
    const int col_c = grad_output.size(1);

    auto grad_input = at::zeros_like(input, input.dtype());

    for(int n = 0; n < b; n ++){
        auto slice_grad_input = grad_input.select(0, n);
        auto slice_grad_output = grad_output.select(0, n);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(
            input.type(), "conv2d_col2im_selfdefine_backward", ([&]{
                col2im(
                    at::cuda::getCurrentCUDAStream(),
                    slice_grad_output.data<scalar_t>(), slice_grad_input.data<scalar_t>(),
                    c, h, w,
                    col_h, col_w,
                    kernel_h, kernel_w,
                    pad_h, pad_w,
                    stride_h, stride_w,
                    dilation_h, dilation_w );
            }));
    }

    return grad_input;
}