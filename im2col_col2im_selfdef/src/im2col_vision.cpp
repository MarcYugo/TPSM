#include <torch/extension.h>

at::Tensor conv2d_im2col(const at::Tensor &input,
                        const int kernel_h, const int kernel_w,
                        const int stride_h, const int stride_w,
                        const int pad_h, const int pad_w,
                        const int dilation_h, const int dilation_w
);

at::Tensor conv2d_col2im(
    const at::Tensor &input, const at::Tensor &grad_output,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_im2col_cuda", &conv2d_im2col, "im2col_cuda");
    m.def("conv2d_col2im_cuda", &conv2d_col2im, "col2im_cuda");
}