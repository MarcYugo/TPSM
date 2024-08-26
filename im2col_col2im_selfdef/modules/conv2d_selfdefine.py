import torch, math
from torch.nn import Module, Parameter, init
from ..functions.function import Conv2d_function_gpu, Conv2d_function_cpu

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # kernel_h, kernel_w = kernel_size
        self.weight = Parameter(torch.empty((out_channels, in_channels, *kernel_size)))
        self.bias = Parameter(torch.empty((1,))) if bias else None
        self._reset_parameters()

    def _reset_parameters(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, X):
        assert X.shape[1] == self.weight.shape[1], f'inputs channels is not equal to the channels of weight: {X.shape[1]} != {self.weight.shape[1]}'
        assert X.device == self.weight.device, f'inputs and weight should be on same device, {X.device} != {self.weight.device}'
        if self.bias is not None:
            assert X.device == self.bias.device, f'inputs and bias should be on same device, {X.device} != {self.bias.device}' 
        
        if X.is_cuda:
            out = Conv2d_function_gpu.apply(X, self.weight, self.bias, 
                                            self.kernel_size, self.stride, self.padding, self.dilation)
        else:
            out = Conv2d_function_cpu.apply(X, self.weight, self.bias, 
                                            self.kernel_size, self.stride, self.padding, self.dilation)
        return out
