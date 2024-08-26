import torch
from im2col_col2im_selfdef.modules import Conv2d

inputs = torch.randn(2,3,56,56)
inputs_gpu = torch.randn(2,3,56,56).cuda()

conv2d = Conv2d(3, 16, 3, 1, 1)
conv2d_gpu = Conv2d(3, 16, 3, 1, 1).cuda()

print('Test for our selfdefine conv2d on cpu')
print('if gradient of weight exists: ', conv2d.weight.grad!=None)
outputs = conv2d(inputs)
loss = outputs.sum()
loss.backward()
print('if gradient of weight exists: ', conv2d.weight.grad!=None, 'sum of it: ', conv2d.weight.grad.sum())

print('Test for our selfdefine conv2d on gpu')
print('if gradient of weight exists: ', conv2d_gpu.weight.grad!=None)
outputs_gpu = conv2d_gpu(inputs_gpu)
loss_gpu = outputs_gpu.sum()
loss_gpu.backward()
print('if gradient of weight exists: ', conv2d_gpu.weight.grad!=None, 'sum of it: ', conv2d_gpu.weight.grad.sum())