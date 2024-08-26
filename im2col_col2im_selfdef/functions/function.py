import torch
import torch.nn.functional as F
from torch.autograd import Function

from conv2d_im2col_col2im_selfdefine import conv2d_im2col_cuda, conv2d_col2im_cuda

class Conv2d_function_gpu(Function):
    @staticmethod
    def forward(ctx, X, weight, bias, kernel_size, stride, padding, dilation):
        '''
            X: (B,C,H,W)
            weight: (C',C,Kh,Kw)
            bias: (1,)
        '''
        kernel_h, kernel_w = kernel_size
        pad_h, pad_w = padding
        dilation_h, dilation_w = dilation
        stride_h, stride_w = stride
        b,c,h,w = X.shape
        c1,_,_,_ = weight.shape
        ctx.kernel_size = kernel_size
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.stride = stride
        ctx.img_size = (h, w)
        # im2col
        col_X = conv2d_im2col_cuda(X, 
                                   kernel_h, kernel_w,
                                   stride_h, stride_w,
                                   pad_h, pad_w,
                                   dilation_h, dilation_w)
        ctx.save_for_backward(X, col_X, weight, bias)
        bias = bias.reshape(1,1,1,1).repeat(b,1,1,1) if bias is not None else None
        # linear projection
        out = col_X.permute(0,2,3,1) @ weight.permute(1,2,3,0).reshape(-1, c1)
        out = out.permute(0,3,1,2)
        # add bias
        out = out + bias if bias is not None else out
        return out
    
    @staticmethod
    def backward(ctx, out_gd):
        X, col_X, weight, bias = ctx.saved_tensors
        kernel_h, kernel_w = ctx.kernel_size
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding
        dilation_h, dilation_w = ctx.dilation
        h, w = ctx.img_size
        c1,c,kernel_h,kernel_w = weight.shape
        # gradient of bias
        bias_gd = out_gd.sum(dim=(1,2,3)) if bias is not None else None
        bias_gd = torch.mean(bias_gd, dim=(0,), keepdim=True)
        # gradient of weight
        weight_gd = out_gd.permute(1,0,2,3).reshape(c1, -1) @ col_X.permute(0,2,3,1).reshape(-1, c*kernel_h*kernel_w) # (C', C*Kh*Kw)
        weight_gd = weight_gd.reshape(c1,c,kernel_h,kernel_w)
        # gradient of X
        X_gd = out_gd.permute(0,2,3,1) @ weight.reshape(c1, -1)
        X_gd = X_gd.permute(0,3,1,2).contiguous()
        # col2im
        X_gd = conv2d_col2im_cuda(X, X_gd,
                                  kernel_h, kernel_w,
                                  stride_h, stride_w,
                                  pad_h, pad_w,
                                  dilation_h, dilation_w)
        return X_gd, weight_gd, bias_gd, None, None, None, None
    

# 用于找到对应的数据位置坐标
def data_index_init(offset, clist, *args):
    if not args:
        return offset
    X, *rest = args
    offset = data_index_init(offset, clist, *rest)
    tmp = offset % X
    clist.append(tmp)
    # print(offset, x)
    return offset // X

# image to column
def im2col_cpu(img,  # 图像张量
               kernel_h, # 卷积核大小
               kernel_w,
               pad_h, # 填充
               pad_w,
               stride_h, # 跨步
               stride_w,
               dilation_h, # 卷积核膨胀系数
               dilation_w):
    '''
        img: (N, C, H, W)
    '''
    b,c,h,w = img.shape
    out_h = (h - (dilation_h*(kernel_h - 1) + 1) + 2*pad_h)//stride_h + 1
    out_w = (w - (dilation_w*(kernel_w - 1) + 1) + 2*pad_w)//stride_w + 1
    out_c = c*kernel_h*kernel_w

    col = torch.zeros(b, out_c, out_h, out_w).to(img.device)

    for col_c in range(out_c):
        clist = []
        data_index_init(col_c, clist, c, kernel_h, kernel_w)
        c_im, offset_h, offset_w = clist[::-1]
        for col_h in range(out_h):
            h_im = col_h*stride_h - pad_h + offset_h*dilation_h
            for col_w in range(out_w):
                w_im = col_w*stride_w - pad_w + offset_w*dilation_w
                if h_im < h and h_im >=0 and w_im < w and w_im >= 0:
                    col[:, col_c, col_h, col_w] = img[:, c_im, h_im, w_im]
    return col

# column to image
def col2im_cpu(col_gd,
               c, h, w,
               kernel_h, kernel_w,
               pad_h, pad_w,
               stride_h, stride_w,
               dilation_h, dilation_w):
    '''
        col: (N, C', H', W')
    '''
    b,out_c,out_h,out_w = col_gd.shape
    img_gd = torch.zeros(b, c, h, w)
    for col_c in range(out_c):
        clist = []
        data_index_init(col_c, clist, c, kernel_h, kernel_w)
        c_im, offset_h, offset_w = clist[::-1]
        for col_h in range(out_h):
            h_im = col_h*stride_h - pad_h + offset_h*dilation_h
            for col_w in range(out_w):
                w_im = col_w*stride_w - pad_w + offset_w*dilation_w
                if h_im < h and h_im >=0 and w_im < w and w_im >= 0:
                    img_gd[:, c_im, h_im, w_im] += col_gd[:, col_c, col_h, col_w]
    return img_gd

# forward and backward
class Conv2d_function_cpu(Function):
    @staticmethod
    def forward(ctx, X, weight, bias, kernel_size, stride, padding, dilation):
        '''
            X: (B,C,H,W)
            weight: (C',C,Kh,Kw)
            bias: (1,)
        '''
        kernel_h, kernel_w = kernel_size
        pad_h, pad_w = padding
        dilation_h, dilation_w = dilation
        stride_h, stride_w = stride
        b,c,h,w = X.shape
        c1,_,_,_ = weight.shape
        ctx.kernel_size = kernel_size
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.stride = stride
        ctx.img_size = (h, w)
        # im2col
        col_X = im2col_cpu(X, kernel_h, kernel_w,
                        pad_h, pad_w, stride_h, stride_w,
                        dilation_h, dilation_w) # (B, C*Kh*Kw, H', W')
        ctx.save_for_backward(col_X, weight, bias)
        bias = bias.reshape(1,1,1,1).repeat(len(X),1,1,1) if bias is not None else bias
        # linear projection
        out = col_X.permute(0,2,3,1) @ weight.permute(1,2,3,0).reshape(-1, c1) # (B, H', W', C*Kh*Kw) x (C*Kh*Kw, C') -> (B, H', W', C')
        out = out.permute(0,3,1,2) # (B, C', H', W')
        # add bias
        out = out + bias if bias is not None else out
        return out
    @staticmethod
    def backward(ctx, out_gd):
        col_X, weight, bias = ctx.saved_tensors
        kernel_h, kernel_w = ctx.kernel_size
        stride_h, stride_w = ctx.stride
        pad_h, pad_w = ctx.padding
        dilation_h, dilation_w = ctx.dilation
        h, w = ctx.img_size
        c1,c,kernel_h,kernel_w = weight.shape
        # gradient of bias
        bias_gd = out_gd.sum(dim=(1,2,3)) if bias is not None else None
        bias_gd = torch.mean(bias_gd, dim=(0,), keepdim=True)
        # gradient of weight
        weight_gd = out_gd.permute(1,0,2,3).reshape(c1, -1) @ col_X.permute(0,2,3,1).reshape(-1, c*kernel_h*kernel_w) # (C', C*Kh*Kw)
        weight_gd = weight_gd.reshape(c1,c,kernel_h,kernel_w)
        # gradient of X
        X_gd = out_gd.permute(0,2,3,1) @ weight.reshape(c1, -1) # (B, H', W', C') x (C', C*Kh*Kw) -> (B, H', W', C*Kh*Kw)
        X_gd = X_gd.permute(0,3,1,2)
        # col2im
        X_gd = col2im_cpu(X_gd, c, h, w,
                          kernel_h, kernel_w,
                          pad_h, pad_w, 
                          stride_h, stride_w,
                          dilation_h, dilation_w)
        return X_gd, weight_gd, bias_gd, None, None, None, None