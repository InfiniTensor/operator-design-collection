# Softmax in Pytorch

> <https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html>
> <https://pytorch.org/docs/stable/generated/torch.nn.Softmax2d.html>
>
> pytorch中存在两种Softmax定义，一种是普适的Softmax，另一种是特定的Softmax2D
## Softmax
> **调用方法**
> 
> torch.nn.Softmax(dim=None)
> 
> **相关定义**
>
> Applies the Softmax function to an n-dimensional input Tensor rescaling them so that the elements of the n-dimensional output Tensor lie in the range [0,1] and sum to 1.
>
> **参数**
> - **dim**（int）：A dimension along which Softmax will be computed (so every slice along dim will sum to 1).
>
> **备注**
> 
>   测试发现当dim为空时，若input的rank < 2, 默认对最后一维做softmax，若rank > 2，则对axis=1的维度做softmax。
## Softmax2D
> **形状**
> - **input** （N, C, H, W）或者（C, H, W）
> - **output** （N, C, H, W）或者（C, H, W）与input形状一致
>
> **备注**
>
>   无论是（N, C, H, W）或者（C, H, W）都只对C维度做softmax