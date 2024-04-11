# MatMul in PyTorch

> <https://pytorch.org/docs/stable/generated/torch.matmul.html>

PyTorch 中存在 `Tensor` 定义，因此这个调用的所有参数都是 `Tensor` 结构，计算的参数将从结构中分析。

PyTorch 的 `Tensor` 支持变换元信息的转置，因此不需要转置参数。

但是 `Tensor` 的阶数不能保证是 2，因此这个调用带有复杂的说明，介绍每个参数在不同阶数时调用的行为。
这些行为包括化作点积、自动补充长度为 1 的维度、广播、批量化等。

这个调用在非常古早的 PyTorch 版本中（1.0.0）就是这样定义的的。

> torch.matmul(input, other, *, out=None) → Tensor
>
> **Parameters**
>
> - **input** (*Tensor*) – the first tensor to be multiplied
> - **input** (*Tensor*) – the first tensor to be multiplied
> - **other** (*Tensor*) – the second tensor to be multiplied
>
> **Keyword Arguments**
>
> **out** (*Tensor, optional*) – the output tensor.
