# Softmax in PaddlePaddle
> <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softmax_cn.html>
> <https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Softmax2D_cn.html>
>
> paddle框架中对齐pytorch，存在两种Softmax，一种是普适的Softmax，另一种是特定的Softmax2D
>
## Softmax
> **调用方法**
> 
> class paddle.nn.Softmax(axis=- 1, name=None)
>
> **计算过程**
> - **步骤1**：输入 x 的 axis 维会被置换到最后一维；
> - **步骤2**：将输入 x 在逻辑上变换为二维矩阵。二维矩阵第一维（列长度）是输入除最后一维之外的其他维度值的乘积，第二维（行长度）和输入 axis 维的长度相同；对于矩阵的每一行，softmax 操作对其进行重新缩放，使得该行的每个元素在 [0,1]范围内，并且总和为 1；\
> &ensp;&ensp;上述步骤 2 中 softmax 操作计算过程如下：
>   1. 对于二维矩阵的每一行，计算 K 维向量（K 是输入第 axis 维的长度）中指定位置的指数值和全部位置指数值的和。
>   2. 指定位置指数值与全部位置指数值之和的比值就是 softmax 操作的输出。
> 
> &ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;对于二维矩阵中的第 i 行和第 j 列有：
>
> $$\operatorname{softmax}[i, j]=\frac{\exp (x[i, j])}{\sum_{j}(\exp (x[i, j])}$$
> - **步骤3**：softmax 操作执行完成后，执行步骤 1 和步骤 2 的逆运算，将二维矩阵恢复至和输入 x 相同的维度。
> 
> **参数**
> - **axis**（int，可选）
> 
>   指定对输入 Tensor 进行运算的轴。axis 的有效范围是 [−D,D)，D是输入 Tensor 的维度，axis 为负值时与 axis+D 等价。默认值为 -1。
>
> - **name**（str，可选）
>
>   一般无需设置，默认值为 None。
>
> **形状**
> - **input**：
>
>   任意形状的Tensor
> - **output**：
>
>   和input具有相同形状的Tensor
## Softmax2D
> **调用方法**
> 
> class paddle.nn.Softmax2D(name=None)
>
> Softmax2D 是 Softmax 的变体，其针对 3D（CHW） 或者 4D（NCHW） 的 Tensor 在空间（C）维度上计算 Softmax。且其仅支持 3D 或者 4D 具体来说，输出的 Tensor 的每个空间维度 (channls,hi,wj) 求和为 1。
> 
> **参数**
> 
> - **name**（str，可选）
>
>   一般无需设置，默认值为 None。
>
> **形状**
> - **input**：
>
>   任意形状的Tensor
> - **output**：
>
>   和input具有相同形状的Tensor