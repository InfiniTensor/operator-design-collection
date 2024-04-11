# MatMul in cuBLASLt

> <https://docs.nvidia.com/cuda/cublas/index.html#using-the-cublaslt-api>

## cuBLASLt 介绍

由于 cuBLASLt 是专为矩阵乘设计的库，只会出现在这一个文件中，因此在此对库本身做简要的介绍。以下内容来自 cuBLASLt 官方文档的 [概述](https://docs.nvidia.com/cuda/cublas/index.html#id41) 部分：

> cuBLASLt 库是一个新的轻量级库，专门用于通用矩阵乘（GEMM）操作，具有新的灵活 API。这个新库增加了矩阵数据布局、输入类型、计算类型的灵活性，还通过参数可编程性选择了算法实现和试探法。
>
> 一旦用户确定了用于预期 GEMM 操作的一组选项，这些选项可以重复用于不同的输入。这类似于库 FFT 和 FFTW 如何首先创建一个计划，并重用具有不同输入数据的相同大小和类型的 FFT。

## cuBLASLt 接口

调用 cuBLASLt 完成矩阵乘需要 3 步：

1. 生成矩阵乘描述符和矩阵描述符，在这一步中确定输入矩阵的布局、数据类型、计算类型等信息。
2. 选择算法实现，在这一步中选择最适合输入数据的算法实现。
3. 调用 cuBLASLt 矩阵乘 API，在这一步中传入描述符、输入矩阵、输出矩阵、算法实现、工作空间等信息，完成矩阵乘操作。

`cublasLtMatmulAlgoGetHeuristic` 函数用于选择算法实现，`cublasLtMatmul` 函数用于调用矩阵乘 API：

```c
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t lightHandle,
    cublasLtMatmulDesc_t operationDesc,
    cublasLtMatrixLayout_t Adesc,
    cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc,
    cublasLtMatrixLayout_t Ddesc,
    cublasLtMatmulPreference_t preference,
    int requestedAlgoCount,
    cublasLtMatmulHeuristicResult_t heuristicResultsArray[]
    int *returnAlgoCount);
```

```c
cublasStatus_t cublasLtMatmul(
    cublasLtHandle_t            lightHandle,
    cublasLtMatmulDesc_t        computeDesc,
    const void                 *alpha,
    const void                 *A,
    cublasLtMatrixLayout_t      Adesc,
    const void                 *B,
    cublasLtMatrixLayout_t      Bdesc,
    const void                 *beta,
    const void                 *C,
    cublasLtMatrixLayout_t      Cdesc,
    void                       *D,
    cublasLtMatrixLayout_t      Ddesc,
    const cublasLtMatmulAlgo_t *algo,
    void                       *workspace,
    size_t                      workspaceSizeInBytes,
    cudaStream_t                stream);
```
