# MatMul in cuBLAS

> <https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex>

围绕着 Blas Level-3 Function，cuBLAS 提供了数十种不同的矩阵乘法调用。`cublasGemmStridedBatchedEx` 是所有这些调用的集大成者，支持最灵活的功能。

```c
cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void    *alpha,
    const void     *A,
    cudaDataType_t Atype,
    int lda,
    long long int strideA,
    const void     *B,
    cudaDataType_t Btype,
    int ldb,
    long long int strideB,
    const void    *beta,
    void           *C,
    cudaDataType_t Ctype,
    int ldc,
    long long int strideC,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);
```
