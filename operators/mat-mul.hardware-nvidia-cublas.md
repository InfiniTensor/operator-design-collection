﻿# MatMul in CuBLAS

> <https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference>

```c
cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const __half          *alpha,
                                         const __half          *A, int lda,
                                         long long int          strideA,
                                         const __half          *B, int ldb,
                                         long long int          strideB,
                                         const __half          *beta,
                                         __half                *C, int ldc,
                                         long long int          strideC,
                                         int batchCount)
cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const float           *alpha,
                                         const float           *A, int lda,
                                         long long int          strideA,
                                         const float           *B, int ldb,
                                         long long int          strideB,
                                         const float           *beta,
                                         float                 *C, int ldc,
                                         long long int          strideC,
                                         int batchCount)
cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const double          *alpha,
                                         const double          *A, int lda,
                                         long long int          strideA,
                                         const double          *B, int ldb,
                                         long long int          strideB,
                                         const double          *beta,
                                         double                *C, int ldc,
                                         long long int          strideC,
                                         int batchCount)
cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const cuComplex       *alpha,
                                         const cuComplex       *A, int lda,
                                         long long int          strideA,
                                         const cuComplex       *B, int ldb,
                                         long long int          strideB,
                                         const cuComplex       *beta,
                                         cuComplex             *C, int ldc,
                                         long long int          strideC,
                                         int batchCount)
cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t handle,
                                           cublasOperation_t transa,
                                           cublasOperation_t transb,
                                           int m, int n, int k,
                                           const cuComplex     *alpha,
                                           const cuComplex     *A, int lda,
                                           long long int        strideA,
                                           const cuComplex     *B, int ldb,
                                           long long int        strideB,
                                           const cuComplex     *beta,
                                           cuComplex           *C, int ldc,
                                           long long int        strideC,
                                           int batchCount)
cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m, int n, int k,
                                         const cuDoubleComplex *alpha,
                                         const cuDoubleComplex *A, int lda,
                                         long long int          strideA,
                                         const cuDoubleComplex *B, int ldb,
                                         long long int          strideB,
                                         const cuDoubleComplex *beta,
                                         cuDoubleComplex       *C, int ldc,
                                         long long int          strideC,
                                         int batchCount)
```