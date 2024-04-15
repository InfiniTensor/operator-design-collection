# Softmax in cudnn
> <https://docs.nvidia.com/deeplearning/cudnn/api/cudnn-ops-library.html#cudnnsoftmaxforward>

## Softmax
> **API**
```
cudnnStatus_t cudnnSoftmaxForward(
    cudnnHandle_t                    handle,
    cudnnSoftmaxAlgorithm_t          algorithm,
    cudnnSoftmaxMode_t               mode,
    const void                      *alpha,
    const cudnnTensorDescriptor_t    xDesc,
    const void                      *x,
    const void                      *beta,
    const cudnnTensorDescriptor_t    yDesc,
    void                            *y)
```
> All tensor formats are supported for all modes and algorithms with 4 and 5D tensors. Performance is expected to be highest with NCHW fully-packed tensors. For more than 5 dimensions tensors must be packed in their spatial dimensions.
>
> **parameters**
> - **handle**
> 
>   Input. Handle to a previously created cuDNN context.
> - **algorithm**
>
>   Input. Enumerant to specify the softmax algorithm.
> - **mode**
>
>   Input. Enumerant to specify the softmax mode.
> - **alpha, beta**
>
>   Inputs. Pointers to scaling factors (in host memory) used to blend the computation result with prior value in the output layer as follows:
> 
>   &ensp;&ensp;&ensp;```dstValue = alpha[0]*result + beta[0]*priorDstValue```
> - **xDesc**
>
>   Input. Handle to the previously initialized input tensor descriptor.
> - **x**
>
>   Input. Data pointer to GPU memory associated with the tensor descriptor xDesc
> - **yDesc**
>
>   Input. Handle to the previously initialized output tensor descriptor
> - **y**
>
>   Output. Data pointer to GPU memory associated with the output tensor descriptor yDesc
> 
> **returns**
> - **CUDNN_STATUS_SUCCESS**
>
>   The function launched successfully.
> - **CUDNN_STATUS_NOT_SUPPORTED**
>
>   The function does not support the provided configuration.
> - **CUDNN_STATUS_BAD_PARAM**
>
>   1. The dimensions n, c, h, w of the input tensor and output tensors differ
>   2. The datatype of the input tensor and output tensors differ
>   3. The parameters algorithm or mode have an invalid enumerant value
> - **CUDNN_STATUS_EXECUTION_FAILED**
>
>   The function failed to launch on the GPU