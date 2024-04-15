# Softmax in Cambricon

> <https://www.cambricon.com/docs/sdk_1.15.0/cambricon_cnnl_1.21.1/developer_guide/cnnl_api/api/softmax.html#cnnlsoftmaxforward>
> <https://www.cambricon.com/docs/sdk_1.15.0/cambricon_cnnl_1.21.1/developer_guide/cnnl_api/api/softmax.html#cnnlsoftmaxforward-v2>

## Softmax-v1
**API**
```
cnnlStatus_tcnnlSoftmaxForward(
    cnnlHandle_thandle, 
    cnnlSoftmaxAlgorithm_talgorithm, 
    cnnlSoftmaxMode_tmode, 
    const void *alpha, 
    constcnnlTensorDescriptor_tx_desc, 
    const void *x, 
    const void *beta, 
    constcnnlTensorDescriptor_ty_desc, 
    void *y
)
```
Computes a Softmax or Log(Softmax) on the input tensor x, and returns the results in the output tensor y
 
This function is applied to a 3D input tensor only. So you need to reshape the input tensor to be 3D with cnnlTranspose before invoking this operation. Also, you need to reshape the output tensor to original shape with cnnlTranspose after invoking this operation

The scaling factors alpha and beta are not supported currently

**Parametes**

> - **handle**
> 
>   Input. Handle to a Cambricon CNNL context that is used to manage MLU devices and queues in the softmax forward operation
> - **algorithm**
>
>   Input. The algorithm used to compute softmax. The algorithms are defined in the cnnlSoftmaxAlgorithm_t enum
> - **mode**
>
>   Input. The mode used to compute softmax. The modes are defined in the cnnlSoftmaxMode_t enum
> - **alpha, beta**
>
>   Inputs. Reserved for future use. Set the value of these parameters to NULL
> - **x_desc**
>
>   Input. The descriptor of the input tensor
> - **x**
>
>   Input. Pointer to the MLU memory that stores the input tensor
> - **y_desc**
>
>   Input. The descriptor of the output tensor
> - **y**
>
>   Output. Pointer to the MLU memory that stores the output tensor

**Returns**
> CNNL_STATUS_SUCCESS
> 
> CNNL_STATUS_BAD_PARAM
> 
> CNNL_STATUS_ARCH_MISMATCH

**Data Type**
> By the order of input - output, the supported data type combinations are as follows:
> 
> 1. float - float
> 2. half - half
> 3. half - float
>
## Softmax-v2
**API**
```
cnnlStatus_tcnnlSoftmaxForward_v2(
    cnnlHandle_thandle, 
    cnnlSoftmaxAlgorithm_talgorithm, 
    cnnlSoftmaxMode_tmode, 
    cnnlComputationPreference_tprefer, 
    const void *alpha, 
    constcnnlTensorDescriptor_tx_desc, 
    const void *x, 
    const void *beta, 
    constcnnlTensorDescriptor_ty_desc, 
    void *y)
```
Compared with **cnnlSoftmaxForward**, this function supports the parameter of **prefer** that sets the computing with faster speed or higher precision.

**Parameters**
> - **prefer**
>
>   Input. The preference used to compute softmax. 

**Data Type**
> By the order of input - output, the supported data type combinations are as follows:
> 
> 1. float - float
> 2. half - half
> 3. half - float
> 4. bfloat16 - bfloat16
> 5. bfloat16 - float
>
> The bfloat16 data type is supported only on MLU500 series.