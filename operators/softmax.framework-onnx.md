# Softmax in ONNX

> - <https://onnx.ai/onnx/operators/onnx__Softmax.html>

## Softmax

> - name: Softmax (GitHub)
> - domain: main
> - since_version: 13
> - function: True
> - support_level: SupportType.COMMON
> - shape inference: True
>
> This version of the operator has been available since version 13.
>
> **Summary**
>
> The operator computes the normalized exponential values for the given input:
> 
> Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
> 
> The “axis” attribute indicates the dimension along which Softmax will be performed. The output tensor has the same shape and contains the Softmax values of the corresponding input.
>
> **Attributes**
> - **axis** - **INT** (default is '-1'):
> 
>   Describes the dimension Softmax will be performed on. Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
>
> **Inputs**
>
> - **input** (heterogeneous) - **T**:
>
>   The input tensor of rank >= axis.
>
> **Outputs**
>
> - **output** (heterogeneous) - **T**:
>
>   The output values with the same shape as the input tensor.
>
> **Type Constraints**
>
> - **T** in ( `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)` ):
>
>   Constrain input and output types to float tensors.
