# MatMul in ONNX

> - <https://onnx.ai/onnx/operators/onnx__MatMul.html>
> - <https://onnx.ai/onnx/operators/onnx__MatMulInteger.html>
> - <https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html>

ONNX 的矩阵乘算子基本遵循 PyTorch 中的定义，尤其是在处理非 2 阶输入张量的行为上。

另外 ONNX 还支持 2 种融合反量化的矩阵乘算子，它们被定义为独立的算子，具有更复杂的参数。

> - name: MatMul (GitHub)
> - domain: main
> - since_version: 13
> - function: False
> - support_level: SupportType.COMMON
> - shape inference: True
>
> This version of the operator has been available since version 13.
>
> **Summary**
>
> Matrix product that behaves like numpy.matmul: <https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html>
>
> **Inputs**
>
> - **A** (heterogeneous) - **T**:
>
>   N-dimensional matrix A
>
> - **B** (heterogeneous) - **T**:
>
>   N-dimensional matrix B
>
> **Outputs**
>
> - **Y** (heterogeneous) - **T**:
>
> Matrix multiply results from A * B
>
> **Type Constraints**
>
> - **T** in ( `tensor(bfloat16)`, `tensor(double)`, `tensor(float)`, `tensor(float16)`, `tensor(int32)`, `tensor(int64)`, `tensor(uint32)`, `tensor(uint64)` ):
>
>   Constrain input and output types to float/int tensors.
