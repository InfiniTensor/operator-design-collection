# 算子信息收集

这个仓库用于收集和整理对不同框架和库中算子定义的调研。算子信息采用带有特定结构的 Markdown 文件存储，以方便自动化处理。

## 格式定义

每个算子在一个领域中的定义保存在一个 Markdown 文件中，文件名为 `operator.domain.md`，采用纯小写，单词之间用 `-` 连接。其中：

- `operator`：算子名字，首批可选的算子包括：
  - rms-normalization
  - mat-mul
  - rope
  - attention
  - swiglu
  - softmax
  - add
  - mul
  - sigmoid
  - ...（其他，补充更多后移至一个单独的算子列表文件）
- `domain`：定义来源的领域；
  - 如果是框架，至少 2 级：
    1. 固定以 `framework` 开头；
    2. 然后是框架的名字，一定要做的包括 `pytorch`、`paddle` 和 `onnx`；
    3. 如果不同版本变化较大，可以加一级版本号；
       > 例如：`framework-pytorch-1.10`；
  - 如果是硬件算子库，至少为 3 级:
    1. 固定以 `hardware` 开头；
    2. 然后是厂商的名字，包括 `nvidia`、`cambricon`、`kunlunxin`、`intel` 等，额外添加一个不是厂商名字的 `cpu` 用来表示 CPU 通用的库；
    3. 如果厂商有多款不兼容的产品，下一项为产品的名字；
    4. 然后是算子库的名字；
    5. 如果不同版本算子库差异较大，可以加一级版本号；
       > 例如：`hardware-nvidia-cudnn-8.9.5`、`hardware-intel-mkl`、`hardware-cpu-openblas`；

下面是一些文件名示例：

- `mat-mul.framework-pytorch-1.10.md`
- `softmax.hardware-nvidia-cudnn-8.9.5.md`
- `add.hardware-cpu-openblas.md`

另外，每个调研文档中需要包含文本的来源，通常是一个文档链接，方便后续追溯。
为了方便自动抓取这个信息，请在 Markdown 的一级标题之后紧跟一个引用块，在引用块中包含一个或一个无序列表中的多个尖括号括起的链接，例如：

```Markdown
# MatMul in CuBLAS

> <https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference>
```

或

```Markdown
# Add in ONNX

> - <https://onnx.ai/onnx/operators/onnx__Add.html#add-14>
> - <https://onnx.ai/onnx/operators/onnx__Add.html#add-13>
> - <https://onnx.ai/onnx/operators/onnx__Add.html#add-7>
> - <https://onnx.ai/onnx/operators/onnx__Add.html#add-6>
> - <https://onnx.ai/onnx/operators/onnx__Add.html#add-1>
```

## 贡献指南

- 贡献者可通过 Pull Request 提交调研对象；
- 每次提交尽量只添加或修改一个文件，但如果是对多个文件统一做同类修改也可以在一次提交中完成；
- 每次 PR 中对同一个文件的修改尽量集中在一次提交；
- 请确保 json 和描述中的 Markdown 格式正确（例如 json 不要有尾随逗号），并使用工具进行合理的格式化；
