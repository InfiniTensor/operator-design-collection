# 算子信息收集

[![issue](https://img.shields.io/github/issues/InfiniTensor/operator-design-collection)](https://github.com/InfiniTensor/operator-design-collection/issues)
[![issue](https://img.shields.io/github/issues-pr/Infinitensor/operator-design-collection
)](https://github.com/InfiniTensor/operator-design-collection/pulls)

这个仓库用于收集和整理对不同框架和库中算子定义的调研。算子信息采用带有特定结构的 Markdown 文件存储，以方便自动化处理。

## 目录

- [格式定义](#格式定义)
- [贡献指南](#贡献指南)
- [算子列表](#算子列表)

## 格式定义

每个算子在一个领域中的定义保存在一个 Markdown 文件中，文件名为 `operator.domain.md`，采用纯小写，单词之间用 `-` 连接。其中：

- `operator`：算子名字。由于算子的名字用于索引同类算子，必须规范唯一。见[算子列表](#算子列表)；
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
# MatMul in cuBLAS

> <https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmstridedbatchedex>
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

- 建议贡献者先使用 [Issue](https://github.com/InfiniTensor/operator-design-collection/issues) 说明将调研哪些算子的哪些接口，然后应使用 [Pull Request](https://github.com/InfiniTensor/operator-design-collection/pulls) 提交文档，并在 PR 合并时自动关闭 issue；
- commit message 应以 **"update: "** 开头，并说明更新的内容，例如：

  ```Markdown
  update: 添加 `MatMul in cuBLAS`
  ```

- 每次提交尽量只添加或修改一个文件，但如果是对多个文件统一做同类修改也可以在一次提交中完成；
- 每次 PR 中对同一个文件的修改尽量集中在一次提交；
- 请确保 Markdown 格式正确，并使用工具进行合理的格式化。
  > VsCode 用户推荐使用 [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) 插件规范 Markdown 写作。

## 算子列表

希望扩充算子列表的贡献者可以直接提交修改 `README.md` 文件的 PR，以 **"添加算子"** 为题。

| 算子命名 | 中文名
|:-------------------:|:-:
| `rms-normalization` | 均方根归一化
|           `mat-mul` | 矩阵乘
|              `rope` | 旋转位置编码
|         `attention` | 注意力机制
|            `swiglu` | -
|           `softmax` | -
|               `add` | 加
|               `mul` | 乘
|           `sigmoid` | -
