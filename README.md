# 算子信息收集

这个仓库用于收集和整理对不同框架和库中算子定义的调研。算子信息采用 json 格式存储以方便自动化处理。

## 格式定义

每个算子在一个领域中的定义保存在一个 json 对象中，结构定义如下：

```json
{
    "operator": "str",
    "domain": "str",
    "reference": "str | [str] | { str: str }",
    "description": "str | [str]"
}
```

- `operator`：算子名字，拼写采用驼峰命名，首批可选的算子包括：
  - RmsNormalization
  - MatMul
  - Rope
  - Attention
  - Swiglu
  - Softmax
  - Add
  - Mul
  - Sigmoid
  - ...（其他，补充更多后移至一个单独的算子列表文件）
- `domain`：定义来源的领域，采用 `/` 隔开的字符串，全小写；
  - 如果是框架，至少 2 级：
    1. 固定以 `framework` 开头；
    2. 然后是框架的名字，一定要做的包括 `pytorch` 和 `paddle`；
    3. 如果不同版本变化较大，可以加一级版本号；
       > 例如：`framework/pytorch/1.10`；
  - 如果是硬件算子库，至少为 3 级:
    1. 固定以 `hardware` 开头；
    2. 然后是厂商的名字，包括 `nvidia`、`cambricon`、`kunlunxin`、`intel` 等，额外添加一个不是厂商名字的 `cpu` 用来表示 CPU 通用的库；
    3. 如果厂商有多款不兼容的产品，下一项为产品的名字；
    4. 然后是算子库的名字；
    5. 如果不同版本算子库差异较大，可以加一级版本号；
       > 例如：`hardware/nvidia/cudnn/8.9.5`、`hardware/intel/mkl`、`hardware/cpu/openblas`；
- `reference`：信息来源和注释，可以有 3 种形式：
  - 一个网址；
  - 一个网址的数组；
  - 一个 `网址 → 注释` 的对象，注释是一个字符串，可以描述从网站的哪部分获得信息，或网站中信息有什么问题之类的；

  对于复杂的算子描述，信息来源可以写的比较简单，然后在 `description` 中内联更详细的网址。

- `description`：接口的具体描述，使用 Markdown 语法。由于大段描述不可避免的需要多行字符串而 json 不直接支持，将多行字符串变通为字符串列表，每行是一个字符串。考虑到人类阅读的美观性，每行可以有尾随空格。例如：

  ```json
  [
      "# Title   ",
      "          ",
      "Summary.  ",
      "          ",
      "## Title 1",
      "          ",
      "Content 1."
  ]
  ```

## 贡献指南

为方便 git 操作，每个对象放在一个独立的 json 文件中，命名为 `operator-domain.json`，全小写中划线隔开，例如 `mat-mul-framework-pytorch-1.10.json`。

- 贡献者可通过 Pull Request 提交调研对象；
- 每次提交尽量只添加或修改一个文件，但如果是对多个文件统一做同类修改也可以在一次提交中完成；
- 每次 PR 中对同一个文件的修改尽量集中在一次提交；
- 请确保 json 和描述中的 Markdown 格式正确（例如 json 不要有尾随逗号），并使用工具进行合理的格式化；
