# Softmax in Kunlun

## Softmax
```c++
template<typename T> int softmax(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, int axis)
```
## 参数
* T: **[float,float16]**
* ctx: 程序上下文信息
* x: 输入；`shape=xshape`
* y: 输出；`shape=xshape`
* xshape: x和y的shape信息
* axis: 做softmax的维度
## 功能
$$ y_{i,j,k} = \frac{e^{x_{i,j,k}}} {\sum_{j=0}^{t}{e^{x_{i,j,k}}}} $$
以二维tensor，axis=1的时候为例，对于矩阵的每一行，softmax操作对其进行重新缩放，使得该行的每个元素在 [0,1] 范围内，并且总和为1。这个值通常用来表示概率。