---
title: 对比学习中温度超参应该如何初始化 -> 1/sqrt(d)（方差稳定假设下的一个解析解）
date: 2024-12-03T13:15:10.048Z
---

## 引言

在**对比学习**（Contrastive Learning）中，**温度参数**（通常记作 $\tau$）在损失函数中扮演着关键角色。本文将深入探讨温度参数的作用，并论证其合适的一个初始化方案:$\tau = \frac{1}{\sqrt{d}}$（$d$为向量的维度数）。

我们知道，对比学习中的温度参数一般是一个绝对值很低的常数，
比如[CLIP模型](https://arxiv.org/pdf/2103.00020 "CLIP模型")中设置的为0.07，
![clip-温度超参0.07.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-12-02/1733120270820.png?raw=true)

[Improving Text Embeddings with Large Language Models](https://arxiv.org/pdf/2401.00368 "Improving Text Embeddings with Large Language Models")微软的该论文中设置的为0.02 （CLIP的0.07 和微软的0.02，从数值的数量级上看，也和$\frac{1}{\sqrt{d}}$很接近）
![微软的温度超参 0.02.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-12-02/1733120284102.png?raw=true)

而且[A Practitioner’s Guide to Continual Multimodal Pretraining](https://arxiv.org/pdf/2408.14471 "A Practitioner’s Guide to Continual Multimodal Pretraining")该论文中实证了温度超参对于模型训练过程的效果影响很大；
![温度超参不能太大.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-12-02/1733120668451.png?raw=true)

**那么你是否有考虑过为何温度参数要设置的这么低，以及其具体设置为多少合适是否有其深刻的底层原理呢？**

我们接下来先回顾一下**对比学习**以及`InfoNce Loss`损失函数的定义。

---

## 一、对比学习的基本原理

对比学习是一种**自监督学习**方法，旨在学习数据的有用表示，而无需明确的标签。核心思想是：

- **拉近**相似样本的表示（如同一图像的不同增强版本）。
- **推远**不相似样本的表示（如不同图像的表示）。

通过这种方式，模型能够学习到数据的**判别性特征**。

---

## 二、InfoNce Loss损失函数与温度参数

### 1. InfoNCE损失函数

在对比学习中，常用的损失函数是**InfoNCE损失**，形式如下：

$L_i = -\log \frac{\exp\left( \frac{\text{sim}\left( \mathbf{z}_i, \mathbf{z}_i^+ \right)}{\tau} \right)}{\sum_{j=1}^{N} \exp\left( \frac{\text{sim}\left( \mathbf{z}_i, \mathbf{z}_j \right)}{\tau} \right)}$

其中：

- $\mathbf{z}_{i}$：锚点样本的表示向量。
- $\mathbf{z}_i^+$：与锚点相关的正样本的表示向量。
- $\mathbf{z}_j$：所有可能的样本表示（包括正样本和负样本）。
- $\text{sim}(\cdot, \cdot)$：相似度度量函数，通常采用**余弦相似度**。
- $\tau$：**温度参数**，用于调节分布的平滑度。

### 2. 温度参数的作用

温度参数 $\tau$ 调节了相似度值的缩放，影响了模型对相似度差异的敏感程度。

- **缩放相似度值**：
  - **高温度（大 $\tau$）**：相似度值被缩小，差异变小，分布更平滑。
  - **低温度（小 $\tau$）**：相似度值被放大，差异增大，分布更尖锐。
- **影响softmax输出**：
  - **低温度**：softmax输出接近独热向量（one hot），模型更关注最高相似度的样本。
  - **高温度**：softmax输出更平滑，模型在更多样本上分配概率。

---

## 三、论证$\tau = \frac{1}{\sqrt{d}}$

### 1. 首先，回顾高维空间中随机单位向量的统计性质

让我们从最基本的概念开始，逐步探讨高维空间中单位超球面上任意两个单位向量的余弦相似度的分布特性。

#### 1.1 **什么是余弦相似度？**

余弦相似度是衡量两个向量之间相似度的指标，定义为：

$$
\text{余弦相似度} = \cos \theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
$$

其中，$\mathbf{x}$和$\mathbf{y}$是两个向量，$\theta$是它们之间的夹角。对于单位向量（即$\|\mathbf{x}\| = \|\mathbf{y}\| = 1$），余弦相似度简化为它们的点积：

$$
\cos \theta = \mathbf{x} \cdot \mathbf{y}
$$

#### 1.2 **在单位超球面上随机选取两个单位向量，它们的点积有何特性？**

由于单位超球面是对称的，任意方向均等可能，因此两个随机单位向量的点积的期望值为零：

$$
E[\mathbf{x} \cdot \mathbf{y}] = 0
$$

现在我们知道期望了，我们再来看方差；

方差的定义为：

$$
\text{Var}[\mathbf{x} \cdot \mathbf{y}] = E[(\mathbf{x} \cdot \mathbf{y})^2] - (E[\mathbf{x} \cdot \mathbf{y}])^2
$$

已知$E[\mathbf{x} \cdot \mathbf{y}] = 0$，因此：

$$
\text{Var}[\mathbf{x} \cdot \mathbf{y}] = E[(\mathbf{x} \cdot \mathbf{y})^2]
$$

#### 1.3 **计算$E[(\mathbf{x} \cdot \mathbf{y})^2]$**

展开点积的平方：

$$
(\mathbf{x} \cdot \mathbf{y})^2 = \left( \sum_{i=1}^n x_i y_i \right)^2 = \sum_{i=1}^n x_i^2 y_i^2 + 2\sum_{i<j} x_i y_i x_j y_j
$$

由于$\mathbf{x}$和$\mathbf{y}$是独立且在单位球面上均匀分布的随机向量，各分量之间独立且对称。

1. 对于$i \ne j$，$E[x_i y_i x_j y_j] = E[x_i y_i] E[x_j y_j]$。由于$E[x_i y_i] = 0$（因为$x_i$和$y_i$独立且对称分布），因此这些交叉项的期望值为零。
2. 对于$i = j$，$E[x_i^2 y_i^2] = E[x_i^2] E[y_i^2]$。由于$x_i$和$y_i$的分布相同，我们只需计算$E[x_i^2]$。

#### 1.4 **计算$E[x_i^2]$**

在单位球面上，$x_i$的分布满足：

$$
E[x_i^2] = \frac{1}{n}
$$

这是因为单位向量的平方和为1，且各分量对称。

#### 1.5 **得出$E[(\mathbf{x} \cdot \mathbf{y})^2]$**

因此：

$$
E[(\mathbf{x} \cdot \mathbf{y})^2] = \sum_{i=1}^n E[x_i^2] E[y_i^2] = n \left( \frac{1}{n} \right)^2 = \frac{1}{n}
$$

#### 1.6 **计算方差**

因此，点积的方差为：

$$
\text{Var}[\mathbf{x} \cdot \mathbf{y}] = E[(\mathbf{x} \cdot \mathbf{y})^2] = \frac{1}{n}
$$

#### 1.7 **结论**

当维度数$n$足够大时，单位超球面上任意两个单位向量的**余弦相似度**（即点积）的分布具有以下特性：

- **均值为零**：$\mathbb{E}[\mathbf{x} \cdot \mathbf{y}] = \mathbb{E}[\cos\theta] = 0$
- **方差为$\frac{1}{n}$**：$\text{Var}[\mathbf{x} \cdot \mathbf{y}] =\text{Var}[\cos\theta] = \frac{1}{n}$

这意味着，在高维空间中，两个随机单位向量之间的**余弦相似度趋于零**（这是优质的性质，我们就希望初始化的向量表征彼此间尽量不相关），且**分布集中在零附近**，且**随着维度的增加，分布越来越集中**（这是不好的性质，严重不利于学习不同特征间的差异）。这反映了“**高维空间中几乎所有向量都彼此正交**”的现象。

这种现象可以理解为高维空间中的“集中性”，即随机变量的值集中在其期望值附近。对于单位超球面上的向量，其余弦相似度的分布随着维度增加而变得越来越窄，说明随机向量之间的夹角接近于90度。

### 2. **温度参数的理论推导**

为了有效区分正负样本对，所以我们需要放大相似度差异。缩放后的相似度为：

$\tilde{z} = \frac{\cos\theta}{\tau}$

缩放后的方差：

$\text{Var}[\tilde{z}] = \left( \frac{1}{\tau} \right)^2 \cdot \frac{1}{d}$

**为了使初始化的缩放后的方差稳定（设为1）**，应满足：

$\left( \frac{1}{\tau} \right)^2 \cdot \frac{1}{d} = 1 \implies \tau = \frac{1}{\sqrt{d}}$

这表明，当温度参数设置为 $\tau = \frac{1}{\sqrt{d}}$ 时，缩放后的相似度方差为1，具有稳定的统计性质。

---

## 四、为什么我们要对齐**初始化的缩放后的方差稳定为1**这条信仰？

因为我们**参数初始化**方法的思想一般是：**尽量让输入输出具有同样的均值和方差**；这样可以稳定梯度的反向传播，尽量避免梯度消失或者梯度爆炸的问题。

我们都知道transformer架构中注意力的公式中要有一个scale的操作，具体为除以$\sqrt{d}$，其实也是为了对齐缩放后的方差稳定为1这个信仰。

$Attention =  \frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d}}$

其中，已知初始化后的$q,k = xW_{q},  xW_{k}$，且$x$是经过层归一化（不论是标准的`layer_norm`还是`RMS_nrom`），因为$W_{q}、W_{k}$的初始化方案（Xavier初始化或者He初始化），最后都使得$q、k$的范数满足：$E(||q||_{2})=E(||k||_{2})=\sqrt{d}$（$d$为向量嵌入的维度数）。

所以

$Attention =  \frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d}}=\frac{||q||\cdot||k||\cdot\cos(\theta)}{\sqrt{d}}= \frac{\sqrt{d}\cdot \sqrt{d}\cdot \cos(\theta)}{\sqrt{d}}= \sqrt{d}\cdot \cos(\theta)$

那么 $Var(Attention) = Var(\cos(\theta))\cdot \sqrt{d}^{2} = \frac{1}{d}\cdot d = 1$

### 【延伸】：**nGPT-超球面上的规范化Transformer模型**

英伟达前段时间发布的[NGPT ](https://arxiv.org/pdf/2410.01131)就是一个约束了模长的在**单位超球面**上进行学习的规范化Transformer范式（规范化具体指L2_normalize，即大家经常提的qk_norm的操作）。

根据我们前文的分析，如果在执行在单位超球面上的规范初始化，那么

$Attention =  \frac{\mathbf{q} \cdot \mathbf{k}}{scaling factor}=\frac{||q||\cdot||k||\cdot\cos(\theta)}{scaling factor}=\frac{\cos(\theta)}{scaling factor}$

那么当继续约束方差稳定为1的前提下，即
$Var(Attention) = \frac{Var(\cos(\theta))}{scaling factor^{2}} = \frac{1}{d*scaling factor^{2}}= 1$

我们可以计算出：$scaling factor =  \frac{1}{\sqrt{d}}$

所以**超球面上的规范化Transformer模型**的注意力公式应为：

$Attention =  \mathbf{q} \cdot \mathbf{k}  \cdot \sqrt{d}$

我们可以看到论文中也的确是这么做的
![nGPT 归一化参数改变.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-12-03/1733193361238.png?raw=true)

---

## 五、总结

通过从高维空间中随机向量相似度的统计性质出发，我们推导了在方差稳定（二阶距稳定）假设前提下，对比学习中的温度参数应与向量维度的倒数平方根成正比，即：

$\tau = \frac{1}{\sqrt{d}}$

这一选择确保了缩放后的相似度具有稳定的方差，有利于模型的训练和性能提升。

