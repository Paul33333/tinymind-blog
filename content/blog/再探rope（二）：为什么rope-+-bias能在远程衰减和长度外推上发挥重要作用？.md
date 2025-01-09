---
title: 再探RoPE（二）：为什么RoPE + Bias能在远程衰减和长度外推上发挥重要作用？
date: 2025-01-09T12:40:55.800Z
---

# 1、引言：RoPE + Bias 有助于长度外推

在[ 抛砖引玉：浅谈ROPE位置编码模式下，q、k的分布（均值与方差）对注意力远程衰减的影响](https://zhuanlan.zhihu.com/p/975380493)该篇博客中，我们特别提到了Qwen2.5系列模型中的 `nn.Linear`层中**只针对**$q,k,v$添加了 `bias`项，也简单给出了笔者的猜测，相关介绍截图如下：

![截图 2025-01-09 14-20-21.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-01-09/1736404031827.png?raw=true)

而当前，其他一些头部模型其实很多是没有添加 `bias`的，比如[deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py)，不添加 `bias`也有诸多解释性的缘由，比如避免过拟合等。那么Qwen2.5系列特意针对$q,k,v$添加了 `bias`项，他们本身的出发点究竟是什么呢？

这个问题终于在Qwen团队发布了Qwen2.5的技术报告[Qwen2.5 Technical Report](https://arxiv.org/pdf/2412.15115)后给出了解释：

![截图 2025-01-09 14-47-22.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-01-09/1736405383171.png?raw=true)

可见Qwen团队引入 `bias`的缘由还是因为苏神在[Bias项的神奇作用：RoPE + Bias = 更好的长度外推性](https://spaces.ac.cn/archives/9577)这篇博客中通过实验测试证明了**添加 `bias`项有助于大模型长度外推**。

---

# 2、问题：为什么加了Bias就有助于长度外推？

大模型长度外推的一大关键是：**当文本序列长度大幅增加后，大模型的注意力需要避免泛滥不聚焦的问题**（关于大模型在长文本情况下的注意力泛滥不聚焦的问题可以参阅[top-k Attention（top-k 稀疏注意力）：一种免再训练的大模型长度外推泛化的推理方法](https://zhuanlan.zhihu.com/p/16144323625)），所以一条分析思路就是：

**长度外推**需要**注意力聚焦不泛滥**，而注意力聚焦不泛滥又需要**设计有效的注意力远程衰减效果**，所以加了 `bias`有助于长度外推其实本质还是因为：

**加了 `bias`有助于实现更有效的注意力远程衰减效果？**

而这又和文章开头提到的[ 抛砖引玉：浅谈ROPE位置编码模式下，q、k的分布（均值与方差）对注意力远程衰减的影响](https://zhuanlan.zhihu.com/p/975380493)该篇博客中的实验结论相契合了：

其实，加了 `bias`等价于影响了$q,k$分布的均值，或者说导致**向量有一个明显的‘’基底‘’偏移量** ，而这有助于RoPE下实现预期的注意力远程衰减的效果，我们接下来会：

1）先对这个观点进行几何角度的形象解读；

2）最后再附上一份不太严谨的数学证明。

其中，解读部分我们会采取层层递进的方式进行演绎，以期将概念拆解的更直观易懂。

---

# 3、几何角度的解读：“有偏置” (Bias) + “旋转”(RoPE) 会带来远程衰减

> 问题：把一个带‘基底偏置’的向量，再加上一个‘旋转’（比如 RoPE 里的旋转矩阵）时，会发生什么呢？

1. **旋转与基底的叠加**
   * 在 RoPE 中，我们会对 $\mathbf{q}$或$\mathbf{k}$的部分分量（deepseek的**MLA架构**就是严格的部分分量）施加相位旋转，类似在“复平面”里把某些坐标绕着原点转一个角度。
   * 如果 **$\mathbf{q}$本身没有什么偏置**（大致均值为 0），那它在旋转时就只是在“原点附近”轻微摆动，远距离时就容易呈现某种随机噪声，不一定会有明确的随距离增减的趋势。
   * **但是如果$\mathbf{q}$带了一个明显的“基底方向”** ，那么这个“基底”就像一个大矢量坐落在复平面的某个角落；在它的基础上再旋转，就相当于**不断给一个非零起点叠加越来越大的相位偏移** 。这时候，相位和基底之间的夹角、长度就会发生系统性的变化。
2. **同向 vs. 反向**
   * 如果$\mathbf{q}$与 $\mathbf{k}$这两个基底方向“大致同向”（也都偏正或者都偏负），那一开始它们可能对齐得很好（$\theta$很小，$\cos(\theta)$很大），点积就非常大；但随着旋转累积，二者会快速出现更大的夹角，$\cos(\theta)$下降就很明显，导致点积随距离**衰减**。
   * 若 $\mathbf{q}$与$\mathbf{k}$的基底方向“相反”（一个是正偏置，一个是负偏置），一开始它们或许就呈现出负的点积（$\theta\approx 180^\circ$）。但随着旋转继续，它们可能逐渐“转到”某个角度，$\cos(\theta)$反而从负数向正数跃迁，出现了某种“远程增强”的现象。
3. **为什么是衰减或增强？**
   * 可以把这个过程想象成“在一个大圆上有两个点，起初或许重叠度很高，随着绕圆周的旋转，一方离另一方变得更远或更近”。只要有一个常驻的“偏置矢量”在场，旋转就意味着逐渐失配或逐渐变得更匹配，于是随距离（旋转角度）的增大，点积要么更快降低，要么出现意外上涨。
   * 这也就是我们在[ 抛砖引玉：浅谈ROPE位置编码模式下，q、k的分布（均值与方差）对注意力远程衰减的影响](https://zhuanlan.zhihu.com/p/975380493)该篇博客的实验里看到的： **“正偏置 + 正偏置” → 远程衰减；“正偏置 + 负偏置” → 远程增强** ，背后只是“点积 + 旋转”的几何图景在发挥作用。

再问：

> 为什么没有偏置时就不明显了？

* 没有偏置，意味着$\mathbf{q}$和$\mathbf{k}$经常分布在原点周围并随机摆动，相当于你无法确定它们有一个‘特定的方向’在旋转过程中一直发挥作用。
* 这就像你拿一个非常小、随机分布在圆心周围的短向量去做旋转， **旋转幅度再大，它对彼此点积的影响也无规律可言** ——时而对齐，时而相反，时而正交，都只是噪声。自然就不显现出一个清晰的随距离衰减或增强的模式。

总结：

> **“偏置 + 旋转 → 远程衰减（或增强）”，本质上就是一个“大矢量上不断叠加相移”在点积世界里如何造成数值随距离变化的几何现象。**

1. **若两大矢量同向** ：一开始点积很大，但随着旋转，夹角增大快，数值就会随距离衰减。
2. **若两大矢量反向** ：一开始点积很小甚至为负，但旋转可能慢慢扭转局势，出现远程增强。
3. **若无明显偏置** ：旋转只会在原点附近打转，难以形成显著的远程衰减或增强。

这就是我们所看到的“在纯粹向量点积模型中（即注意力机制），偏置（均值）如何决定了旋转后的远程衰减/增强”背后的直观几何解释。

---

# 4、数学推导

以下部分会尝试给出一个比较系统的、**从理论层面**去解释「RoPE下的注意力远程衰减现象如何与 $q,k$ 的分布(均值/方差) 相关」的推导思路。为了使推导更聚焦，我们做如下简化假设：

1. **只考虑单头注意力**（不去区分多头），并把所有向量的维度看作 $d$。
2. **将 RoPE 视作在每对坐标（2维）上执行相同的“旋转”操作**，即用复数表示会更直观。
3. **$q, k$ 均来自某个多元高斯分布**（是否有非零均值、不同方差），我们重点关心它们的均值向量 $\boldsymbol{\mu}_q,\boldsymbol{\mu}_k$ 和协方差矩阵 $\Sigma$。

> **注：** 这份推导并不是 100% 严谨完备的“定理+证明”形式，而是帮助我们看清：
> 
> - 当 $\boldsymbol{\mu}_q,\boldsymbol{\mu}_k$ 为零或很小的时候，RoPE并不会显著地产生远程衰减；
> - 当 $\boldsymbol{\mu}_q,\boldsymbol{\mu}_k$ 同向且较大时，会带来随位置差增大的相位错配，从而引发远程衰减；
> - 当二者方向相反或方差过大时，又会导致远程增强或衰减消失。

---

## 4.1、 RoPE 的数学表征：用复数表示旋转

为了更方便地处理“旋转”操作，我们先把维度 $d$ 视作由 $\frac{d}{2}$ 组二维坐标组成。在每组二维坐标 $(x_{2i}, x_{2i+1})$ 上，RoPE会做一个如下变换（对第 $i$ 组位置而言）：

$$
\begin{pmatrix}
x_{2i} \\
x_{2i+1}
\end{pmatrix}
\;\longmapsto\;
\begin{pmatrix}
\cos(\theta_i) & -\sin(\theta_i) \\
\sin(\theta_i) & \cos(\theta_i)
\end{pmatrix}
\begin{pmatrix}
x_{2i} \\
x_{2i+1}
\end{pmatrix}.
$$

在实际的 RoPE 实现中，$\theta_i$ 会和“序列位置”$p$ 以及一个频率刻度 $\alpha$（常见是 $\alpha \approx \frac{1}{10000^{2i/d}}$）相关，比如

$$
\theta_i(p) \;=\; \alpha_i \cdot p,\quad \text{其中 }\alpha_i = 10000^{-\frac{2i}{d}}.
$$

**用复数形式**会更简洁：记

$$
z_i = x_{2i} + j \,x_{2i+1} \quad (\text{其中 } j^2 = -1),
$$

那么二维旋转就是与复数的乘法

$$
z_i \;\mapsto\; z_i \,\exp\bigl(j\,\theta_i(p)\bigr).
$$

如果 $\mathbf{q}\in\mathbb{R}^d$ 表示某个 token 的向量，那么在位置 $p$ 处的 RoPE 变换可写为**复数向量**

$$
\mathbf{q}^\text{rope}(p) \;=\;
\Bigl(z_1\,e^{j\,\theta_1(p)},\;z_2\,e^{j\,\theta_2(p)},\;\dots,\;z_{\frac{d}{2}}\,e^{j\,\theta_{\frac{d}{2}}(p)}\Bigr).
$$

同理，对 $\mathbf{k}$ 也是一样，只不过位置可能是 $p'$，会得到

$$
\mathbf{k}^\text{rope}(p') \;=\;
\Bigl(w_1\,e^{j\,\theta_1(p')},\;w_2\,e^{j\,\theta_2(p')},\;\dots,\;w_{\frac{d}{2}}\,e^{j\,\theta_{\frac{d}{2}}(p')}\Bigr).
$$

---

## 4.2、注意力分数 $\mathbf {q}^\text {rope}\cdot \mathbf {k}^\text {rope}$ 的表达

### 4.2.1 在复数形式下的“点积”

我们关心的是：

$$
\mathbf{q}^\text{rope}(p)\;\cdot\;\mathbf{k}^\text{rope}(p')
\;=\;
\sum_{i=1}^{\frac{d}{2}}
\text{Re}\Bigl(z_i\,e^{j\theta_i(p)} \cdot \overline{\bigl(w_i\,e^{j\theta_i(p')}\bigr)}\Bigr),
$$

这里 $\overline{\cdot}$ 表示复共轭，$\text{Re}(\cdot)$ 表示取实部。将乘法展开可得

$$
= \sum_{i=1}^{\frac{d}{2}}
\text{Re}\Bigl(z_i \,\overline{w_i}\Bigr)
\;\exp\Bigl(j\,(\theta_i(p) - \theta_i(p'))\Bigr).
$$

为简化记号，记

$$
\Delta_i(p,p') \;=\; \theta_i(p) - \theta_i(p'),
$$

那么

$$
\mathbf{q}^\text{rope}(p)\;\cdot\;\mathbf{k}^\text{rope}(p')
\;=\;
\sum_{i=1}^{\frac{d}{2}}
\text{Re}\Bigl[\bigl(z_i\,\overline{w_i}\bigr)
\,e^{j\,\Delta_i(p,p')}\Bigr].
$$

### 4.2.2 若 $\theta_i(p)$ 线性依赖于位置 $p$

在常见的RoPE中，$\theta_i(p) = \alpha_i\,p$，于是

$$
\Delta_i(p,p') \;=\;\alpha_i\,(p - p').
$$

对“远距离”而言（比如 $|p - p'|$ 很大），$\Delta_i$ 也就很大，从而会在复平面里产生较大幅度的旋转。

---

## 4.3、$q, k$ 的分布：从高斯随机向量到期望点积

为了研究“远程衰减”这类宏观现象，最直接的方法之一就是去**计算其期望值**（或者均值的模量等统计量），即

$$
\mathbb{E}\Bigl[
\mathbf{q}^\text{rope}(p)\;\cdot\;\mathbf{k}^\text{rope}(p')
\Bigr].
$$

若我们能证明随着 $|p - p'|$ 增大，此期望值呈现**下降**（衰减）或其他行为，就能为实验结论提供一定的理论解释。

### 4.3.1 假设：$\mathbf {q}, \mathbf {k}$ ~ 多元正态分布

令

$$
\mathbf{q} = (z_1, z_2, \dots, z_{d/2}),\quad
\mathbf{k} = (w_1, w_2, \dots, w_{d/2}),
$$

在复数视角里，假设它们满足：

$$
\begin{cases}
z_i = \mu_{q,i} + \varepsilon_{q,i}, \\
w_i = \mu_{k,i} + \varepsilon_{k,i},
\end{cases}
$$

其中 $\varepsilon_{q,i}, \varepsilon_{k,i}$ 是满足某些（复）高斯分布的随机量，具有协方差 $\Sigma$，均值为 0；$\mu_{q,i},\mu_{k,i}$ 则是**可学习的均值向量**。为了不掩盖主要思想，常见简化包括：

1. **假设 $\mathbf{q}, \mathbf{k}$ 独立**（或在同一个 token 上是同分布，但不同 token 也相互独立）；
2. **假设各分量之间独立且方差相同**（“同方差”），即 $\varepsilon_{q,i}\sim \mathcal{N}(0,\sigma^2)$ 等价于复正态(均值0,方差$\sigma^2$)。

这样，我们就可以把期望拆成**均值部分** + **噪声协方差** 的两部分。

### 4.3.2 期望展开

我们要计算

$$
\mathbb{E}\Bigl[
\sum_{i=1}^{d/2}
\text{Re}\Bigl( (z_i\overline{w_i}) e^{j\,\Delta_i} \Bigr)
\Bigr].
$$

注意到

$$
z_i\overline {w_i} =
(\mu_ {q,i}+\varepsilon_ {q,i})\,\overline {(\mu_{k,i}+\varepsilon_ {k,i})}
=
\mu_{q,i}\,\overline{\mu_{k,i}}
\;+\;\mu_{q,i}\,\overline{\varepsilon_{k,i}}
\;+\;\varepsilon_{q,i}\,\overline{\mu_{k,i}}
\;+\;\varepsilon_{q,i}\,\overline{\varepsilon_{k,i}}.
$$

令

$$
\Gamma_{i}
\;=\;
\mathbb{E}\bigl[\varepsilon_{q,i}\,\overline{\varepsilon_{k,i}}\bigr]
\quad\bigl(\text{这在独立同分布时通常为 }0\text{或 } \sigma^2\delta_{q,k}\bigr),
$$

其中如果 $\mathbf{q}$ 和 $\mathbf{k}$ 独立，则 $\Gamma_i=0$。若 $\mathbf{q}=\mathbf{k}$（自注意力同一个embedding），会有 $\Gamma_i=\sigma^2$ 之类的形式。

那么期望拆解如下：

$$
\mathbb{E}\bigl[z_i\overline{w_i}\bigr]
\;=\;
\mu_{q,i}\,\overline{\mu_{k,i}}
\;+\;
\mu_{q,i}\,\mathbb{E}[\overline{\varepsilon_{k,i}}]
\;+\;
\overline{\mu_{k,i}}\,\mathbb{E}[\varepsilon_{q,i}]
\;+\;
\Gamma_i.
$$

由于 $\varepsilon_{q,i},\varepsilon_{k,i}$ 均值为0，$\mathbb{E}[\overline{\varepsilon_{k,i}}]=0$，则简化为：

$$
\mathbb{E}\bigl[z_i\overline{w_i}\bigr]
\;=\;
\mu_{q,i}\,\overline{\mu_{k,i}}
\;+\;\Gamma_i.
$$

因此，

$$
\mathbb{E}\Bigl[
\mathbf{q}^\text{rope}(p)\cdot \mathbf{k}^\text{rope}(p')
\Bigr]
\;=\;
\sum_{i=1}^{d/2}
\text{Re}\Bigl[
\bigl(\mu_{q,i}\,\overline{\mu_{k,i}} + \Gamma_i\bigr)
\,\mathbb{E}\bigl[e^{j\,\Delta_i}\bigr]
\Bigr].
$$

假设 $\mathbf{q},\mathbf{k}$ 与位置无关（即不会因为句子位置改变它们的分布），则 $\Delta_i$ 是确定的（和随机 $\varepsilon_{q,i}$ 无关）。故

$$
\mathbb{E}\bigl[e^{j\,\Delta_i}\bigr] = e^{j\,\Delta_i}.
$$

这时

$$
= \sum_{i=1}^{d/2}
\text{Re}\Bigl[
\bigl(\mu_{q,i}\,\overline{\mu_{k,i}} + \Gamma_i\bigr)
\,e^{j\Delta_i}
\Bigr].
$$

---

## 4.4、远程衰减/增强的来源：均值项 vs. 协方差项

由上式可见，期望分为两部分：

1. **来自 $\mu_{q,i}\,\overline{\mu_{k,i}}$ 的贡献**
   
   $$
   \sum_{i=1}^{d/2}
   \text{Re}\Bigl[
   \mu_{q,i}\,\overline{\mu_{k,i}}
   \,e^{j\,\Delta_i}
   \Bigr].
   $$
   
   如果 $\mu_{q,i},\mu_{k,i}$ 都是非零（并且方向类似），那么这是一个随 $\Delta_i$ 变动而**可能产生周期性衰减或增强**的主导项。因为 $\Delta_i$ 会随着 $|p-p'|$ 增大而带来“相位不断积累”，让实部出现规律性的正负波动，整体常表现为**平均值随距离增大而变小**（衰减），或者在方向相反时出现另一种远程现象（远程增强）。
2. **来自 $\Gamma_i$（噪声协方差）的贡献**
   
   $$
   \sum_{i=1}^{d/2}
   \text{Re}\Bigl[
   \Gamma_i \, e^{j\,\Delta_i}
   \Bigr].
   $$
   
   - 若 $\mathbf{q},\mathbf{k}$ 独立($\Gamma_i = 0$)，则这部分不贡献任何平均值；
   - 若是**同一个向量**($\mathbf{q}=\mathbf{k}$，自注意力场景并且分量独立方差$\sigma^2$)，则 $\Gamma_i=\sigma^2$是一个常量，但乘上 $e^{j\,\Delta_i}$ 仍会在复平面震荡，最后再取实部求和，通常会**随 $|p-p'|$ 呈振荡状，并且可能均值接近零**。
   - 这解释了**若 $\mu_{q}=\mu_{k}=0$** 并且是大方差，则我们不会在期望上看到明显的远程衰减，更多地是一种随机震荡/噪声。

因此，从这里我们可以初步得出一个重要结论：

> **若想让 $\mathbf{q}^\text{rope}(p)\cdot \mathbf{k}^\text{rope}(p')$ 在平均意义上呈现“随 $|p-p'|$ 增大而单调衰减”的趋势，最关键的一环就是 $\mu_{q,i},\mu_{k,i}$ 的非零部分如何与相位旋转相互作用。**

- 当 $\mu_q,\mu_k$ **同向且绝对值不小**，这会导致随着位置差变大，“相位失配”越来越严重，点积的实部下降显著；
- 当 $\mu_q,\mu_k$ **方向相反**，在小距离时它们就相互抵消，随着距离增大，可能因相位而“翻转”导致某些远程增强效应；
- 当 $\mu_q,\mu_k\approx 0$ 或方差无比大，则噪声占主导，让旋转信号变得难以在期望上显现有序的衰减/增强。

---

## 总结

**在 RoPE 机制下，$q,k$ 的“非零均值”扮演了“基底向量”的角色，而bias项的加入就是为非零均值助力。**

**当存在这个‘基底向量’时，随距离增大的旋转相位会与它越来越错配，使得点积平均值随距离衰减；反之，若均值为 0 或方向相反，衰减就不明显或甚至变为远程增强。而注意力远程衰减又是保障大模型长度外推的关键环节，所以为了更好的长度外推性质，建议实施：RoPE+Bias。**
