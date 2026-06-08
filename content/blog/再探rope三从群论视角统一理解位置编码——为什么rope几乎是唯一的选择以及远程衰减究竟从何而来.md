---
title: 再探RoPE（三）：从群论视角统一理解位置编码——为什么RoPE几乎是唯一的选择，以及远程衰减究竟从何而来
date: 2026-06-08T13:54:18.431Z
---

# 1、引言

在前两篇博客中：

- [抛砖引玉：浅谈ROPE位置编码模式下，q、k的分布（均值与方差）对注意力远程衰减的影响](https://github.com/Paul33333/tinymind-blog/blob/main/content/blog/%E6%8A%9B%E7%A0%96%E5%BC%95%E7%8E%89%EF%BC%9A%E6%B5%85%E8%B0%88rope%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81%E6%A8%A1%E5%BC%8F%E4%B8%8B%EF%BC%8Cq%E3%80%81k%E7%9A%84%E5%88%86%E5%B8%83%EF%BC%88%E5%9D%87%E5%80%BC%E4%B8%8E%E6%96%B9%E5%B7%AE%EF%BC%89%E5%AF%B9%E6%B3%A8%E6%84%8F%E5%8A%9B%E8%BF%9C%E7%A8%8B%E8%A1%B0%E5%87%8F%E7%9A%84%E5%BD%B1%E5%93%8D.md)（以下简称"第一篇"）
- [再探RoPE（二）：为什么RoPE + Bias能在远程衰减和长度外推上发挥重要作用？](https://github.com/Paul33333/tinymind-blog/blob/main/content/blog/%E5%86%8D%E6%8E%A2rope%EF%BC%88%E4%BA%8C%EF%BC%89%EF%BC%9A%E4%B8%BA%E4%BB%80%E4%B9%88rope-%2B-bias%E8%83%BD%E5%9C%A8%E8%BF%9C%E7%A8%8B%E8%A1%B0%E5%87%8F%E5%92%8C%E9%95%BF%E5%BA%A6%E5%A4%96%E6%8E%A8%E4%B8%8A%E5%8F%91%E6%8C%A5%E9%87%8D%E8%A6%81%E4%BD%9C%E7%94%A8%EF%BC%9F.md)（以下简称"第二篇"）

笔者通过实验和数学推导，得出了一些关于RoPE远程衰减性质的结论：

> 1. RoPE的远程衰减**并不是无条件成立的**，它高度依赖q、k的分布；
> 2. q、k均值同向且非零时，衰减明显；均值为零时，衰减消失；均值反向时，甚至出现远程增强；
> 3. 方差越大，衰减越弱——这本质上是信噪比的问题；
> 4. 因此为 $W_Q, W_K$ 添加bias项有助于保障长度外推性。

这些结论在当时更多地停留在"实验观察 + 初步推导"的层面，缺少一个**统一的理论框架**来解释"为什么是RoPE？""还有没有其他选择？""远程衰减的本质到底是什么？"

最近我读到了Jane Street Research的一篇高质量博客：[Using group theory to explore the space of positional encodings for attention](https://blog.janestreet.com/using-group-theory-to-explore-positional-encodings-attention/)，作者Alok Puranik用**群论**（Group Theory）对位置编码的全部可能性进行了一次完备的分类，结论极其干净：

> **满足基本公理的位置编码，本质上只有几个家族：无编码（NoPE）、指数衰减、旋转（RoPE）、衰减旋转（Damped RoPE）、以及多项式编码（Jordan块/缺陷矩阵）。**

这个群论框架不仅能解释"RoPE为什么是对的"，还能从更高的视角重新审视我在前两篇中观察到的现象，并将它们放进一个统一的图景中。

本文的目标就是：**用尽可能直观的方式，把Jane Street的群论框架介绍给大家，然后展示它如何与前两篇的实验结论深度联结。**

---

# 2、核心问题：位置编码能是什么？

## 2.1 问题的起点

我们都知道，原始Attention的计算核心是内积 $\langle q, k \rangle$，这个操作是**置换不变的**——你把所有key的顺序打乱，每个query与每个key的内积完全不变。但语言、时间序列的意义深度依赖顺序，所以必须引入位置编码。

位置编码的做法是：用某种依赖时间/位置的函数 $F(t)$ 和 $G(s)$ 分别变换query和key：

$$
\tilde{q}(t) = F(t) \, q(t), \quad \tilde{k}(s) = G(s) \, k(s)
$$

使得attention score $\tilde{q}(t)^\top \tilde{k}(s) = q(t)^\top F(t)^\top G(s) \, k(s)$ 能"感知"位置关系。

那么，$F(t)$ 和 $G(s)$ 可以是什么呢？是不是存在某个我们没有发现的、比RoPE更好的选择？

## 2.2 四条公理

Jane Street的Alok做了一件非常漂亮的事：他不是去搜索新的编码，而是先列出位置编码应当满足的**基本性质**（公理），然后看这些性质本身把设计空间约束到多小。

**公理1：线性。** $F(t)$ 和 $G(s)$ 是矩阵（线性变换）。这是向量空间上最自然的操作。

**公理2：平移不变性。** $F(t)^\top G(s)$ 只依赖相对位置 $t - s$，而不依赖绝对位置。这保证了模型不会因为序列变长就"见到从没训练过的位置值"。

**公理3：归一化。** 当相对位置为零（$t = s$）时，位置编码退化为恒等变换。自己和自己"在同一位置"不需要修正。

**公理4：连续性。** 编码矩阵是位置的连续函数。排除那些需要选择公理才能构造出来的病态函数。

## 2.3 推论：一参数群

从这四条公理出发，可以推导出一个极强的结论。

定义 $A(\tau) = F(t)^\top G(t - \tau)$，即"往回看过去 $\tau$ 步"时对应的编码矩阵。

> 这里 $A(\tau)$ 的符号约定是面向"回望过去"：$\tau > 0$ 时 $A(\tau)$ 作用于更早的key。Jane Street原文的 $A(\Delta) = F(0)^\top G(\Delta)$ 面向"向前推进"（$\Delta > 0$ 时作用于更晚的key），两者在群性质推导上完全等价，只是 $\tau \leftrightarrow -\Delta$ 的符号差异。本文采用"回望"约定是为了与attention score中 $t > s$ 时 $\tau = t-s > 0$ 的自然直观一致。

从公理2和3可以推出：

$$
A(\tau_1 + \tau_2) = A(\tau_1) \cdot A(\tau_2), \quad A(0) = I
$$

这意味着 $\{A(\tau)\}_{\tau \in \mathbb{R}}$ 是一个**群**——满足结合律、有单位元、有逆元——并且映射 $\tau \mapsto A(\tau)$ 是从实数加法群到矩阵群的**连续同态**。

这在数学上叫做**连续的一参数矩阵群**（one-parameter matrix group），它是李群理论中的经典对象。关于它，有一个核心定理：

> **每一个连续的一参数矩阵群都具有形式 $A(\tau) = e^{M\tau}$，其中 $M$ 是某个固定矩阵，称为"生成元"（generator）。**

也就是说，**所有满足四条公理的位置编码，都由一个生成元矩阵 $M$ 通过矩阵指数完全确定。**

问题从"搜索所有可能的矩阵值函数"坍缩成了"分类生成元 $M$ 的所有可能结构"。

---

# 3、生成元分类：位置编码的完备清单

$M$ 的结构由它的特征值决定。根据 $M$ 是否可对角化，分为两大类。

## 3.1 可对角化的情况

如果 $M$ 可对角化，通过基变换可以把 $M$ 化为对角形式，位置编码在每个特征子空间上独立作用。

### （1）实特征值 $\lambda$（一维子空间）

$$
A(\tau)\big|_{1D} = e^{\lambda\tau}
$$

- $\lambda > 0$：远处的key影响**指数爆炸**，不合理，排除。
- $\lambda = 0$：$A(\tau) = 1$，位置编码在这个维度上完全不起作用。→ **NoPE**
- $\lambda < 0$：**指数衰减**，时间越久远的key影响越小。出现在线性Attention变体中（注：仅在因果Attention中有意义）。

### （2）共轭复特征值 $\lambda = \alpha \pm i\theta$（二维子空间）

$$
A(\tau)\big|_{2D} = e^{\alpha\tau} \begin{pmatrix} \cos\theta\tau & -\sin\theta\tau \\ \sin\theta\tau & \cos\theta\tau \end{pmatrix}
$$

- $\alpha = 0$：纯旋转。→ **这就是RoPE！**
- $\alpha < 0$：衰减旋转。→ **这就是RetNet和Mamba-3使用的Damped RoPE！**
> Jane Street原文将复特征值写为 $\lambda = -\mu + i\omega$（$\mu \ge 0$），要求 $\mu > 0$ 产生衰减。本文的 $\alpha$ 对应 $-\mu$，即 $\alpha < 0$ 等价于 $\mu > 0$。两种写法完全等价，只是符号约定不同。
- $\alpha > 0$：增长旋转，不合理，排除。

### （3）小结

可对角化情况下，所有合理的位置编码就是以下几种的混合：

| 生成元特征值类型 | 编码效果 | 已知实例 |
|---|---|---|
| $\lambda = 0$（实） | 无编码 | NoPE |
| $\lambda < 0$（实） | 指数衰减 | 线性Attention衰减因子 |
| $\pm i\theta$（纯虚） | 旋转 | **RoPE** |
| $\alpha \pm i\theta, \alpha < 0$ | 衰减旋转 | RetNet, Mamba-3 |

**没有别的了。**

## 3.2 不可对角化的情况（缺陷矩阵）

如果 $M$ 有重复特征值且不可对角化（存在Jordan块），会产生**多项式因子**。最简单的例子：

$$
M = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad e^{M\tau} = \begin{pmatrix} 1 & \tau \\ 0 & 1 \end{pmatrix}
$$

这个矩阵作用在向量 $(x, v)^\top$ 上的效果是 $(x + v\tau, v)^\top$——恰好是**匀速直线运动**的时间演化！

> Alok证明了，**ALiBi**（那个给attention score加线性惩罚 $-m|t-s|$ 的编码）实际上可以通过扩展q、k维度，用这种缺陷矩阵精确实现！

具体来说，设辅助q分量 $\tilde{q} = (\sqrt{m}, 0)^\top$，辅助k分量 $\tilde{k} = (0, -\sqrt{m})^\top$，则：

$$
\tilde{q}^\top e^{M(t-s)} \tilde{k} = -m(t-s)
$$

在因果Attention中 $t > s$，故 $t-s = |t-s|$，上式恰好给出ALiBi的线性偏置项 $-m|t-s|$。所以ALiBi是缺陷生成元的一个实用实例。
> 注意ALiBi原论文用的是绝对值 $-m|t-s|$，而矩阵推导得到的是 $-m(t-s)$（带符号）。两者在**因果Attention**（$t \ge s$，即query只能看到过去的key）下完全等价。本文后续讨论均默认因果Attention的设定。


## 3.3 完备分类的意义

把上面的结果汇总，我们得到一个完备的位置编码"元素周期表"：

| 生成元类型 | 编码类型 | 衰减形状 | 代表 |
|---|---|---|---|
| $\lambda = 0$ | 无编码 | 无 | NoPE |
| $\lambda < 0$（实） | 指数衰减 | $e^{\lambda\tau}$ | 线性Attention |
| $\pm i\theta$ | 旋转 | 无内禀衰减 | **RoPE** |
| $\alpha \pm i\theta$ | 衰减旋转 | $e^{\alpha\tau}$ | RetNet, Mamba-3 |
| Jordan块 | 多项式 | $\tau^n$ | ALiBi |

所有满足四条公理的位置编码都在这张表里。**设计空间已经被穷尽了。**

这是一个非常安心的结论：我们不需要再绞尽脑汁去发明全新的位置编码——除非你打算放弃某条公理。

然而，有了这个分类之后，一个自然的问题浮现出来：**在RoPE被归类为"纯旋转、无内禀衰减"的前提下，笔者在前两篇博客中反复观察到的远程衰减现象，究竟从何而来？** 带着这个问题，我们进入全文最核心的讨论。

---

# 4、关键洞察：群的性质 vs. 群作用于数据的性质

好了，群论框架讲完了。现在我要引出本文最核心的观察：

> **Jane Street的分类定理回答了"变换规则能是什么"，但它没有回答"给定一个变换规则，attention score的宏观行为如何"。后者取决于变换规则与数据分布的交互作用——而这恰恰是我在前两篇博客中探索的问题。**

具体来说，Jane Street的分类告诉我们，纯RoPE对应的群作用 $A(\tau)$ 在每个二维子空间上是**等距的**（unitary/酉变换）——它既不放大也不缩小任何向量，只是旋转。这意味着**RoPE的群作用本身不包含任何衰减机制**。

但我在第一篇博客的实验中清楚地看到了衰减！这是怎么回事？

答案是：**衰减不是来自群作用，而是来自群作用与q、k分布的交互。**

## 4.1 信号-噪声分解：核心公式

为了用群论的语言精确表述这一观点，我将第二篇博客中的核心公式重新整理如下。在每个二维子空间上，我们将RoPE的群作用写成复平面上的旋转 $A_i(\tau) = e^{j\alpha_i\tau}$，则attention score的期望可以分解为两项：
> 严格来说 $A_i(\tau)$ 在群论框架中是 $2 \times 2$ 旋转矩阵。但当我们把每个二维子空间上的向量用复数 $z = x + jy$ 表示时，旋转矩阵的作用退化为乘以复标量 $e^{j\alpha_i\tau}$。下文为简洁均采用复数表示。

$$
\mathbb{E}[\text{score}(\tau)] = \underbrace{\sum_i \text{Re}\bigl[\mu_{q,i}\overline{\mu_{k,i}} \cdot A_i(\tau)\bigr]}_{\text{信号项：均值向量沿群轨道的内积}} + \underbrace{\sum_i \text{Re}\bigl[\Gamma_i \cdot A_i(\tau)\bigr]}_{\text{噪声项：方差在群轨道上的投影}}
$$

其中 $\mu_{q,i}, \mu_{k,i}$ 是第 $i$ 个二维子空间上 $q, k$ 的均值（复数值），$\Gamma_i = \mathbb{E}[\varepsilon_{q,i}\,\overline{\varepsilon_{k,i}}]$ 是随机分量的协方差。

这个分解告诉我们：attention score的期望由两项构成——一项由确定性均值驱动（信号），一项由随机涨落驱动（噪声）。两者都在群轨道上演化，但命运截然不同。

## 4.2 信号项的物理图像

$\mu_q$ 和 $\mu_k$ 是两个确定的向量（非零均值 = bias贡献）。群 $A(\tau)$ 将 $\mu_k$ 沿着群的轨道（每个二维子空间中的一个圆）"传输"。信号项度量的是**固定的 $\mu_q$ 与"被传输后的 $\mu_k$"之间的对齐程度**。

当 $\tau = 0$（近处），$\mu_k$ 没有被旋转，与 $\mu_q$ 对齐最好，点积最大。

当 $\tau$ 增大（远处），$\mu_k$ 被旋转越来越多，与 $\mu_q$ 的对齐程度发生系统性变化——**这就是远程衰减的来源。**

用一句话说：非零均值向量在群轨道上随 $\tau$ 增大而逐渐"失配"——群提供了旋转的动力，均值提供了可被旋转的"方向"，两者缺一，衰减就不会出现。

## 4.3 噪声项的命运

当 $q, k$ 独立时，$\Gamma_i = 0$，噪声项直接消失。即使 $\Gamma_i \neq 0$，它乘上不同频率的旋转因子 $e^{j\alpha_i\tau}$ 后求和，由于各频率分量随 $\tau$ 增大而快速失相（dephasing），大量独立旋转项的贡献倾向于相互抵消，在长距离上趋向零。

**所以远程衰减的"信号"完全来自均值项，噪声项只贡献随机涨落。**

---

# 5、实验验证：在群论框架下重新审视前两篇的结论

理解了上面的理论框架后，让我们用实验来验证几个新的推论。

## 5.1 实验一：信号项的多频干涉衰减

根据上面的分析，当 $\mu_q, \mu_k$ 同向时，信号项为：

$$
S(\tau) = \sum_{i=1}^{d/2} |\mu_{q,i}||\mu_{k,i}| \cos(\alpha_i \tau)
$$

这是一个**多频cosine的叠加**。我们知道 $\alpha_i = 10000^{-2i/d}$ 是一组不可公度的频率，根据多频叠加的干涉理论，$S(\tau)$ 应该从 $\tau = 0$ 处的峰值逐渐衰减到围绕零的振荡。

这个"衰减"不是指数型的，而更接近**sinc函数**（或更一般的多频干涉包络）的形状。

让我们直接计算这个信号项，不带任何随机噪声，看看纯理论预测长什么样：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

d = 768
half_d = d // 2
base = 10000

# RoPE频率
alphas = base ** (-2 * torch.arange(half_d, dtype=torch.float) / d)

# 纯信号项：假设mu_q = mu_k，各分量均为常数mu
mu = 1.0
signal_coeffs = mu * mu  # |mu_{q,i}| * |mu_{k,i}|，假设各分量相同

taus = torch.arange(0, 5000, dtype=torch.float)

# S(tau) = sum_i |mu_q_i||mu_k_i| * cos(alpha_i * tau)
S = torch.zeros(len(taus))
for idx, tau in enumerate(taus):
    S[idx] = (signal_coeffs * torch.cos(alphas * tau)).sum()

S_normalized = S / S[0]

plt.figure(figsize=(12, 4))
plt.plot(S_normalized.numpy())
plt.title('理论信号项 S(τ)/S(0)：多频cosine叠加的干涉衰减')
plt.xlabel('相对距离 τ')
plt.ylabel('归一化信号强度')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
```

结果如下：

![plot1_signal_decay.png](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-06-07/1780836463487.png)

可以看到：纯信号项（不带任何随机噪声）的确呈现出一种**"从峰值快速下降，然后伴随振荡逐渐归零"**的衰减模式。这不是指数衰减，而是**多频cosine叠加的干涉效应**——和光学中的多缝衍射图案在数学上同构。这就是RoPE + Bias产生远程衰减的理论曲线。

## 5.2 实验二：对比"三种衰减机制"

从群论的分类中，我们知道有三种不同的途径来实现远程衰减：

1. **Damped RoPE**（内禀衰减）：直接修改生成元 $M$，加入负实部 $\alpha < 0$
2. **RoPE + Bias**（干涉衰减）：保持纯旋转，通过非零均值的相消效应实现衰减
3. **ALiBi**（多项式衰减）：使用缺陷生成元的Jordan块

让我们在同一张图上对比它们的衰减曲线：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

d = 768
half_d = d // 2
base = 10000
seq_len = 5000

alphas = base ** (-2 * torch.arange(half_d, dtype=torch.float) / d)
taus = torch.arange(0, seq_len, dtype=torch.float)

# --- 方式1：Damped RoPE（内禀衰减），alpha_decay = -0.001 ---
# 注：这里为简洁假设所有频率分量共享同一个衰减率。
# 在通用设定中，不同二维子空间可以有各自独立的衰减率 alpha_i < 0。
alpha_decay = -0.001
damped_rope = torch.zeros(len(taus))
for idx, tau in enumerate(taus):
    damped_rope[idx] = (torch.exp(alpha_decay * tau) * torch.cos(alphas * tau)).sum()
damped_rope = damped_rope / damped_rope[0]

# --- 方式2：RoPE + Bias（干涉衰减），mu=1 ---
mu = 1.0
rope_bias = torch.zeros(len(taus))
for idx, tau in enumerate(taus):
    rope_bias[idx] = (mu**2 * torch.cos(alphas * tau)).sum()
rope_bias = rope_bias / rope_bias[0]

# --- 方式3：ALiBi（线性衰减），m=0.0005 ---
# 使用clip防止过大的负值（模拟实际softmax前的截断效应）
m = 0.0005
alibi = np.clip(1 - m * taus.numpy(), -0.5, 1.0)
alibi = alibi / alibi[0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(damped_rope.numpy(), color='#1565C0')
axes[0].set_title('Damped RoPE（内禀指数衰减）')
axes[0].set_xlabel('τ')
axes[0].set_ylabel('归一化score')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

axes[1].plot(rope_bias.numpy(), color='#C62828')
axes[1].set_title('RoPE + Bias（多频干涉衰减）')
axes[1].set_xlabel('τ')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

axes[2].plot(alibi, color='#2E7D32')
axes[2].set_title('ALiBi（线性衰减）')
axes[2].set_xlabel('τ')
axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.suptitle('三种衰减机制的理论曲线对比')
plt.tight_layout()
plt.show()
```

结果如下：

![plot2_three_mechanisms.png](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-06-07/1780836501611.png)

三种衰减机制的对比非常直观：

- **Damped RoPE**（左）：指数衰减平滑而快速地收敛到零，振荡被指数包络压制，是最"干净"的衰减。
- **RoPE + Bias**（中）：初始衰减很快，但之后伴随着**明显的振荡**，包络线缓慢下降。这种振荡是多频干涉的特征。
- **ALiBi**（右）：简单直白的线性衰减，没有振荡，但斜率固定，缺乏灵活性。

三者在群论框架中分别对应三种不同类型的生成元：可对角化复特征值（有实部）、可对角化纯虚特征值、不可对角化的Jordan块。

## 5.3 实验三：信噪比决定衰减可观测性

前面的理论分析指出，衰减可观测的条件是**信噪比**足够大，即：

$$
\text{SNR} \sim \frac{|\mu|^2}{\sigma^2} \gg 0
$$

这正是第一篇博客中"均值越大衰减越明显""方差越大衰减越弱"的统一解释。让我们用一组实验来验证：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seq_len = 5000
output_dim = 768
frequency = 10000

position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
indices = torch.arange(0, output_dim // 2, dtype=torch.float)
indices = torch.pow(frequency, -2 * indices / output_dim)
embeddings = position_ids * indices
embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
embeddings = embeddings.unsqueeze(0)
embeddings = torch.reshape(embeddings, (1, seq_len, output_dim)).to(device)

cos_pos = embeddings[..., 1::2].repeat_interleave(2, dim=-1)
sin_pos = embeddings[..., ::2].repeat_interleave(2, dim=-1)

# 不同的信噪比：固定mean=1，改变std
configs = [
    (1.0, 0.1, 'μ=1, σ=0.1 (SNR=100)'),
    (1.0, 0.5, 'μ=1, σ=0.5 (SNR=4)'),
    (1.0, 1.0, 'μ=1, σ=1.0 (SNR=1)'),
    (1.0, 3.0, 'μ=1, σ=3.0 (SNR=0.11)'),
    (0.0, 1.0, 'μ=0, σ=1.0 (SNR=0)'),
    (3.0, 1.0, 'μ=3, σ=1.0 (SNR=9)'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('信噪比(SNR = μ²/σ²)对远程衰减可观测性的影响')

for i, (mean_val, std_val, label) in enumerate(configs):
    torch.manual_seed(42)
    q = torch.normal(mean=mean_val, std=std_val, size=(1, seq_len, output_dim)).to(device)
    k = torch.normal(mean=mean_val, std=std_val, size=(1, seq_len, output_dim)).to(device)

    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], -1).reshape(q.shape)
    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], -1).reshape(k.shape)

    q_rope = q * cos_pos + q2 * sin_pos
    k_rope = k * cos_pos + k2 * sin_pos

    score_rope = torch.einsum('bmd,bnd->bmn', q_rope, k_rope)
    score_decay = torch.flip(score_rope[0][-1] / torch.max(score_rope[0][-1]), dims=[0]).cpu().numpy()

    row, col = i // 3, i % 3
    axes[row, col].plot(score_decay)
    axes[row, col].set_title(label)
    axes[row, col].set_xlabel('相对距离')
    axes[row, col].set_ylabel('归一化score')

plt.tight_layout()
plt.show()
```

结果如下：

![plot3_snr_observability.png](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-06-07/1780836541770.png)

这组实验验证了信噪比理论：

- **SNR=100**（左上，$\mu=1, \sigma=0.1$）：衰减曲线极其清晰，几乎是纯信号。
- **SNR=4**（中上，$\mu=1, \sigma=0.5$）：衰减清晰可见，但噪声开始显现。
- **SNR=1**（右上，$\mu=1, \sigma=1.0$）：衰减仍可辨识，但噪声已经很大——这就是"临界状态"。
- **SNR=0.11**（左下，$\mu=1, \sigma=3.0$）：信号几乎被完全淹没，衰减不可观测。
- **SNR=0**（中下，$\mu=0, \sigma=1.0$）：**完全没有衰减**——纯噪声。这正是第一篇博客以及论文"Round and Round We Go"中观察到的现象。
- **SNR=9**（右下，$\mu=3, \sigma=1.0$）：大均值带来高信噪比，衰减非常明显。

> 第一篇博客中发现的"均值越大衰减越明显""方差越大衰减越弱"，其实都是同一个东西的两面：**信噪比 $\text{SNR} = |\mu|^2/\sigma^2$**。信噪比才是控制远程衰减可观测性的**统一参数**。

## 5.4 实验四：群轨道的几何可视化

最后，让我们直接画出三种情况下群轨道在二维子空间中的样子，建立最直观的几何理解：

![plot4_group_orbits.png](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-06-07/1780836581078.png)

- **纯RoPE**（左）：初始向量 $\mu$（红点）在一个**圆**上运动。内积随着另一个固定向量与圆上运动点之间夹角的变化而振荡衰减。轨道是封闭的、不会收缩——所以衰减完全来自"相位失配"，不来自群作用本身。
- **Damped RoPE**（中）：初始向量 $\mu$ 沿**螺旋线**向原点收缩。无论初始方向如何，最终都会被吸引到原点——这就是"内禀衰减"。
- **$\mu=0$**（右）：没有确定的初始向量，只有原点附近的随机点云。旋转一个"几乎为零"的向量得到的还是"几乎为零"——**没有轨道，没有系统性方向变化，没有衰减**。

以下四张动态GIF进一步展示了群轨道、SNR连续扫描、多频干涉叠加过程以及噪声对轨道影响的动态演化（完整动画复现可以直接运行[create_animations.py脚本](https://github.com/Paul33333/experimental-notebook/blob/main/RoPE/create_animations.py)）：

| 群轨道动态演示 | SNR连续扫描 | 多频干涉叠加过程 | 噪声对轨道的影响 |
|---|---|---|---|
|![animation_orbit.gif](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-06-07/1780835623807.gif) |![animation_snr_sweep.gif](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-06-07/1780835814635.gif) |![animation_freq_buildup.gif](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-06-07/1780835901840.gif) | ![animation_snr_orbit.gif](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2026-06-07/1780835723418.gif) |

- **animation_orbit.gif**：三种群轨道（圆/螺旋/随机点云）的并排动态对比，实时显示内积数值变化。
- **animation_snr_sweep.gif**：固定 $\mu=1$，将 $\sigma$ 从 0.1 连续扫描到 3.5，观察衰减曲线如何从清晰逐渐被噪声淹没。
- **animation_freq_buildup.gif**：逐次叠加频率分量，展示多频干涉图案如何从单个cosine逐步累积到群论预言的衰减包络。
- **animation_snr_orbit.gif**：展示噪声水平连续增大时，圆形轨道如何被逐渐"模糊化"直至不可辨识。

---

# 6、深度联结：在群论框架下重新理解前两篇的每个结论

现在让我逐一回顾前两篇的核心结论，用群论的语言给出更深层的解释。

## 6.1 "均值为零时远程衰减消失"

**群论解释：** 纯RoPE的群作用 $A(\tau)$ 是酉变换（等距映射），它保持向量的模长不变。当 $\mu_q = \mu_k = 0$ 时，q和k都分布在原点附近，没有确定的方向。群作用旋转一个"几乎为零"的向量，得到的还是"几乎为零"的向量——轨道退化为原点附近的随机游走，没有系统性的方向变化，自然不会产生系统性的衰减。

> **一句话：等距群作用旋转零向量还是零向量，所以无信号可衰减。**

## 6.2 "均值同向时衰减明显，反向时远程增强"

**群论解释：** $\mu_q$ 和 $\mu_k$ 是两个确定的非零向量，群 $A(\tau)$ 将 $\mu_k$ 沿圆形轨道传输。

- **同向**：$\tau = 0$ 时，$\mu_q$ 和 $A(0)\mu_k = \mu_k$ 完全对齐，内积最大。$\tau$ 增大后，$A(\tau)\mu_k$ 偏离 $\mu_q$ 的方向，内积下降。这在**每个**频率分量上都成立（$\cos(\alpha_i \cdot 0) = 1$ 是最大值），求和后衰减效应叠加增强。
- **反向**：$\tau = 0$ 时，$\mu_q$ 和 $\mu_k$ 已经反向对齐，内积是极小值。旋转只会让它们从"完全反向"变得"不那么反向"，内积从负值向零回升——这就是"远程增强"。

> **一句话：群轨道上的内积在起点取极值；同向起点是极大值，只能下降；反向起点是极小值，只能上升。**

## 6.3 "方差越大衰减越弱"

**群论解释：** 方差 $\sigma^2$ 对应"噪声"，它不在群轨道上产生系统性的方向变化，只贡献随机涨落。信号项的幅度正比于 $|\mu|^2$，噪声的幅度正比于 $\sigma^2$。衰减能被观测到的条件是信噪比 $\text{SNR} = |\mu|^2/\sigma^2$ 足够大。方差增大时SNR下降，信号被淹没在噪声中。

> **一句话：群轨道上的确定性信号被噪声掩盖了。**

## 6.4 "为 $W_Q, W_K$ 添加bias有助于长度外推"

**群论解释：** bias项为q、k引入了非零均值 $\mu_q, \mu_k$，使得信号项不为零，从而让群作用的旋转能够产生可观测的远程衰减。没有bias时，模型并非完全无法产生远程衰减——它可以通过学习 $W_Q, W_K$ 的权重矩阵，配合某些输入分布，**间接地**产生非零的q、k均值。但这种间接路径比直接加bias更困难、更脆弱：权重矩阵不仅需要学习"如何变换输入"，还需要额外学习"如何产生一个合适的平移量"。将这两重任务解耦——权重负责变换、bias负责平移——是一种更干净的归纳偏置（inductive bias）。这也就是为什么实践中RoPE + Bias的组合在长度外推上表现更好。

> **一句话：bias直接保障了群轨道上有一个非退化的初始向量可供旋转，而不需要权重矩阵去"兼职"产生这个平移。**

---

# 7、一个统一的图景与新的启发

## 7.1 群的结构 vs. 群轨道的动力学

读到这里，你可能已经感受到了一个贯穿全文的主题：

> **Jane Street的框架研究的是"群的结构"（什么变换是合法的），而我前两篇博客研究的是"群轨道的动力学"（给定合法变换，它在实际数据上表现如何）。**

在数学上，这对应于一个经典的区别：**变换规则的定义** vs. **变换规则在特定初始条件下的演化行为**。同一个旋转群，作用在不同的初始条件上（不同的 $\mu$ 和 $\sigma$），可以表现出完全不同的宏观行为——从清晰的远程衰减，到完全无衰减的纯噪声，再到反直觉的远程增强。

这两个层面是正交的、互补的。只有把两者结合起来，才能完整理解位置编码在实践中的表现。

## 7.2 三种衰减机制的深层对比

从群论的视角，我们现在可以清晰地区分三种完全不同的远程衰减机制：

| 衰减机制 | 数学来源 | 衰减形状 | 鲁棒性 | 灵活性 |
|---|---|---|---|---|
| **Damped RoPE** | 修改群作用（$\alpha < 0$） | 指数衰减 $e^{\alpha\tau}$ | 强（无条件衰减） | 弱（衰减率固定） |
| **RoPE + Bias** | 改变被作用对象的分布 | 多频干涉衰减（类sinc） | 弱（依赖学到的分布） | 强（不同头可学不同模式） |
| **ALiBi** | 缺陷生成元（Jordan块） | 线性衰减 $-m\tau$ | 强（无条件衰减） | 弱（衰减率固定） |

这里有一个重要的启发：

> **RoPE + Bias 产生的衰减形状是"多频干涉型"的，它不是单调衰减，而是伴随振荡逐渐归零。** 这在质上不同于Damped RoPE的指数衰减或ALiBi的线性衰减。

从理论曲线看，这种振荡在大距离上幅度较小（参见实验一的plot1_signal_decay.png），但它在中等距离（几百到一千步）上仍有相当的振幅。这暗示了一个有趣的可能性：某些特定距离上的key可能会有注意力的短暂"复苏"——这在直觉上或许能帮助模型捕捉语言中具有特定间距的周期性结构（如段落边界、诗歌韵律等）。不过，这一猜想是否在真实训练后的模型中成立，还需要从实际模型权重中提取q、k分布进行验证。

---

# 8、总结

本文通过引入Jane Street的群论框架，将前两篇博客中的实验发现提升到一个统一的理论视角：

1. **群论的分类定理**告诉我们，满足基本公理的位置编码只有有限几个家族。RoPE（纯旋转）是其中之一，且几乎是唯一自然的"不带内禀衰减"的选择。
2. **RoPE的远程衰减不是群作用的内禀性质**，而是群作用与q、k分布的交互效应。纯旋转是等距的，本身不包含衰减；衰减来自非零均值向量在多频旋转下的干涉相消。
3. **信噪比 $|\mu|^2/\sigma^2$ 是控制衰减可观测性的核心参数**。Bias的作用是保障 $\mu \neq 0$（有信号）；初始化方差不能太大的原因是保障 $\sigma^2$ 不至于淹没信号。
4. **三种衰减机制**（Damped RoPE的内禀衰减、RoPE + Bias的干涉衰减、ALiBi的多项式衰减）对应群论分类中的三种不同生成元结构，各有优劣。

用一句话概括全文：

> **群论告诉我们"时间的流逝只能被编码为旋转、衰减或多项式"；而远程衰减的出现，取决于"被旋转的对象是否有一个明确的方向"——这就是bias在群论视角下的本质作用。**