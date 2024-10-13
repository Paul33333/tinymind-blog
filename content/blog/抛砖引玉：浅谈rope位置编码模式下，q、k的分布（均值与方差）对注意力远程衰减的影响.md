---
title: 抛砖引玉：浅谈ROPE位置编码模式下，q、k的分布（均值与方差）对注意力远程衰减的影响
date: 2024-10-13T03:35:07.165Z
---

# 1、引言

今天看到一篇论文：
["ROUND AND ROUND WE GO! WHAT MAKES ROTARY
POSITIONAL ENCODINGS USEFUL?ROUND AND ROUND WE GO! WHAT MAKES ROTARY
POSITIONAL ENCODINGS USEFUL?"](https://arxiv.org/pdf/2410.06205 )

作者在论文中指出：**虽然普遍认为 RoPE 的有用之处在于它有助于随着相对距离的增加而衰减 token 之间的依赖性**（这部分可以参阅苏神的帖子：["Transformer升级之路：2、博采众长的旋转式位置编码"](https://spaces.ac.cn/archives/8265)），**但该论文作者认为这不太可能是主要原因**。因为作者实验发现当$Q、K$都为**均值为0的高斯初始化方案时，远程衰减性并不存在**。
![微信截图_20241012225701.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-10-13/1728787625494.png?raw=true)

---

# 2、实验探查

这一下子颠覆了固有的认知，我觉得还是很有必要做下本地实验，验证下是否的确如此。

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda'
seq_len = 5000
output_dim = 768
frequency = 10000
batch_size = 1

position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
indices = torch.arange(0, output_dim // 2, dtype=torch.float)
indices = torch.pow(frequency, -2 * indices / output_dim)
embeddings = position_ids * indices

embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
embeddings = embeddings.to(device)

pos_emb = embeddings
cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

mean = 0

# initilize the q k with Gaussian distribution
torch.manual_seed(333)
q = torch.normal(mean=mean, std=1.0, size=(1, seq_len, output_dim)).to(device)
k = torch.normal(mean=mean, std=1.0, size=(1, seq_len, output_dim)).to(device)

q2 = torch.stack([-q[..., 1::2], q[...,::2]], -1)
q2 = q2.reshape(q.shape)
k2 = torch.stack([-k[..., 1::2], k[...,::2]], -1)
k2 = k2.reshape(k.shape)

q_rope = q * cos_pos + q2 * sin_pos
k_rope = k * cos_pos + k2 * sin_pos

Activation_score_original = torch.einsum('bmd,bnd->bmn', q, k)
Activation_score_rope = torch.einsum('bmd,bnd->bmn', q_rope, k_rope)

score_decay = torch.flip(Activation_score_rope[0][-1]/torch.max(Activation_score_rope[0][-1]), dims=[0]).cpu().numpy()

plt.plot(score_decay)
plt.title('Activation Decay Test')
plt.xlabel('Seqence_len')
plt.ylabel('Activation score')
plt.title(f'Mean of initial Gaussian Distribution: {mean}')
plt.tight_layout()
plt.show()
```

结果如下：
![v2-fc0939ffc1368748ed611d01cae2c42d_720w.png](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-10-13/1728787697371.png?raw=true)

的确如该论文所言，注意力远程衰减的性质并不存在。

那问题出在哪里呢？毕竟ROPE现在广泛地应用在主流的大模型框架中，效果层面应该是接受了事实的检验的。

> 现在主流大模型突破长文本的优化训练方式之一就是增加$\theta$（比如从10000增加到1000000）

笔者思考后，觉得可以试下对论文中**Proposition 3.2**部分的假设$q, k \backsim N(0, I)$进行放松，查看注意力衰减的效果，具体如下：

---

## 2-1）放松均值为0的假设，查看不同均值情况下的注意力远程衰减效果：

```python
means = list(np.arange(-1, 1.25, 0.25))  # Different mean values for Gaussian distribution
num_subplots = len(means)
num_cols = 2
num_rows = (num_subplots + 1) // 2 

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))  # Adjust the figsize as needed
fig.suptitle('Activation Decay Test')

for i, mean_val in enumerate(means):
  # initilize the q k with Gaussian distribution
  torch.manual_seed(333)
  q = torch.normal(mean=mean_val, std=1.0, size=(1, seq_len, output_dim)).to(device)
  k = torch.normal(mean=mean_val, std=1.0, size=(1, seq_len, output_dim)).to(device)

  q2 = torch.stack([-q[..., 1::2], q[...,::2]], -1)
  q2 = q2.reshape(q.shape)
  k2 = torch.stack([-k[..., 1::2], k[...,::2]], -1)
  k2 = k2.reshape(k.shape)

  q_rope = q * cos_pos + q2 * sin_pos
  k_rope = k * cos_pos + k2 * sin_pos

  Activation_score_original = torch.einsum('bmd,bnd->bmn', q, k)
  Activation_score_rope = torch.einsum('bmd,bnd->bmn', q_rope, k_rope)

  score_decay = torch.flip(Activation_score_rope[0][-1]/torch.max(Activation_score_rope[0][-1]) , dims=[0]).cpu().numpy()

  row_idx = i // num_cols
  col_idx = i % num_cols
  axes[row_idx, col_idx].plot(score_decay)
  axes[row_idx, col_idx].set_xlabel('Seqence_len')
  axes[row_idx, col_idx].set_ylabel('Activation score')
  axes[row_idx, col_idx].set_title(f'Mean of initial Gaussian Distribution: {mean_val}')

plt.tight_layout()
plt.show()
```

结果如下：
![v2-2511808b1fd6b1b136572f485b817f9f_720w (1).webp](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-10-13/1728788242176.webp?raw=true)

我们可以发现：
- 当初始化的高斯分布的**均值绝对值越大**时，**注意力远程衰减性质越明显**；当均值趋于0时，远程衰减性质消失。

上面的实验中，我们默认设置的是$Q_{mean}=K_{mean}$，即均值同向，我们再试验下当$Q_{mean}=-K_{mean}$（均值异向）时的效果，具体设置如下：

```python
torch.manual_seed(333)
q = torch.normal(mean=1, std=1.0, size=(1, seq_len, output_dim)).to(device)
k = torch.normal(mean=-1, std=1.0, size=(1, seq_len, output_dim)).to(device)
```

![v2-cd16daf4e663074c0c89b4e9e77ec89c_720w.webp](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-10-13/1728788692347.webp?raw=true)

可以看到，反而出现了**注意力远程增加**的性质！

综上，我们目前得到的信息是：

**在ROPE位置编码中，注意力远程衰减的性质与$Q、K$的分布均值关系密切；如果$Q、K$分布的均值同向且数值大的话，注意力远程衰减的性质越强；反之，则相反**；

那么反向推理的话，**我们是否可以认为是模型在训练过程中，针对不同的层、不同的注意力头，模型学到了不同的$Q、K$分布（对应不同的分布均值）从而实现了不同的注意力远程衰减的性质呢？**

- 有些层、注意力头可能更注重远程衰减，从而更关注局部信息
- 有些层、注意力头可能基本不具备远程衰减性质，从而更关注全局信息
- 而这些是通过训练实现不同层、注意力头的$Q、K$分布实现的

而$Q, K = xW_{Q},  xW_{K}$

其中，$x$为该层的输入，$W_{Q}、W_{K}$为待训练的权重，为了更好地学习不同的分布均值，保留`bias`选项是否更好呢？

然后我们以Qwen2.5系列模型为例看一下所有的**nn.Linear**层哪些添加了**bias**，哪些没有呢？
![v2-b7ec6dc53e4ab72400804a355c599a09_720w.webp](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-10-13/1728789044224.webp?raw=true)

可见qwen2.5系列为$W_{Q}、W_{K}、W_{V}$权重是保留了**bias**的，其他的则默认都没有保留**bias**，这么巧？

---

## 2-2）放松方差为1的假设，查看不同方差情况下的注意力远程衰减效果：

```python
stds = list(np.arange(1, 3, 0.5))  # Different mean values for Gaussian distribution
num_subplots = len(stds)
num_cols = 2
num_rows = (num_subplots + 1) // 2 

fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))  # Adjust the figsize as needed
fig.suptitle('Activation Decay Test')

for i, std_val in enumerate(stds):
  # initilize the q k with Gaussian distribution
  torch.manual_seed(333)
  q = torch.normal(mean=1, std=std_val, size=(1, seq_len, output_dim)).to(device)
  k = torch.normal(mean=1, std=std_val, size=(1, seq_len, output_dim)).to(device)

  q2 = torch.stack([-q[..., 1::2], q[...,::2]], -1)
  q2 = q2.reshape(q.shape)
  k2 = torch.stack([-k[..., 1::2], k[...,::2]], -1)
  k2 = k2.reshape(k.shape)

  q_rope = q * cos_pos + q2 * sin_pos
  k_rope = k * cos_pos + k2 * sin_pos

  Activation_score_original = torch.einsum('bmd,bnd->bmn', q, k)
  Activation_score_rope = torch.einsum('bmd,bnd->bmn', q_rope, k_rope)

  score_decay = torch.flip(Activation_score_rope[0][-1]/torch.max(Activation_score_rope[0][-1]) , dims=[0]).cpu().numpy()

  row_idx = i // num_cols
  col_idx = i % num_cols
  axes[row_idx, col_idx].plot(score_decay)
  axes[row_idx, col_idx].set_xlabel('Seqence_len')
  axes[row_idx, col_idx].set_ylabel('Activation score')
  axes[row_idx, col_idx].set_title(f'std of initial Gaussian Distribution: {std_val}')

plt.tight_layout()
plt.show()
```

结果如下：
![v2-2e94c76c85aed0206dac382bec89d95f_720w.webp](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2024-10-13/1728789249633.webp?raw=true)

我们可以发现：

当初始化的高斯分布的**方差越大**时，**注意力远程衰减性质越弱**；越小时，注意力远程衰减性质越好。

从这个性质出发，在$W_{Q}、W_{K}、W_{V}$参数初始化环节中设置的方差不要太大，其实也是有助于训练过程中有比较好的注意力远程衰减性质。

> 比如GPT-2系列$W_{Q}、W_{K}、W_{V}$参数初始化的方差设置为0.02

---

# 总结、猜想与不足

ROPE位置编码模式下，**注意力远程衰减的性质与q、k的分布息息相关**。当我们假定q、k分布服从高斯分布的前提下，具体表现为：

- 1）q、k均值同向（都大于0，或者都小于0）时，**均值绝对值越大，注意力远程衰减性质越明显**；
- 2）q、k均值**异向**（一个大于0，一个小于0）时，甚至出现**注意力远程增加**的性质；
- 3）q、k**方差越大，注意力远程衰减性质越弱**，越小时，注意力远程衰减性质越好；

因为经过训练后，不同层下不同注意力头生成的q、k分布各异，所以赋予了不同层、不同注意力头下具备各异的注意力远程衰减力度（有些关注局部、有些关注整体），甚至个别呈现出注意力远程增加性质（适配长上下文的插针测试任务）。

因为q、k的分布与注意力远程衰减的关系，所以$W_{Q}、W_{K}$初始化方案中，设置bias选项默认为`True`更好，初始化的方差不能太大

不足：

本文只能给出实验性的结论，暂未给出ROPE下注意力远程衰减与q、k的分布关系的理论性的推导证明；本文抛砖引玉，期待看到社区大佬们进一步的深入研究。

