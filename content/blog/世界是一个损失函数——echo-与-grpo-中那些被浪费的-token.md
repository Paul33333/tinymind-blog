---
title: 「世界是一个损失函数」——ECHO 与 GRPO 中那些被浪费的 token
date: 2026-05-19T12:34:08.644Z
---

## 一、引言

昨天看到一篇有意思的[博文](https://x.com/DimitrisPapail/status/2056368948870811746)，介绍了来自 **Microsoft Research AI Frontiers** 团队提出的 [**ECHO**（Environment Cross-entropy Hybrid Objective）](https://github.com/microsoft/echo-rl/blob/main/echo.pdf)——其核心想法是：

在标准 GRPO 强化学习训练 CLI Agent 时，**不再把终端输出 token 从 loss 中 mask 掉**，而是对它们额外加一个交叉熵损失。这个改动只有几行代码，却带来了**几乎零成本的性能翻倍**。

---

## 二、问题背景：GRPO 在浪费什么？

### 2.1 CLI Agent 的交互结构

CLI Agent 的 rollout 天然是一个**交错序列**：

```
[Agent action tokens] → [Terminal output tokens] → [Agent action tokens] → [Terminal output tokens] → ...
```

![Agent框架图.jpg](https://github.com/Paul33333/tinymind-blog/blob/main/assets/images/2025-04-16/1744797871773.jpg?raw=true)

也就是说，模型每执行一个 bash 命令，终端就会返回 stdout/stderr/exit code 等响应。这个响应**已经存在于上下文中，已经通过了模型的前向传播，模型已经为它计算了 logits**。

### 2.2 标准 GRPO 的做法：仅关注 action loss

标准 GRPO（如 [SkyRL](https://github.com/NovaSky-AI/SkyRL) 的实现）的做法是：

> 只在 **action tokens**（模型自己生成的 token）上计算 policy gradient loss，**mask 掉所有 terminal output tokens**，即**terminal output tokens对参数更新没有贡献**

```
[action_1 tokens] → [terminal_output_1 tokens] → [action_2 tokens] → [terminal_output_2 tokens] → ...
     ↑ 有 loss              ↑ mask 掉，没有 loss            ↑ 有 loss              ↑ mask 掉
```

这是一个巨大的浪费，原因有三：

1. **前向传播已经算了** — 这些 **terminal output token** 的 `logits` 已经产生了，mask 掉只是在反向传播时不传梯度，前向计算成本已经支付了。
2. **终端输出是 ground truth** — 环境返回什么就是什么，不存在「标注错误」的问题。
3. **稀疏奖励问题** — 终端任务奖励是稀疏、延迟、二值的。在 Qwen3-8B 的设置中，**不到 15% 的 on-policy rollout 是成功的**。但失败轨迹并不是无用数据：它们仍然包含文件列表、错误信息、堆栈跟踪、grep 输出等丰富的后果信号。

---

## 三、ECHO 的核心方法：加个辅助损失（Aux Loss）

---

### 3.1 公式

ECHO 的损失函数只有一项加法：

$$
\boxed{\mathcal{L}_{ECHO} = \mathcal{L}_{GRPO}(\underbrace{\text{Actions}}_{策略优化}) + \lambda \cdot \underbrace{\mathcal{L}_{env}(\text{Observations})}_{环境预测交叉熵}}
$$

$$\mathcal{L}_{env}(O) = -\frac{1}{|O|}\sum_{t \in O} \log P_{\theta}(o_t | x, a_1, o_1, a_2, o_2, ..., a_k, o_{<t})$$

其中：

- $L_{GRPO}(Actions)$：标准 GRPO 在 **action token** 位置上的策略梯度损失
- $L_{env}(Observations)$：在 **terminal output token 位置**上的**长度归一化交叉熵损失**
  > 这是一个**绝对信号**——让模型更好地预测"环境实际返回了什么"。但它在 rollout 的语境下自动获得了对比效应：
  > Agent 做出 action → 环境返回 observation；
  > 模型学习的不是任意文本预测，而是**给定我的动作，环境如何响应**；
  > 这个信号隐式地区分了哪些 action 导致成功、哪些导致失败。
- $\lambda$：平衡两个目标的超参数

### 3.2 实现代价：几乎为零

论文明确指出：

> "The expensive part of backprop is the matmuls through attention and MLP layers, and those run over the same token sequence regardless of which output positions contribute to the loss."

反向传播最昂贵的部分是 attention 和 MLP 层的矩阵乘法，这些**对所有 token 一视同仁**。Action mask 和 observation mask 只是从同一批 logits 中 gather 不同的子集来计算不同的 loss 项。

**不需要额外的 rollout，不需要 teacher model，不需要额外的 forward pass。** 这就是「for free」的含义。

### 3.3 关键技术细节

| 细节 | 说明 |
|------|------|
| **On-policy 学习** | ECHO 从当前模型自己产生的 terminal response 中学习，而非固定的离线数据集。随着 agent 变好，它探索环境的新区域，获得新的 action→observation 监督信号。**更好策略 → 更好反馈 → 更好预测 → 更好策略**，形成一个正向循环 |
| **λ 超参数** | 太小则环境 loss 不足以塑造模型；太大则策略可能优化「产生可预测输出」而非「完成任务」 |
| **目标 token 选择** | 训练时只对**实际终端输出**计算 loss，不包括 harness warning 等机械性内容。warning 容易记忆，真正的信号在文件名、堆栈跟踪和错误信息中 |
| **基于 SkyRL** | ECHO 作为 SkyRL 的轻量扩展实现，只需一个 minimal hook patch |

---

## 四、实验结果

### 4.1 核心结果：性能接近翻倍

| 基准 | 模型 | GRPO | ECHO | 提升 |
|------|------|------|------|------|
| TerminalBench-2.0 pass@1 | Qwen3-8B | 2.7 | **5.2** | ~2× |
| TerminalBench-2.0 pass@1 | Qwen3-14B | 5.2 | **10.8** | ~2× |
| TerminalBench Lite | Qwen3-8B | — | 显著提升 | — |
| ITD | Qwen3-8B | — | 显著提升 | — |

[原帖](https://x.com/DimitrisPapail/status/2056368948870811746)评论：

> "Performance nearly doubles at no extra cost" is a line you very rarely read across your whole research career

### 4.2 训练速度：2.3 倍加速

ECHO 在 TerminalBench Lite 上**比 GRPO 快 280 步达到同等性能**——在 500 步的总预算中，ECHO 用了约 220 步就达到了 GRPO 500 步的水平。

### 4.3 ECHO 真的学到了终端动力学吗？

这是博客中很有趣的一段。作者非常谨慎地区分了「World Model」（世界模型）和「学到了终端行为模式」：

他们用了一个**干净的 off-policy 测试**：用一个更强的 Qwen3-32B teacher 模型（从未参与训练）生成验证集轨迹，然后测量不同 checkpoint 在这些轨迹的 terminal-output token 上的交叉熵。

结果非常清晰：

- **GRPO 几乎不改变环境 token 的交叉熵**（相对于初始策略）
- **ECHO 显著降低了交叉熵**

这意味着 ECHO 训练的模型在它**自己没有生成的轨迹**上也能更好地预测终端输出——这是对「学到了终端动力学」的操作性证明。

> "A model that's a better predictor is, in information-theoretic terms, a better compressor of the system it's predicting."

---

## 五、两个新发现

### 5.1 发现一：ECHO 可以替代 Expert SFT

这是一个非常重要的结果。当前训练 CLI Agent 的标准流程是：

1. 先用更强模型（如 GLM-4.6）的 expert demonstration 做 SFT（行为克隆，即**蒸馏**）
2. 再跑 GRPO 强化学习

实验对比了三种配置：

| 配置 | 说明 |
|------|------|
| Base → GRPO | 从基础模型直接 GRPO |
| Base → ECHO | 从基础模型直接用 ECHO（无 expert SFT） |
| Base → SFT → GRPO | 先 expert SFT 再 GRPO |

结果：ECHO **恢复了 expert SFT 带来的大部分收益**——在 ITD 上恢复 104%，TBLite 上恢复 89%，TB2 上恢复 50%。

**深层含义**：Expert SFT 的价值很大一部分可能不是教模型「专家会怎么做」，而是教模型**理解终端交互的模式**——即什么命令会产生什么后果。ECHO 不模仿专家策略，而是通过预测终端后果来**自己学会这些交互模式**。更好的策略可以通过交互而非模仿涌现出来。

> "The terminal is the teacher!"

### 5.2 发现二：无需 Reward 的自改进

这是最激进也最令人兴奋的实验。他们把 GRPO loss **完全关掉**，只保留环境交叉熵损失：

$$
L = L_{env}(Observations)
$$

模型只通过**行动→观察→预测终端输出**来学习，没有任何 verifier 告诉它任务是否成功。

结果：

- **val100（分布内）**：+3.8 pp
- **ITD**：+5.2 pp（过滤干净轨迹后）
- **PyTerm（OOD Python 任务）**：**+10.0 pp**

核心发现是**条件性的**：当 rollout 足够干净、终端反馈足够信息丰富时（尤其是 Python 任务中 code → traceback → fix 的密集反馈循环），纯环境预测也能推动策略提升。

> "Not that the agent self-improves perfectly, but that it self-improves at all, from nothing but acting and predicting what comes back."

---

## 六、深层洞察：为什么这个想法如此重要？

### 6.1 「世界是一个损失函数」

这是博客中最哲学化的部分，也是最有启发性的：

> "When an agent acts in an environment, the environment's response to that action is always true."

类比物理世界：你按开关，灯亮或不亮。无论哪种结果，你都获得了关于这个世界如何运作的一小段信息。你不需要理解电学和线路的完整机制，只需要看到结果，就足以开始建立一个关于「按开关如何控制灯」的心智模型。

终端也是如此。`ls` 的输出、error traceback、exit code——这些都是系统状态变化的**低维投影**，足够让模型建立对终端行为的隐式理解。

引用 Ilya Sutskever 的话：

> "Predicting the next token well means that you understand the underlying reality that led to the creation of that token."

ECHO 的操作性版本：**如果模型能很好地预测终端输出，那么它在一个小但真实的意义上，已经建立了对终端的隐式模型。**

### 6.2 失败也是信号

这是 ECHO 与传统 RL 的一个关键差异：

- GRPO：失败轨迹 = 相对奖励为负 = 没什么可学的，仅抑制此acion tokens的激活轨迹的生成概率
- ECHO：失败轨迹 = 仍然包含「这个命令会导致这个错误」的信息 = 有价值的学习信号

> "Even when an action does not solve the task, the terminal response still teaches the model what that action caused! And predicting the consequences of failed actions can help the agent choose better ones."

### 6.3 正向循环

ECHO 的学习过程隐含了一个优雅的正反馈循环：

```
更好的策略 → 探索更多环境状态 → 获得更多样化的 action→observation 对
    ↑                                                          ↓
更好的 action prior ← 更好地预测终端输出 ← 更丰富的环境监督信号
```

这是 **on-policy 学习**天然的优势：模型用自己的探索来给自己提供越来越好的监督信号。

---

## 七、相关工作定位

博客将 ECHO 定位在一条学术脉络中：

| 工作 | 关系 |
|------|------|
| [Agent Learning via Early Experience](https://arxiv.org/abs/2510.08558) | 将 action-consequence 信号作为 RL 前的预训练阶段 |
| [VAGEN](https://arxiv.org/abs/2510.16907) | 为 VLM agent 添加世界建模奖励 |
| [RWML](https://arxiv.org/html/2602.05842v1) | 在 next-state 预测上预训练 |
| [CWM](https://arxiv.org/abs/2510.02387) | 对代码模型在 observation-action 轨迹上做 mid-training |

ECHO 的独特定位是：**在线、在 RL 循环内、面向 CLI 的版本**。它不是预训练阶段，不是单独的 world model，而是把世界建模直接嵌入到 policy gradient 训练中。

---

## 八、局限性与未来方向

博客最后指出的局限和开放问题：

1. **λ 的调节** — 需要平衡环境预测和策略优化，太小没用、太大有害
2. **观察信息量不均匀** — Python 任务的密集反馈（code → traceback → fix）效果最好，更广泛的终端任务反馈更间接
3. **Verifier-free 自改进有条件** — 需要干净的 rollout 和信息丰富的终端反馈
4. **TB2 上还有 50% 的 SFT gap** — **在更难、分布更远的任务上，ECHO 还不能完全替代 expert SFT**
5. **未来方向**：
   - 用摘要或任务相关的状态表示替代原始终端输出
   - 扩展到浏览器 agent、多工具系统、长程编程 agent
   - 在用户交互场景中（follow-ups、corrections、preferences）应用相同思路

> "Our bet is that anywhere an agent acts and the world responds in tokens, those response tokens — or better representations of them — should be part of the learning signal. ... some form of environment-token prediction will be standard in agent RL trainers by the end of 2026."

---

## 九、总结

ECHO 的核心洞见可以用一句话概括：

> **Agent rollout 中包含的不只是最终奖励——每一个终端响应都是免费的监督信号。我们只是不再把它们扔掉。**

这个想法的优雅之处在于：

- 🟢 **极其简单**：几行代码的 mask 改动
- 🟢 **几乎零成本**：同一批 forward pass，同一批 logits
- 🟢 **效果显著**：核心 benchmark 性能翻倍，训练加速 2.3 倍
- 🟢 **减少对 expert SFT 的依赖**：终端本身就是老师
- 🟢 **开启了 verifier-free 自改进的可能性**：即使没有奖励信号，仅靠预测环境反馈也能提升

如果想深入了解，可以查看完整 [论文 (PDF)](https://github.com/microsoft/echo-rl/blob/main/echo.pdf) 和 [开源代码](https://github.com/microsoft/echo-rl)。ECHO 基于 SkyRL 实现，只需要一个小的 hook patch 就能在任何 GRPO trainer 上使用。
