---
title: 世界是一个损失函数（续）——从 ECHO 到 World Model：也许 Planner 与 Simulator 从来不是两个东西
date: 2026-06-04T14:23:07.838Z
---

## 引言

两周前写完《[世界是一个损失函数—— echo 与 grpo 中那些被浪费的token](https://github.com/Paul33333/tinymind-blog/blob/main/content/blog/%E4%B8%96%E7%95%8C%E6%98%AF%E4%B8%80%E4%B8%AA%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%E2%80%94%E2%80%94echo-%E4%B8%8E-grpo-%E4%B8%AD%E9%82%A3%E4%BA%9B%E8%A2%AB%E6%B5%AA%E8%B4%B9%E7%9A%84-token.md)》之后，我一直觉得 ECHO 最有趣的地方还没有被完全说透。

表面上看，ECHO 做的事情非常简单：

* GRPO 只在 action token 上计算 loss
* ECHO 额外在 observation token 上增加一个交叉熵损失

代码层面的改动甚至只有一个 mask。

但奇怪的是：

这个额外的 next-token prediction loss，居然提升了 policy。

为什么？

如果只是把它理解成：

> 「利用了更多 token」

那么实验结果未免有些过于强烈。

今天看到李飞飞刚发的关于世界模型（World Models）的长文：[世界模型的功能分类学](https://x.com/drfeifei/status/2062247238143996275)，突然意识到：

ECHO 可能不仅仅是一篇 RL 论文。

它也许触碰到了一个更大的问题：

> 为什么预测世界，会提升行动能力？

---

## 一、李飞飞的分类学：Renderer、Simulator、Planner

李飞飞将当前被称为 World Model 的系统划分为三类：

| 类别        | 输出          | 回答的问题     |
| --------- | ----------- | --------- |
| Renderer  | Observation | 我会看到什么？   |
| Simulator | State       | 世界实际上是什么？ |
| Planner   | Action      | 下一步应该做什么？ |

其背后的统一结构来自 POMDP：

```text
Agent
  ↓
Action
  ↓
State
  ↓
Observation
  ↓
Agent
```

不同世界模型的区别，本质上是：

> 它们选择预测这个循环中的哪一部分。

---

## 二、ECHO 中隐藏的 POMDP

CLI Agent 的 rollout：

```text
command
→
terminal output
→
command
→
terminal output
→ ...
```

可以自然对应为：

```text
Action
→
Observation
→
Action
→
Observation
```

文件系统状态则对应隐藏的 State。

需要强调：

这里真正重要的不是 POMDP 映射本身。

因为几乎任何交互系统都能映射到 POMDP。

真正重要的是：

> Loss 落在循环的哪里。

---

标准 GRPO：

```text
Loss
↓
Action Tokens
```

ECHO：

```text
Loss
↓
Action Tokens

+
Observation Tokens
```

区别仅此而已。

然而效果却显著提升。

这意味：

> ECHO 第一次让 Observation 端成为训练目标的一部分。

---

## 三、世界模型的一个操作性定义

李飞飞从功能角度定义世界模型：

> 学习空间与时间中的统计结构。

这是一个描述性定义。

而 ECHO 给出了另一种更操作化的定义：

如果一个模型能够预测：

```text
Action
↓
Observation
```

之间的映射，

那么它就在学习世界。

对于 CLI Agent：

```text
ls
→
会出现哪些文件

grep
→
会返回哪些内容

rm
→
会产生什么后果
```

这些都属于：

> 动作导致什么环境反馈。

换句话说：

世界模型未必一定是物理引擎。

它也可以是：

> 关于环境反馈的预测能力。

---

## 四、失败为什么有价值

李飞飞认为：

模拟器最大的价值之一，是允许失败。

因为：

```text
撞车
↓
获得数据
↓
学会不撞车
```

而 ECHO 做的是：

```text
报错
↓
获得数据
↓
学会不报错
```

标准 GRPO：

失败轨迹几乎只有负奖励价值。

ECHO：

失败轨迹仍然包含：

* traceback
* error message
* 文件信息
* 编译信息

这些都是环境因果结构的直接体现。

于是：

> 失败不再只是负样本，失败变成世界模型监督信号。

---

## 五、Verifier-Free 的真正含义

论文最令人震惊的结果之一：

关闭 GRPO。

仅保留：

```math
L_{env}
```

模型依然变强。

这意味着：

```text
预测环境
↓
策略提升
```

在某些条件下是成立的。

而且：

中间没有奖励。

---

这实际上触碰到了一个更深的问题：

到底是奖励塑造智能，

还是预测塑造智能？

传统 RL 的答案：

```text
Reward
→
Policy
```

World Model 路线的答案：

```text
Prediction
→
Policy
```

ECHO 站在两者交汇的位置。

---

## 六、World Model 与 Policy 的边界开始模糊

到这里为止，

我们仍然可以说：

```text
World Model
+
Planner
```

是两个模块。

只是 ECHO 同时训练它们。

但问题在于：

Transformer 本身似乎并不这样认为。

---

对于 Transformer：

整个 rollout 其实只是：

```text
command_1
stdout_1
command_2
stdout_2
...
```

一个 token 序列。

模型内部并不知道：

```text
这是 Action

这是 Observation
```

这些区分都是训练者后加上的。

对于模型而言：

> 它们都只是 token。

---

唯一的区别在于：

```text
Loss 打在哪里
```

---

GRPO：

```text
Action Token
参与梯度
```

ECHO：

```text
Action Token
+
Observation Token
参与梯度
```

模型结构没有变化。

数据没有变化。

Rollout 没有变化。

变化的只有监督信号。

---

## 七、也许世界模型不是模块，而是训练目标

这是 ECHO 最值得深思的地方。

在经典强化学习里：

```text
World Model
负责预测

Planner
负责决策
```

两者通常是不同模块。

Dreamer 如此。

MuZero 如此。

机器人系统也大多如此。

---

但 Transformer 提供了另一种可能。

对于它而言：

预测未来 Observation：

```text
Predict Future Observation Tokens
```

和预测未来 Action：

```text
Predict Future Action Tokens
```

本质上是同一种操作：

```text
Predict Future Tokens
```

---

于是一个更激进的视角出现了：

> World Model 也许不是一个模块。

而是一种训练目标。

当我们要求模型预测：

```text
世界接下来会发生什么
```

世界模型能力出现。

当我们要求模型预测：

```text
自己接下来该做什么
```

策略能力出现。

两种能力可能来自同一个预测器。

---

## 八、从边界崩塌到边界消失

李飞飞认为：

Renderer、Simulator、Planner 的边界正在崩塌。

例如：

同一个 Marble 模型同时承担：

* 渲染
* 模拟

两种职责。

---

ECHO 暗示的则是另一种崩塌。

不是：

```text
多个模块
被统一到一个模型
```

而是：

```text
多个能力
本来就是同一种学习过程
```

---

对于 Transformer：

预测世界：

```text
What Happens Next
```

预测行动：

```text
What Should I Do Next
```

可能都只是：

```text
What Token Comes Next
```

的不同表现。

---

## 九、世界是一个损失函数

于是我们终于可以重新理解标题。

上篇文章中：

「世界是一个损失函数」

更像一句口号。

现在它有了更精确的含义。

---

对于学习系统而言：

世界并不是一个对象。

世界是：

> 那些能够产生梯度的反馈。

哪里产生损失，

哪里就被模型当成现实。

---

标准 GRPO：

```text
Reality
=
Reward
```

ECHO：

```text
Reality
=
Reward
+
Environment Feedback
```

---

更进一步：

如果 Planner 与 World Model 最终只是同一个预测器的不同侧面。

那么：

世界模型未必是一个额外模块。

它可能只是：

> 把 Loss 放在世界反馈上。

---

## 十、结语

李飞飞告诉我们：

世界模型正在统一渲染、模拟与规划。

ECHO 告诉我们：

世界反馈本身就是监督信号。

而更进一步的问题是：

如果一个 Transformer 同时能够预测：

* 世界会发生什么
* 自己应该做什么

那么：

Planner 与 Simulator 的边界，

是否本来就是人为划分出来的？

---

也许未来统一世界模型真正统一的，

并不是：

```text
Renderer
+
Simulator
+
Planner
```

三个模块。

而是：

```text
一个足够强大的预测器
```

在不同损失函数的照射下，

同时学会：

```text
看见世界

理解世界

预测世界

改变世界
```

如果真是如此。

那么 ECHO 最重要的发现，

或许不是利用了被浪费的 token。

而是在无意间揭开了一件更大的事情：

> World Model 与 Policy Learning，可能从来就不是两件事。
