---
title: GPT-OSS为何“复古地”在Attention机制中配置了偏差项(bias)？
date: 2025-08-06T04:20:55.094Z
---

GPT-OSS今天发布了，在[模型架构配置参数文档](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json)中，我们可以看到其核心的注意力机制模块配置了偏差项（`attention bias`）
![GPT-OSS attention bias True.png](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2025-08-06/1754452456905.png)

而当前很多的头部大模型明确都舍弃了这个bias的配置（比如DeepSeeK、Qwen系列）
>特别是Qwen系列，本来在Qwen2.5时代还保留了这个bias的配置，但是**到了Qwen3时代也舍弃了**

正好博主去年有从数学推导的角度上，详细推理**注意力机制中配置上偏差项`bias`**是如何**有助于大模型长度外推泛化**的内在机制

故此老帖重提，为大家理解**GPT-OSS**模型中核心的注意力机制为何"复古地"**配置偏差项**（`attention bias`）提供一个解读视角

老帖链接：[再探RoPE（二）：为什么RoPE + Bias能在远程衰减和长度外推上发挥重要作用？](https://zhuanlan.zhihu.com/p/17397790476)