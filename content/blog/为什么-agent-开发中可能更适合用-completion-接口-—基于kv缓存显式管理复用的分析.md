---
title: 为什么 Agent 开发中可能更适合用 completion 接口？ —基于KV缓存显式管理（复用）的分析
date: 2025-11-27T13:00:32.624Z
---

## 一、`completion` 和`chat/completion`接口简单回顾

首先，我们先理清这两个接口的区别：

### 1) **completion接口**​（如`/v1/completions`，这是更原始的接口）

```
输入: "Once upon a time, there was a princess"
输出: "who lived in a tall tower..."
```

* 输入：一个完整的文本字符串（prompt）
* 输出：模型生成的**续写**文本
* 本质：纯粹的**文本**续写
  > 你给模型一个**完整的、连续的** token 序列，模型返回后续 tokens。

### 2) **chat/completion接口**​（如`/v1/chat/completions`，这是再抽象后的接口）

```
message: [
  {"role": "system", "content": "You are a helpful assistant"},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi!"},
  {"role": "user", "content": "How are you?"}
]
```

* 输入：一个messages**数组**，包含多轮对话历史（含`系统提示`、`用户历史请求`、`assistant的历史思考、历史答复、历史工具请求`、`历史工具返回`、`用户最新请求`等...）
* 输出：assistant的最新回复
* 本质：**对话**续写
  > 服务商在服务端将 messages 数组**转换**为实际的 prompt 字符串（通过 `chat template`），然后再做文本续写

---

## 二、从KV缓存管理（复用）的角度分析两者显著差异：

### 1)  **completion接口的优势：**

* 用户完全控制prompt的构造

> 当然，也要承担对不齐模型聊天模版的风险：使用`completion`接口需要自己对齐 chat template 进行prompt构建，如果格式对不上，模型表现会下降。

* 可以精确控制哪些部分是"前缀"，从而更好地利用prefix caching
* 服务商可以根据prompt的公共前缀进行KV缓存复用
* 用户可以设计prompt格式来最大化缓存命中

| 特性 | 说明 |
|------|------|
| **完全透明** | 你构造的 prompt 就是实际送入模型的 token 序列 |
| **精确控制** | 你可以精心设计 prompt 结构，让**可复用部分**放在前缀位置|
| **可预测性** | 相同的 prompt 前缀 → 相同的 KV Cache → 确定性复用 |

### 2)  **chat/completion接口的特点：**

* 服务商在服务端将messages数组转换为实际的prompt（套用各**角色**(`system`、`user`、`assistant`、`tool`等)标记的聊天模版，然后拼接+**可能的裁切**）
* **这个转换过程对用户是不透明的**

> 特别地，有些模型的`chat template`聊天模版，会裁切历史的消息记录（比如`reasoning_content`），会直接导致后续请求中，不仅历史KV缓存无法完整复用，更麻烦的是导致多轮的复杂智能体开发中的上下文丢失问题！（再请求时，已经丢失了上一步中已经完成的思考，信息有丢失）
> 
> - 案例： [qwen3-235b-a22b-thinking聊天模版](https://huggingface.co/spaces/huggingfacejs/chat-template-playground?modelId=Qwen%2FQwen3-VL-235B-A22B-Thinking)
> ![qwen3 聊天模板.png](https://raw.githubusercontent.com/Paul33333/tinymind-blog/main/assets/images/2025-11-27/1764247349865.png)

> ```
> # 第一次请求
> messages: [system, user1, assistant1(含reasoning、content、tool_call), ..., user2]
> → 实际prompt: "...<think>xxx</think>..."  ← 包含推理过程
> 
> # 第二次请求（假设服务商裁切了reasoning_content）
> messages: [system, user1, assistant1(不含reasoning), user2, assistant2, user3]
> → 实际prompt: "..."  ← 前缀已变，KV Cache 失效 + 历史思考丢失！
> ```

* 不同厂商的chat template不同
* 用户无法精确控制最终的prompt结构

| 问题 | 影响 |
|------|------| 
| **黑盒转换** | \`messages → actual\_prompt\` 的 chat template 对你不透明 |
| **格式依赖** | 不同厂商模板不同（ChatML、Llama-style、Claude-style...） | 
| **缓存更易不可控** | 你无法精确预测最终的 token 序列，难以设计缓存策略 | 
| **元数据开销** | 角色标记、特殊token、裁切规则会打断你精心设计的前缀连续性 |

### 3) **KV缓存复用的关键点：**

- **Prefix Caching**​：LLM推理时，如果多个请求有相同的前缀，可以**复用前缀部分的KV缓存**，避免重复计算（再推理相同前缀部分的KV），降低算力耗费和api费用。

```
│  [System Prompt] [User contetent] [Assistant content] [Tool call] [Tool_response]...│ [User new query] │
│  ←──────────────────可缓存复用的部分───────────────────────────────────────────────→ │      <新计算>      │
```

> 对于`chat/completion`接口：服务商内部会将`messages数组`转换为`prompt`，再转换为`tokenid`，最后送去模型推理（`list[dict]`->`str`->`tokenid`），这个**不透明**的过程（`list[dict]`->`str`）会影响KV缓存的复用效率。而且不同厂商可能有不同的转换策略，导致即使messages看起来相似，最终生成的实际prompt也可能存在微妙差异。这种不确定性会降低缓存命中的可能性，增加计算开销。

---

## 三、总结：便利性 VS 控制权

```
Transformer 本质：P(next_token | previous_tokens)
```

从模型角度，**根本不存在"对话"这个概念**——它只看到一个 token 序列（previous_tokens），然后预测下一个 token。

> 无论是`completion`还是`chat/completion`，最终都是将输入转换为tokenid序列，预测下一个token。这意味着接口的差异主要在于抽象层的便利性，而非底层能力。
> 
> 性能选择取决于具体场景。对于需要精确控制和最大化性能的场景，Completion接口提供了更直接的模型交互方式，允许用户精细调整prompt结构，并更有效地利用前缀缓存机制。
> 
> 相比之下，**`chat/completion`接口提供了更标准化、更简单的API**，服务商可以针对聊天场景进行专门优化（开发者不需要关心 prompt 模板）。

```
Completion:     你的 prompt ──────────→ 模型 ──→ 输出
                    ↑
                你完全控制

Chat Completion: messages ──→ [Chat Template] ──→ actual_prompt ──→ 模型 ──→ 输出
                                    ↑
                             服务商控制（黑盒）
```

从KV缓存显式管理（复用）的角度看，哪种方式更优呢？

-> 从纯效率/控制角度：确实`completion`接口更优

结论：**`chat/completion`是对`completion`的一个抽象层**，它**以便利性换取了控制权**。

而Agent 是一个“多轮推理 + 工具协作 + 长上下文”的系统，需要对状态、前缀和 KV Cache 有极强的可控性，而 Completion 才是唯一能把这种可控性完全交还给开发者的接口。

> 值得注意的是，很多服务商已经在 Chat Completion 接口上额外新增了**显式暴露**缓存控制能力（如Anthropic里的`cache_control` 参数），这是在`chat/completion`的抽象层上**补回**了`completion`接口的控制能力，说明业界也认识到了纯`chat/completion`抽象的局限性。